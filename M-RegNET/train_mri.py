"""
MRI-based Registration Training Script

Uses:
- Template MRI + Template Segmentation + Sample MRI as input
- Anatomical Correction Module for structure-aware deformation
- Segmentation Attention Module for boundary-aware features
- Multi-scale progressive refinement

Evaluation: Dice score on warped template segmentation vs sample segmentation

Usage:
    python train_mri.py                    # Use default config.yaml
    python train_mri.py --config my.yaml   # Use custom config file
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import numpy as np
import os
import sys
import json
import logging
import time
import signal
from contextlib import contextmanager
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Local imports
from model_mri import MRIRegistrationNet
from losses_mri import MRIRegistrationLoss, compute_dice_score, jacobian_det
from get_data_mri import MRIDataset
from cascade_utils import run_cascade_forward


from git_provenance import write_git_sha


# =============================================================================
# Configuration
# =============================================================================

def load_yaml_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class Config:
    """Training configuration. Loads from config.yaml by default."""
    
    def __init__(self, config_path=None):
        # Load from YAML if provided
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        if not Path(config_path).exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                "config.yaml is the single source of truth — no defaults fallback."
            )
        yaml_config = load_yaml_config(config_path)
        self._load_from_yaml(yaml_config)
    
    def _load_from_yaml(self, cfg):
        """Load configuration from parsed YAML dict."""
        # Paths
        self.train_txt = cfg['data']['train_txt']
        self.val_txt = cfg['data']['val_txt']
        self.template_mri_path = cfg['data']['template_mri_path']
        self.template_seg_path = cfg['data']['template_seg_path']

        # Output
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = cfg['output']['base_dir']
        self.output_dir = f"{base_dir}/{timestamp}"
        self.checkpoint_dir = os.path.join(self.output_dir, cfg['output']['checkpoint_subdir'])
        self.log_dir = os.path.join(self.output_dir, cfg['output']['log_subdir'])

        # Model
        self.target_size = tuple(cfg['model']['target_size'])
        self.num_classes = cfg['model']['num_classes']

        # Affine pre-alignment
        affine_cfg = cfg.get('affine', {})
        self.use_affine = affine_cfg.get('enabled', False)

        # Anatomical Correction Module
        self.use_acm = cfg.get('acm', {}).get('enabled', True)

        # Flow parameterization (SVF vs displacement)
        flow_cfg = cfg.get('flow', {})
        self.flow_parameterization = flow_cfg.get('parameterization', 'displacement')
        self.integration_steps = int(flow_cfg.get('integration_steps', 7))
        self.velocity_bound = float(flow_cfg.get('velocity_bound', 0.0))
        self.velocity_field = flow_cfg.get('velocity_field', 'dense')
        self.cp_spacing = int(flow_cfg.get('cp_spacing', 4))

        # Cascade
        cascade_cfg = cfg.get('cascade', {})
        self.num_stages = int(cascade_cfg.get('num_stages', 1))

        # Training
        self.batch_size = cfg['training']['batch_size']
        self.num_epochs = cfg['training']['num_epochs']
        self.num_workers = cfg['training']['num_workers']
        self.pin_memory = cfg['training']['pin_memory']

        # Spatial augmentation (sample only)
        self.spatial_aug_config = cfg.get('spatial_augmentation', {'enabled': False})

        # Optimizer
        self.lr = cfg['training']['learning_rate']
        self.min_lr = cfg['training']['min_learning_rate']
        self.weight_decay = cfg['training']['weight_decay']
        self.warmup_epochs = cfg['training']['warmup_epochs']

        # AMP
        self.use_amp = cfg['training']['use_amp']

        # Checkpointing
        self.save_every = cfg['training']['save_every']
        self.patience = cfg['training']['patience']

        # Device
        self.device = cfg['device']['gpu'] if torch.cuda.is_available() else "cpu"
        self.cudnn_benchmark = cfg['device'].get('cudnn_benchmark', True)

        # Resume
        self.resume_from = cfg['resume']['checkpoint_path']

        # Reproducibility
        self.seed = cfg.get('seed', 42)

        # Loss weights (merge affine weights from config)
        self.loss_weights = cfg['loss']
        if self.use_affine:
            self.loss_weights.setdefault('affine_reg', affine_cfg.get('regularization_weight', 0.01))
            self.loss_weights.setdefault('affine_ortho', affine_cfg.get('orthogonality_weight', 0.01))

    def save(self, path):
        """Save config to JSON."""
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                       for k, v in self.__dict__.items()}
        config_dict['target_size'] = list(self.target_size)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)


# =============================================================================
# Logger Setup
# =============================================================================

def setup_logger(log_dir, name="mri_registration"):
    """Setup file and console logging."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file


# =============================================================================
# Learning Rate Scheduler with Warmup
# =============================================================================

class WarmupCosineScheduler:
    """Cosine annealing with linear warmup."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


# =============================================================================
# NFS-safe Checkpoint Save
# =============================================================================

@contextmanager
def _timeout(seconds):
    def _handler(signum, frame):
        raise TimeoutError(f"I/O timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def log_loss_components_artifact(epoch, train_components, val_components,
                                  weights, output_dir, logger=None):
    """
    Append per-epoch loss-component breakdown to loss_components.jsonl.

    For each term we record:
      raw      = unweighted loss value (mean over batches)
      weighted = weight * raw  (actual contribution to the total loss)

    Storing both means later analyses survive weight retuning.
    """
    def weighted_view(components):
        return {k: float(weights.get(k, 0.0)) * float(v) for k, v in components.items()}

    train_w = weighted_view(train_components)
    val_w   = weighted_view(val_components)

    record = {
        'epoch': epoch,
        'weights': {k: float(v) for k, v in weights.items()},
        'train': {
            'raw':      {k: float(v) for k, v in train_components.items()},
            'weighted': train_w,
        },
        'val': {
            'raw':      {k: float(v) for k, v in val_components.items()},
            'weighted': val_w,
        },
    }

    path = Path(output_dir) / "loss_components.jsonl"
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')

    # Console: top 5 contributors on val, with share of total |weighted|.
    if logger is not None:
        total = sum(abs(v) for v in val_w.values()) or 1.0
        ranked = sorted(val_w.items(), key=lambda kv: -abs(kv[1]))[:5]
        parts = [f"{k}={v:.4g} ({100*abs(v)/total:.1f}%)" for k, v in ranked]
        logger.info(f"  Val   - Top contributors: {' | '.join(parts)}")


def safe_save(obj, path, logger=None, timeout=300):
    """torch.save with a hard timeout to survive NFS hangs."""
    try:
        with _timeout(timeout):
            torch.save(obj, path)
    except TimeoutError as e:
        if logger:
            logger.warning(f"Checkpoint save timed out ({path}): {e} — skipping")
    except Exception as e:
        if logger:
            logger.warning(f"Checkpoint save failed ({path}): {e} — skipping")


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, dataloader, loss_fn, optimizer, scaler, device, epoch, config):
    """Train for one epoch with tqdm progress bar."""
    model.train()

    total_loss = 0
    total_dice = 0
    loss_components_sum = {}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs} [Train]",
                leave=False, ncols=100)

    for batch in pbar:
        # Move data to device
        template_mri = batch['template_mri'].to(device)
        template_seg = batch['template_seg'].to(device)
        sample_mri = batch['sample_mri'].to(device)
        sample_seg = batch['sample_seg'].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config.use_amp):
            out = run_cascade_forward(
                model, model.stn_image, model.stn_flow,
                template_mri, template_seg, sample_mri,
                num_stages=config.num_stages,
                n_integration_steps=config.integration_steps,
                flow_parameterization=config.flow_parameterization,
            )

            # Compute loss
            loss, loss_dict = loss_fn(
                out['warped_mri'], sample_mri, out['warped_seg'], sample_seg,
                out['final_flow'], out['intermediate_velocities'], out['lambda_maps'],
                affine_matrix=out['affine_matrix'],
                final_velocity=out['final_velocity'],
                return_components=True,
            )

        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Track metrics
        total_loss += loss.item()
        dice_score = 1 - loss_dict['dice'].item()
        total_dice += dice_score
        
        for k, v in loss_dict.items():
            if k not in loss_components_sum:
                loss_components_sum[k] = 0
            loss_components_sum[k] += v.item() if isinstance(v, torch.Tensor) else v
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice_score:.4f}'})
    
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'components': {k: v / num_batches for k, v in loss_components_sum.items()}
    }


@torch.no_grad()
def validate_epoch(model, dataloader, loss_fn, device, epoch, config):
    """Validate for one epoch with tqdm progress bar."""
    model.eval()

    total_loss = 0
    total_dice = 0
    total_folding_pct = 0
    total_interior_folding_pct = 0
    total_boundary_folding_pct = 0
    total_fold_voxels = 0       # integer count of folded voxels (sum over val set)
    worst_min_det = float('inf')  # worst single normalised det; +inf so "no folds" reads as the smallest positive det rather than ambiguous 0.0
    worst_min_det_subject = -1   # subject index that produced it
    all_dice_per_class = [[] for _ in range(config.num_classes)]
    loss_components_sum = {}

    # Diagnostic stats.
    total_lambda_mean = 0.0
    total_lambda_std = 0.0
    total_flow_mean = 0.0
    total_flow_max = 0.0
    # SVF kill-switch monitor (deep dive §12.5, postmortem §2 'train script diagnostics').
    # Track P50/P99/P99.9 of |v| and the max P99.9 across the val set. P99.9 > 2.0
    # warns runaway velocity that will eventually break the Lipschitz bound for
    # scaling-and-squaring at N=7 (budget 2^N=128 on ||J_v||_op).
    velocity_p999_max = 0.0
    total_velocity_p50 = 0.0
    total_velocity_p99 = 0.0
    total_velocity_p999 = 0.0

    D, H, W = config.target_size
    det_ref = (2.0 / D) * (2.0 / H) * (2.0 / W)

    # 5-voxel boundary band used to split folding into interior vs boundary.
    # Sustained `boundary > 1.5 × interior` means the flow STN padding-mode is
    # wrong or the model is producing edge-streak artifacts. Real folds bias
    # toward the interior because the brain mask is concentrated there.
    boundary_band = 5

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs} [Val]",
                leave=False, ncols=100)

    for batch_idx, batch in enumerate(pbar):
        template_mri = batch['template_mri'].to(device)
        template_seg = batch['template_seg'].to(device)
        sample_mri = batch['sample_mri'].to(device)
        sample_seg = batch['sample_seg'].to(device)

        out = run_cascade_forward(
            model, model.stn_image, model.stn_flow,
            template_mri, template_seg, sample_mri,
            num_stages=config.num_stages,
            n_integration_steps=config.integration_steps,
            flow_parameterization=config.flow_parameterization,
        )
        final_flow = out['final_flow']
        final_velocity = out['final_velocity']
        intermediate_velocities = out['intermediate_velocities']
        lambda_maps = out['lambda_maps']
        affine_matrix = out['affine_matrix']
        warped_mri = out['warped_mri']
        warped_seg = out['warped_seg']

        # Diffeomorphism diagnostics: folding % and worst det at this epoch.
        det = jacobian_det(final_flow) / det_ref
        fold_mask = (det < 0)
        total_folding_pct += fold_mask.float().mean().item() * 100.0
        batch_min_det = det.min().item()
        total_fold_voxels += int(fold_mask.sum().item())
        if batch_min_det < worst_min_det:
            worst_min_det = batch_min_det
            worst_min_det_subject = batch_idx

        # Boundary vs interior split.
        b = boundary_band
        # interior mask: the central cube excluding `b` voxels on each face
        interior_slice = fold_mask[..., b:-b, b:-b, b:-b] if b > 0 else fold_mask
        # boundary voxels = total folded − interior folded; normalise by their own counts
        interior_count = interior_slice.numel()
        full_count = fold_mask.numel()
        boundary_count = full_count - interior_count
        n_interior_folds = int(interior_slice.sum().item())
        n_full_folds = int(fold_mask.sum().item())
        n_boundary_folds = n_full_folds - n_interior_folds
        if interior_count > 0:
            total_interior_folding_pct += 100.0 * n_interior_folds / interior_count
        if boundary_count > 0:
            total_boundary_folding_pct += 100.0 * n_boundary_folds / boundary_count

        # Lambda + flow shape diagnostics.
        if lambda_maps is not None and len(lambda_maps) > 0:
            final_lambda = lambda_maps[-1]
            total_lambda_mean += final_lambda.mean().item()
            total_lambda_std += final_lambda.std().item()
        total_flow_mean += final_flow.abs().mean().item()
        total_flow_max += final_flow.abs().max().item()

        # Velocity P50/P99/P99.9 (SVF mode only — final_velocity is None in displacement mode).
        if final_velocity is not None:
            v_abs = final_velocity.abs().flatten()
            v_p50 = torch.quantile(v_abs, 0.5).item()
            v_p99 = torch.quantile(v_abs, 0.99).item()
            v_p999 = torch.quantile(v_abs, 0.999).item()
            total_velocity_p50 += v_p50
            total_velocity_p99 += v_p99
            total_velocity_p999 += v_p999
            if v_p999 > velocity_p999_max:
                velocity_p999_max = v_p999

        # Compute loss
        loss, loss_dict = loss_fn(
            warped_mri, sample_mri, warped_seg, sample_seg,
            final_flow, intermediate_velocities, lambda_maps,
            affine_matrix=affine_matrix,
            final_velocity=final_velocity,
            return_components=True,
        )

        # Track metrics
        total_loss += loss.item()
        dice_score = 1 - loss_dict['dice'].item()
        total_dice += dice_score

        # Per-class dice
        dice_per_class, _ = compute_dice_score(warped_seg, sample_seg, config.num_classes)
        for c in range(config.num_classes):
            all_dice_per_class[c].append(dice_per_class[c])

        for k, v in loss_dict.items():
            if k not in loss_components_sum:
                loss_components_sum[k] = 0
            loss_components_sum[k] += v.item() if isinstance(v, torch.Tensor) else v

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice_score:.4f}'})

    num_batches = len(dataloader)
    avg_dice_per_class = [np.mean(scores) for scores in all_dice_per_class]

    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'dice_per_class': avg_dice_per_class,
        'folding_pct': total_folding_pct / num_batches,
        'interior_folding_pct': total_interior_folding_pct / num_batches,
        'boundary_folding_pct': total_boundary_folding_pct / num_batches,
        'fold_voxels_total': total_fold_voxels,
        'worst_min_det': worst_min_det,
        'worst_min_det_subject': worst_min_det_subject,
        'lambda_mean': total_lambda_mean / num_batches,
        'lambda_std': total_lambda_std / num_batches,
        'flow_abs_mean': total_flow_mean / num_batches,
        'flow_abs_max': total_flow_max / num_batches,
        'velocity_p50': total_velocity_p50 / num_batches,
        'velocity_p99': total_velocity_p99 / num_batches,
        'velocity_p999': total_velocity_p999 / num_batches,
        'velocity_p999_max': velocity_p999_max,
        'components': {k: v / num_batches for k, v in loss_components_sum.items()}
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MRI Registration Training')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml (default: config.yaml in script directory)')
    args = parser.parse_args()
    
    # Configuration
    config = Config(args.config)

    # Reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = not config.cudnn_benchmark

    # Create directories
    for dir_path in [config.output_dir, config.checkpoint_dir, config.log_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger, log_file = setup_logger(config.log_dir)
    
    logger.info("=" * 70)
    logger.info("MRI-based Registration with Anatomical Correction Module")
    logger.info("=" * 70)
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Device: {config.device}")

    # Git provenance — GIT_SHA.txt in the run folder + git_* keys in config.json.
    prov = write_git_sha(config.output_dir, logger)
    config.git_sha    = prov['git_sha']
    config.git_branch = prov['git_branch']
    config.git_dirty  = prov['git_dirty']

    # Save config (now carries git provenance via config.git_* attributes)
    config.save(os.path.join(config.output_dir, "config.json"))
    
    # Device
    device = torch.device(config.device)
    if 'cuda' in config.device:
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
    
    # ==========================================================================
    # Data Loading
    # ==========================================================================
    logger.info("Loading datasets...")
    
    # Training dataset with spatial augmentation (if enabled in config)
    train_dataset = MRIDataset(
        config.train_txt, config.template_mri_path, config.template_seg_path,
        target_size=config.target_size,
        spatial_aug_config=config.spatial_aug_config,
    )
    # Validation dataset without augmentation for consistent evaluation
    val_dataset = MRIDataset(
        config.val_txt, config.template_mri_path, config.template_seg_path,
        target_size=config.target_size,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    
    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    if config.spatial_aug_config.get('enabled', False):
        sa = config.spatial_aug_config
        logger.info(
            f"Spatial augmentation: ENABLED for training (sample only) — "
            f"rot ±{sa.get('rotation_deg', 10.0)}°, "
            f"trans ±{sa.get('translation_voxels', 10)} vox, "
            f"elastic mag {sa.get('elastic_magnitude_voxels', 3.0)} vox"
        )
    else:
        logger.info("Spatial augmentation: DISABLED")

    # ==========================================================================
    # Model Setup
    # ==========================================================================
    logger.info("Initializing model...")
    
    model = MRIRegistrationNet(
        seg_channels=config.num_classes,
        use_affine=config.use_affine,
        use_acm=config.use_acm,
        target_size=config.target_size,
        flow_parameterization=config.flow_parameterization,
        velocity_bound=config.velocity_bound,
        velocity_field=config.velocity_field,
        cp_spacing=config.cp_spacing,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Affine pre-alignment: {'ENABLED' if config.use_affine else 'DISABLED'}")
    logger.info(f"ACM: {'ENABLED' if config.use_acm else 'DISABLED'}")
    logger.info(
        f"Flow parameterization: {config.flow_parameterization.upper()} "
        f"(integration_steps={config.integration_steps}, velocity_bound={config.velocity_bound})"
    )
    if config.velocity_field == 'bspline':
        logger.info(f"Velocity field: B-SPLINE (cp_spacing={config.cp_spacing})")
    logger.info(f"Cascade num_stages: {config.num_stages}")

    # Loss function (use weights from config if available)
    loss_fn = MRIRegistrationLoss(weights=config.loss_weights)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    
    # Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer, config.warmup_epochs, config.num_epochs, config.min_lr
    )
    
    # AMP scaler
    scaler = GradScaler(enabled=config.use_amp)
    
    logger.info(f"Optimizer: AdamW (lr={config.lr}, weight_decay={config.weight_decay})")
    logger.info(f"Scheduler: Cosine with {config.warmup_epochs} warmup epochs")
    
    # Resume from checkpoint
    start_epoch = 1
    best_dice = 0.0
    
    if config.resume_from and os.path.exists(config.resume_from):
        logger.info(f"Resuming from: {config.resume_from}")
        # weights_only=False: PyTorch 2.6 flipped the default; our checkpoints
        # contain numpy scalars in the metrics fields.
        checkpoint = torch.load(config.resume_from, map_location=device, weights_only=False)
        # strict=False — old checkpoints lack `stn_image.id_grid` / `stn_flow.id_grid`
        # buffers added when the STNs were moved onto MRIRegistrationNet, and
        # earlier SVF-less checkpoints lack the velocity-mode decoder heads
        # being identical-shape (they are, just used differently). The id_grid
        # buffers are recomputed from `target_size`.
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing or unexpected:
            logger.warning(
                f"load_state_dict non-strict: missing={missing} unexpected={unexpected}"
            )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        logger.info(f"Resumed from epoch {start_epoch - 1}, best dice: {best_dice:.4f}")
    
    # ==========================================================================
    # Training Loop
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)
    
    epochs_without_improvement = 0
    training_start = time.time()
    
    for epoch in range(start_epoch, config.num_epochs + 1):
        epoch_start = time.time()
        
        # Update learning rate
        current_lr = scheduler.step(epoch - 1)

        # Train
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, epoch, config
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, loss_fn, device, epoch, config
        )
        
        epoch_time = time.time() - epoch_start
        
        # Logging
        logger.info("-" * 70)
        logger.info(f"Epoch {epoch}/{config.num_epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.5f} | Dice: {train_metrics['dice']:.4f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.5f} | Dice: {val_metrics['dice']:.4f}")
        logger.info(
            f"  Val   - Folding: {val_metrics['folding_pct']:.3f}% "
            f"(interior {val_metrics['interior_folding_pct']:.3f}% | "
            f"boundary {val_metrics['boundary_folding_pct']:.3f}%) | "
            f"{val_metrics['fold_voxels_total']:,} voxels | "
            f"Worst voxel: {val_metrics['worst_min_det']:.3f} "
            f"(subject idx {val_metrics['worst_min_det_subject']})"
        )
        # Warn only on real anomalies — sustained boundary >> interior means
        # the flow STN padding-mode is wrong (deep dive §10.1) or the model
        # is producing edge-streak artifacts. Single-epoch outliers are OK.
        if (val_metrics['boundary_folding_pct'] >
                max(0.5, 1.5 * val_metrics['interior_folding_pct'])):
            logger.warning(
                f"  ⚠️  Boundary folding ({val_metrics['boundary_folding_pct']:.3f}%) > "
                f"1.5× interior ({val_metrics['interior_folding_pct']:.3f}%) — "
                f"check stn_flow padding_mode and edge artifacts."
            )

        logger.info(
            f"  Val   - λ mean={val_metrics['lambda_mean']:.3f} std={val_metrics['lambda_std']:.3f} | "
            f"flow |mean|={val_metrics['flow_abs_mean']:.4f} |max|={val_metrics['flow_abs_max']:.4f}"
        )
        if config.flow_parameterization == 'svf':
            logger.info(
                f"  Val   - velocity P50={val_metrics['velocity_p50']:.4f} "
                f"P99={val_metrics['velocity_p99']:.4f} "
                f"P99.9={val_metrics['velocity_p999']:.4f} "
                f"(max across val P99.9={val_metrics['velocity_p999_max']:.4f}, "
                f"bound B={config.velocity_bound})"
            )
            # Kill switch: P99.9 > 2.0 indicates runaway velocity. With N=7
            # integration steps the Lipschitz budget on ||J_v||_op is 2^7=128;
            # P99.9 > 2.0 on |v| itself is a leading indicator of the gradient
            # spikes that eventually break that budget.
            if val_metrics['velocity_p999_max'] > 2.0:
                logger.warning(
                    f"  ⚠️  velocity P99.9 max = {val_metrics['velocity_p999_max']:.4f} > 2.0 — "
                    f"runaway velocity; SVF integration will eventually fold."
                )

        # Persist per-term loss contributions to loss_components.jsonl.
        log_loss_components_artifact(
            epoch,
            train_metrics['components'], val_metrics['components'],
            loss_fn.weights, config.output_dir, logger,
        )

        # Per-class dice (every 5 epochs)
        if epoch % 5 == 0:
            dice_str = " | ".join([f"C{i}: {d:.4f}" for i, d in enumerate(val_metrics['dice_per_class'])])
            logger.info(f"  Per-class Dice: {dice_str}")

        # Check for improvement
        is_best = val_metrics['dice'] > best_dice
        if is_best:
            best_dice = val_metrics['dice']
            epochs_without_improvement = 0
            logger.info(f"  🔥 New best model! Dice: {best_dice:.4f}")
        else:
            epochs_without_improvement += 1
        
        # Save checkpoint
        if is_best or epoch % config.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_dice': val_metrics['dice'],
                'best_dice': best_dice,
            }
            
            if is_best:
                safe_save(checkpoint, os.path.join(config.checkpoint_dir, "best_model.pth"), logger)

            if epoch % config.save_every == 0:
                safe_save(checkpoint, os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth"), logger)
        
        # Early stopping
        if epochs_without_improvement >= config.patience:
            logger.warning(f"Early stopping! No improvement for {config.patience} epochs.")
            break
        
        # Clear cache
        if 'cuda' in config.device:
            torch.cuda.empty_cache()
    
    # ==========================================================================
    # Training Complete
    # ==========================================================================
    total_time = time.time() - training_start
    
    logger.info("=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time / 3600:.2f} hours")
    logger.info(f"Best validation Dice: {best_dice:.4f}")
    logger.info(f"Checkpoints: {config.checkpoint_dir}")
    logger.info(f"Logs: {log_file}")


if __name__ == "__main__":
    main()
