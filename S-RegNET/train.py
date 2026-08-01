"""
Seg-only Registration Training Script

Uses:
- Template seg + Sample seg as input (both 5-ch one-hot)
- Vanilla UNet with dual heads (flow + lambda)
- Optional AffineNet pre-alignment

Evaluation: Dice on warped template seg vs sample seg, plus per-voxel
folding diagnostics via the Jacobian determinant.

Usage:
    python train.py                    # use default config.yaml
    python train.py --config my.yaml   # use a custom config
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import numpy as np
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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

from model import SegRegistrationNet, SpatialTransformer
from losses import SegRegistrationLoss, compute_dice_score, jacobian_det
from get_data import SegDataset
from git_provenance import write_git_sha


# =============================================================================
# Configuration
# =============================================================================

def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class Config:
    """Training configuration. Loads from config.yaml by default."""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        if Path(config_path).exists():
            self._load_from_yaml(load_yaml_config(config_path))
        else:
            raise FileNotFoundError(f"config.yaml not found at {config_path}")

    def _load_from_yaml(self, cfg):
        # Paths
        self.train_txt = cfg['data']['train_txt']
        self.val_txt = cfg['data']['val_txt']
        self.template_seg_path = cfg['data']['template_seg_path']
        self.seg_filename = cfg['data'].get('seg_filename', 'seg4_onehot.npy')

        # Output
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = cfg['output']['base_dir']
        self.output_dir = f"{base_dir}/{timestamp}"
        self.checkpoint_dir = os.path.join(self.output_dir, cfg['output']['checkpoint_subdir'])
        self.log_dir = os.path.join(self.output_dir, cfg['output']['log_subdir'])

        # Model
        self.target_size = tuple(cfg['model']['target_size'])
        self.num_classes = cfg['model']['num_classes']
        self.seg_channels = cfg['model'].get('seg_channels', self.num_classes)

        # Affine
        affine_cfg = cfg.get('affine', {})
        self.use_affine = affine_cfg.get('enabled', False)

        # Training
        self.batch_size = cfg['training']['batch_size']
        self.num_epochs = cfg['training']['num_epochs']
        self.num_workers = cfg['training']['num_workers']
        self.pin_memory = cfg['training']['pin_memory']

        self.lr = cfg['training']['learning_rate']
        self.min_lr = cfg['training']['min_learning_rate']
        self.weight_decay = cfg['training']['weight_decay']
        self.warmup_epochs = cfg['training']['warmup_epochs']

        self.use_amp = cfg['training']['use_amp']

        self.save_every = cfg['training']['save_every']
        self.patience = cfg['training']['patience']
        self.max_grad_norm = cfg['training'].get('max_grad_norm', 1.0)

        # Device
        self.device = cfg['device']['gpu'] if torch.cuda.is_available() else "cpu"
        self.cudnn_benchmark = cfg['device'].get('cudnn_benchmark', True)

        # Resume
        self.resume_from = cfg['resume']['checkpoint_path']

        # Reproducibility
        self.seed = cfg.get('seed', 42)

        # Loss weights (merge affine weights into loss dict when enabled)
        self.loss_weights = cfg['loss']
        self.class_weights = self.loss_weights.pop('class_weights', None)
        if self.use_affine:
            self.loss_weights.setdefault('affine_reg', affine_cfg.get('regularization_weight', 0.01))
            self.loss_weights.setdefault('affine_ortho', affine_cfg.get('orthogonality_weight', 0.01))

    def save(self, path):
        config_dict = {k: str(v) if isinstance(v, Path) else v
                       for k, v in self.__dict__.items()}
        config_dict['target_size'] = list(self.target_size)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)


# =============================================================================
# Logger Setup
# =============================================================================

def setup_logger(log_dir, name="seg_registration"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

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
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
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
                                  weights, output_dir, logger=None,
                                  train_lambda_stats=None, val_lambda_stats=None):
    """Append per-epoch loss-component breakdown to loss_components.jsonl.

    Lambda-map diagnostics live outside the raw/weighted blocks: they aren't
    loss values, so including them in the weighted-contribution ranking
    would distort it.
    """
    def weighted_view(components):
        return {k: float(weights.get(k, 0.0)) * float(v) for k, v in components.items()}

    train_w = weighted_view(train_components)
    val_w = weighted_view(val_components)

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

    if train_lambda_stats is not None or val_lambda_stats is not None:
        record['lambda_stats'] = {
            'train': {k: float(v) for k, v in (train_lambda_stats or {}).items()},
            'val':   {k: float(v) for k, v in (val_lambda_stats or {}).items()},
        }

    path = Path(output_dir) / "loss_components.jsonl"
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')

    if logger is not None:
        total = sum(abs(v) for v in val_w.values()) or 1.0
        ranked = sorted(val_w.items(), key=lambda kv: -abs(kv[1]))[:5]
        parts = [f"{k}={v:.4g} ({100 * abs(v) / total:.1f}%)" for k, v in ranked]
        logger.info(f"  Val   - Top contributors: {' | '.join(parts)}")


def safe_save(obj, path, logger=None, timeout=300):
    """torch.save with a hard timeout so NFS hangs don't kill the run."""
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

def _warp_template(template_seg, final_flow, affine_matrix, stn):
    """
    Replay the critical invariant: affine first, then dense flow.
    Bilinear sampling is used so the dice loss has a usable gradient;
    eval converts to hard labels via argmax inside compute_dice_score.
    """
    if affine_matrix is not None:
        affine_grid = F.affine_grid(affine_matrix, template_seg.size(), align_corners=False)
        aligned_seg = F.grid_sample(template_seg, affine_grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=False)
        return stn(aligned_seg, final_flow)
    return stn(template_seg, final_flow)


def _accumulate_lambda_stats(stats_sum, lambda_map):
    """Per-batch mean/std/p05/p95 of the λ map, accumulated into stats_sum.

    Aggregating means-of-batches (not a histogram over the whole epoch)
    is the right discipline here: each batch produces one λ map, and
    batch-mean stats stay invariant to dataset size at fixed batch_size.
    """
    lam = lambda_map.detach()
    stats_sum['mean'] += lam.mean().item()
    stats_sum['std']  += lam.std().item()
    stats_sum['p05']  += lam.quantile(0.05).item()
    stats_sum['p95']  += lam.quantile(0.95).item()


def train_epoch(model, stn, dataloader, loss_fn, optimizer, scaler, device, epoch, config):
    model.train()

    total_loss = 0
    total_dice = 0
    loss_components_sum = {}
    lambda_stats_sum = {'mean': 0.0, 'std': 0.0, 'p05': 0.0, 'p95': 0.0}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs} [Train]",
                leave=False, ncols=100)

    for batch in pbar:
        template_seg = batch['template_seg'].to(device)
        sample_seg = batch['sample_seg'].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config.use_amp):
            flow_fw, flow_rv, lambda_map, affine_matrix = model(template_seg, sample_seg)
            warped_seg_fw = _warp_template(template_seg, flow_fw, affine_matrix, stn)
            warped_seg_rv = stn(sample_seg, flow_rv)

            loss, loss_dict = loss_fn(
                warped_seg_fw, sample_seg, warped_seg_rv, template_seg,
                flow_fw, flow_rv, lambda_map, stn,
                affine_matrix=affine_matrix, return_components=True,
            )

        # Backward + grad clip + step. unscale BEFORE clip — clip_grad_norm_
        # on scaled grads makes max_norm meaningless under AMP.
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        dice_score = 1 - (loss_dict['dice'].item() / 2.0)
        total_dice += dice_score

        for k, v in loss_dict.items():
            loss_components_sum[k] = loss_components_sum.get(k, 0) + (
                v.item() if isinstance(v, torch.Tensor) else v
            )
        _accumulate_lambda_stats(lambda_stats_sum, lambda_map)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice_score:.4f}'})

    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'components': {k: v / num_batches for k, v in loss_components_sum.items()},
        'lambda_stats': {k: v / num_batches for k, v in lambda_stats_sum.items()},
    }


@torch.no_grad()
def validate_epoch(model, stn, dataloader, loss_fn, device, epoch, config):
    model.eval()

    total_loss = 0
    total_dice = 0
    total_folding_pct = 0
    total_fold_voxels = 0
    worst_min_det = float('inf')
    worst_min_det_subject = -1
    all_dice_per_class = [[] for _ in range(config.num_classes)]
    loss_components_sum = {}
    lambda_stats_sum = {'mean': 0.0, 'std': 0.0, 'p05': 0.0, 'p95': 0.0}

    D, H, W = config.target_size
    det_ref = (2.0 / D) * (2.0 / H) * (2.0 / W)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs} [Val]",
                leave=False, ncols=100)

    for batch_idx, batch in enumerate(pbar):
        template_seg = batch['template_seg'].to(device)
        sample_seg = batch['sample_seg'].to(device)

        flow_fw, flow_rv, lambda_map, affine_matrix = model(template_seg, sample_seg)

        # Diffeomorphism diagnostics.
        det_fw = jacobian_det(flow_fw) / det_ref
        total_folding_pct += (det_fw < 0).float().mean().item() * 100.0
        batch_min_det = det_fw.min().item()
        total_fold_voxels += int((det_fw < 0).sum().item())
        if batch_min_det < worst_min_det:
            worst_min_det = batch_min_det
            worst_min_det_subject = batch_idx

        warped_seg_fw = _warp_template(template_seg, flow_fw, affine_matrix, stn)
        warped_seg_rv = stn(sample_seg, flow_rv)

        loss, loss_dict = loss_fn(
            warped_seg_fw, sample_seg, warped_seg_rv, template_seg,
            flow_fw, flow_rv, lambda_map, stn,
            affine_matrix=affine_matrix, return_components=True,
        )

        total_loss += loss.item()
        # Since dice is symmetric, we can average them for reporting, or just use loss_dict['dice']/2.
        dice_score = 1 - (loss_dict['dice'].item() / 2.0)
        total_dice += dice_score

        dice_per_class, _ = compute_dice_score(warped_seg_fw, sample_seg, config.num_classes)
        for c in range(config.num_classes):
            all_dice_per_class[c].append(dice_per_class[c])

        for k, v in loss_dict.items():
            loss_components_sum[k] = loss_components_sum.get(k, 0) + (
                v.item() if isinstance(v, torch.Tensor) else v
            )
        _accumulate_lambda_stats(lambda_stats_sum, lambda_map)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice_score:.4f}'})

    num_batches = len(dataloader)
    avg_dice_per_class = [np.mean(scores) for scores in all_dice_per_class]

    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'dice_per_class': avg_dice_per_class,
        'folding_pct': total_folding_pct / num_batches,
        'fold_voxels_total': total_fold_voxels,
        'worst_min_det': worst_min_det,
        'worst_min_det_subject': worst_min_det_subject,
        'components': {k: v / num_batches for k, v in loss_components_sum.items()},
        'lambda_stats': {k: v / num_batches for k, v in lambda_stats_sum.items()},
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Seg-only Registration Training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (default: config.yaml in script directory)')
    args = parser.parse_args()

    config = Config(args.config)

    # Reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = not config.cudnn_benchmark

    for dir_path in [config.output_dir, config.checkpoint_dir, config.log_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    logger, log_file = setup_logger(config.log_dir)

    logger.info("=" * 70)
    logger.info("Seg-only Registration (vanilla UNet + lambda head)")
    logger.info("=" * 70)
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Device: {config.device}")

    write_git_sha(config.output_dir, logger)
    config.save(os.path.join(config.output_dir, "config.json"))

    device = torch.device(config.device)
    if 'cuda' in config.device:
        logger.info(f"GPU: {torch.cuda.get_device_name()}")

    # ==========================================================================
    # Data Loading
    # ==========================================================================
    logger.info("Loading datasets...")

    train_dataset = SegDataset(
        config.train_txt, config.template_seg_path,
        target_size=config.target_size,
        seg_filename=config.seg_filename,
    )
    val_dataset = SegDataset(
        config.val_txt, config.template_seg_path,
        target_size=config.target_size,
        seg_filename=config.seg_filename,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
    )

    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ==========================================================================
    # Model Setup
    # ==========================================================================
    logger.info("Initializing model...")

    model = SegRegistrationNet(
        target_size=config.target_size,
        seg_channels=config.seg_channels, use_affine=config.use_affine,
    ).to(device)
    stn = SpatialTransformer(size=config.target_size, device=device).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Affine pre-alignment: {'ENABLED' if config.use_affine else 'DISABLED'}")

    loss_fn = SegRegistrationLoss(
        weights=config.loss_weights,
        class_weights=config.class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer, config.warmup_epochs, config.num_epochs, config.min_lr,
    )
    scaler = GradScaler(enabled=config.use_amp)

    logger.info(f"Optimizer: AdamW (lr={config.lr}, weight_decay={config.weight_decay})")
    logger.info(f"Scheduler: Cosine with {config.warmup_epochs} warmup epochs")

    # Resume
    start_epoch = 1
    best_dice = 0.0

    if config.resume_from and os.path.exists(config.resume_from):
        logger.info(f"Resuming from: {config.resume_from}")
        # weights_only=False: our checkpoints contain numpy scalars in metrics.
        checkpoint = torch.load(config.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
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

        current_lr = scheduler.step(epoch - 1)

        train_metrics = train_epoch(
            model, stn, train_loader, loss_fn, optimizer, scaler, device, epoch, config,
        )
        val_metrics = validate_epoch(
            model, stn, val_loader, loss_fn, device, epoch, config,
        )

        epoch_time = time.time() - epoch_start

        logger.info("-" * 70)
        logger.info(f"Epoch {epoch}/{config.num_epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.5f} | Dice: {train_metrics['dice']:.4f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.5f} | Dice: {val_metrics['dice']:.4f}")
        logger.info(
            f"  Val   - Folding: {val_metrics['folding_pct']:.3f}% "
            f"({val_metrics['fold_voxels_total']:,} voxels total) | "
            f"Worst voxel: {val_metrics['worst_min_det']:.3f} "
            f"(subject idx {val_metrics['worst_min_det_subject']})"
        )
        ls = val_metrics['lambda_stats']
        logger.info(
            f"  Val   - Lambda: mean={ls['mean']:.3f} std={ls['std']:.3f} "
            f"| p05={ls['p05']:.3f} p95={ls['p95']:.3f} "
            f"(collapse signal: std→0 and p95-p05→0)"
        )

        log_loss_components_artifact(
            epoch,
            train_metrics['components'], val_metrics['components'],
            loss_fn.weights, config.output_dir, logger,
            train_lambda_stats=train_metrics['lambda_stats'],
            val_lambda_stats=val_metrics['lambda_stats'],
        )

        if epoch % 5 == 0:
            dice_str = " | ".join([f"C{i}: {d:.4f}" for i, d in enumerate(val_metrics['dice_per_class'])])
            logger.info(f"  Per-class Dice: {dice_str}")

        is_best = val_metrics['dice'] > best_dice
        if is_best:
            best_dice = val_metrics['dice']
            epochs_without_improvement = 0
            logger.info(f"  New best model! Dice: {best_dice:.4f}")
        else:
            epochs_without_improvement += 1

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
                safe_save(
                    checkpoint,
                    os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth"),
                    logger,
                )

        if epochs_without_improvement >= config.patience:
            logger.warning(f"Early stopping! No improvement for {config.patience} epochs.")
            break

        if 'cuda' in config.device:
            torch.cuda.empty_cache()

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
