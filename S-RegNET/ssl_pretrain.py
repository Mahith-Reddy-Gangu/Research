"""
Self-Supervised Pretraining via Equivariant Warp Regression

Pretrains S-RegNET by generating random synthetic deformations, applying
them to segmentation volumes, and training the network to recover the
known velocity field — no segmentation labels needed.

The primary supervision signal is velocity_mse_loss against the known
ground-truth synthetic velocity (per Prof. Zeng's directive: "you already
have the known transformation parameters, why not use them for loss
function?").  Auxiliary terms include reconstruction Dice/CE and the
standard geometric regularisers.

After pretraining, the checkpoint is consumed by the supervised train.py
via::

    resume:
      checkpoint_path: "<ssl_run>/checkpoints/best_model.pth"
      weights_only: true

which loads the pretrained weights into a fresh optimizer for fine-tuning
on labelled data.

Usage:
    python ssl_pretrain.py                          # default config_ssl.yaml
    python ssl_pretrain.py --config config_ssl.yaml
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler
import numpy as np
import os
import sys
import json
import logging
import time
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from model import SegRegistrationNet, SpatialTransformer
from losses import (SegRegistrationLoss, compute_dice_score,
                    jacobian_det)
from ssl_warp_generator import generate_synthetic_warp, warp_seg_nearest
from git_provenance import write_git_sha

# Utilities shared with the supervised training loop.  These are
# module-level functions in train.py (not __main__-guarded), so import
# is safe and avoids code duplication.
from train import (WarmupCosineScheduler, safe_save, setup_logger,
                   log_loss_components_artifact, _warp_template,
                   _warp_sample_inverse, _accumulate_lambda_stats,
                   load_yaml_config)


# =============================================================================
# Configuration
# =============================================================================

class SSLConfig:
    """SSL pretraining configuration.  Loads from config_ssl.yaml.

    The model section MUST match the supervised config.yaml exactly —
    target_size, num_classes, seg_channels, head, lowpass_size — otherwise
    the pretrained weights cannot be loaded via resume.weights_only.
    """

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent / "config_ssl.yaml"
        if not Path(config_path).exists():
            raise FileNotFoundError(f"SSL config not found: {config_path}")
        self._load_from_yaml(load_yaml_config(config_path))

    def _load_from_yaml(self, cfg):
        # Data
        self.train_txt = cfg['data']['train_txt']
        self.val_txt = cfg['data']['val_txt']
        self.seg_filename = cfg['data'].get('seg_filename', 'seg4_onehot.npy')

        # Output
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = cfg['output']['base_dir']
        self.output_dir = f"{base_dir}/{timestamp}"
        self.checkpoint_dir = os.path.join(
            self.output_dir, cfg['output']['checkpoint_subdir'])
        self.log_dir = os.path.join(
            self.output_dir, cfg['output']['log_subdir'])

        # Model (must match supervised config exactly)
        self.target_size = tuple(cfg['model']['target_size'])
        self.num_classes = cfg['model']['num_classes']
        self.seg_channels = cfg['model'].get('seg_channels', self.num_classes)
        self.head = cfg['model'].get('head', 'default')
        self.lowpass_size = cfg['model'].get('lowpass_size')

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

        # SSL-specific
        ssl_cfg = cfg.get('ssl', {})
        self.coarse_sizes = tuple(ssl_cfg.get('coarse_sizes', [16, 24, 32]))
        self.magnitude_range = tuple(
            ssl_cfg.get('magnitude_range', [0.02, 0.08]))
        self.num_warps = ssl_cfg.get('num_warps', 8)
        # Freeze the affine branch during SSL so the velocity MSE target
        # is exact: synthetic warps have no affine component, so the UNet
        # velocity should match v_synth without any affine compensation.
        # The affine net is small (64 FC params) and converges quickly
        # during supervised fine-tuning.
        self.freeze_affine = ssl_cfg.get('freeze_affine', True)

        # Device — validate the configured GPU ordinal actually exists.
        # CUDA_VISIBLE_DEVICES remaps physical GPUs: e.g. setting it to "7"
        # makes that GPU appear as cuda:0 to PyTorch.  A config that says
        # cuda:1 will crash with "invalid device ordinal" in that case.
        self.cudnn_benchmark = cfg['device'].get('cudnn_benchmark', True)
        requested = cfg['device']['gpu']
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            # Extract ordinal from e.g. "cuda:1"
            try:
                ordinal = int(requested.split(':')[1]) if ':' in requested else 0
            except (ValueError, IndexError):
                ordinal = 0
            if ordinal >= n_gpus:
                self.device = "cuda:0"
            else:
                self.device = requested
        else:
            self.device = "cpu"

        # Resume (for continuing a broken SSL run, NOT supervised→SSL)
        resume_cfg = cfg.get('resume', {})
        self.resume_from = resume_cfg.get('checkpoint_path')

        # Reproducibility
        self.seed = cfg.get('seed', 42)

        # Loss weights
        self.loss_weights = dict(cfg['loss'])
        self.class_weights = self.loss_weights.pop('class_weights', None)
        if self.use_affine:
            self.loss_weights.setdefault(
                'affine_reg',
                affine_cfg.get('regularization_weight', 0.01))
            self.loss_weights.setdefault(
                'affine_ortho',
                affine_cfg.get('orthogonality_weight', 0.0))

    def save(self, path):
        config_dict = {k: str(v) if isinstance(v, Path) else v
                       for k, v in self.__dict__.items()}
        config_dict['target_size'] = list(self.target_size)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)


# =============================================================================
# SSL Dataset
# =============================================================================

PRELOAD_WORKERS = 12


class SSLSegDataset(Dataset):
    """Minimal dataset for SSL pretraining: loads individual segmentations.

    Each sample returns a single seg volume.  The training loop generates
    synthetic warps on-the-fly and constructs the (template, sample) pair
    from it — no GT labels, no target velocities, no SDFs.

    Preloads all segs into RAM as uint8 (one-hot is lossless at uint8)
    to avoid per-epoch NFS I/O, exactly like SegDataset.
    """

    def __init__(self, data_list_file, target_size, seg_filename,
                 preload=True, num_warps=8):
        """
        Args:
            data_list_file: text file, one path per line; each points at a
                subject's seg .npy.  Subject dir is inferred via dirname.
            target_size: (D, H, W) resize target.
            seg_filename: seg file inside each subject dir (e.g.
                'seg4_white_onehot.npy' or 'synthseg_white_onehot_v1.npy').
            preload: if True, cache all segs in RAM at init.
        """
        with open(data_list_file, 'r') as f:
            self.subject_dirs = [os.path.dirname(p) for p in
                                 f.read().splitlines() if p.strip()]
        self.target_size = target_size
        self.seg_filename = seg_filename
        self.num_warps = num_warps

        # Fail fast: missing seg files.
        missing = [d for d in self.subject_dirs
                   if not os.path.exists(os.path.join(d, seg_filename))]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)}/{len(self.subject_dirs)} subjects have no "
                f"{seg_filename} (first: {missing[:3]})")

        if preload:
            self._preload_all()

    def _load_seg(self, path):
        """Load and resize a one-hot seg volume."""
        seg = np.load(path)
        seg = torch.tensor(seg, dtype=torch.float32)
        seg = F.interpolate(
            seg.unsqueeze(0), size=self.target_size, mode='nearest'
        ).squeeze(0)
        return seg

    def _preload_one(self, subject_dir):
        path = os.path.join(subject_dir, self.seg_filename)
        return self._load_seg(path).to(torch.uint8)

    def _preload_all(self):
        """Load every subject's seg into RAM, in parallel."""
        n = len(self.subject_dirs)
        print(f"Preloading {n} SSL seg volumes into RAM "
              f"({PRELOAD_WORKERS} workers)...")
        self._cache = []
        with ThreadPoolExecutor(max_workers=PRELOAD_WORKERS) as pool:
            for seg in tqdm(pool.map(self._preload_one, self.subject_dirs),
                            total=n, desc="Preloading SSL", ncols=80):
                self._cache.append(seg)
        print(f"Preloading complete. RAM cached {n} seg volumes.")

    def __len__(self):
        return len(self.subject_dirs) * self.num_warps

    def __getitem__(self, idx):
        subject_idx = idx // self.num_warps
        if self._cache is not None:
            seg = self._cache[subject_idx].float()
        else:
            path = os.path.join(self.subject_dirs[subject_idx],
                                self.seg_filename)
            seg = self._load_seg(path)
        return {'seg': seg, 'idx': idx}


# =============================================================================
# Training Functions
# =============================================================================

def ssl_train_epoch(model, stn, dataloader, loss_fn, optimizer, scaler,
                    device, epoch, config):
    """One SSL pretraining epoch.

    For each subject seg S in the dataloader:

    1. Generate a random synthetic warp: (v_synth, flow_fw_synth).
    2. Create the synthetic target: S' = warp_nearest(S, flow_fw_synth).
    3. Feed (S, S') to the model → predicted velocity v_pred.
    4. Loss = vel_mse(v_pred, v_synth)
              + dice(warp_bilinear(S, flow_pred), S')
              + CE(...)
              + bending + jacobian + displacement + lambda terms.
    """
    model.train()

    total_loss = 0
    total_vel_mse = 0
    total_dice = 0
    loss_components_sum = {}
    lambda_stats_sum = {'mean': 0.0, 'std': 0.0, 'p05': 0.0, 'p95': 0.0}

    pbar = tqdm(dataloader,
                desc=f"Epoch {epoch}/{config.num_epochs} [SSL Train]",
                leave=False, ncols=100)

    for batch in pbar:
        seg = batch['seg'].to(device)                   # (B, 5, D, H, W)
        B = seg.shape[0]
        base_seed = 12345
        v_list = []
        fw_list = []
        rv_list = []
        for i in range(B):
            seed = base_seed + int(batch['idx'][i].item())
            v, ffw, frv = generate_synthetic_warp(
                1, config.target_size, stn, device,
                coarse_sizes=config.coarse_sizes,
                magnitude_range=config.magnitude_range,
                seed=seed)
            v_list.append(v)
            fw_list.append(ffw)
            rv_list.append(frv)
        v_synth = torch.cat(v_list, dim=0)
        flow_synth_fw = torch.cat(fw_list, dim=0)

        with torch.no_grad():
            seg_warped = warp_seg_nearest(seg, flow_synth_fw, stn)

        # ---- Network forward + loss ----
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config.use_amp):
            # template_seg = seg (original, "moving")
            # input_seg = seg_warped (synthetic target, "fixed")
            flow_fw, flow_rv, lambda_map, affine_matrix, v_pred = model(
                seg, seg_warped, return_velocity=True)

            warped_seg_fw = _warp_template(
                seg, flow_fw, affine_matrix, stn)
            warped_seg_rv = _warp_sample_inverse(
                seg_warped, flow_rv, affine_matrix, stn)

            loss, loss_dict = loss_fn(
                warped_seg_fw, seg_warped,       # forward direction
                warped_seg_rv, seg,              # reverse direction
                flow_fw, flow_rv, lambda_map,
                affine_matrix=affine_matrix,
                pred_vel=v_pred, target_vel=v_synth,
                return_components=True)

        # ---- Backward + grad clip + step ----
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # ---- Metrics ----
        total_loss += loss.item()
        total_vel_mse += loss_dict['vel_mse'].item()
        dice_score = 1 - (loss_dict['dice'].item() / 2.0)
        total_dice += dice_score

        for k, v in loss_dict.items():
            loss_components_sum[k] = loss_components_sum.get(k, 0) + (
                v.item() if isinstance(v, torch.Tensor) else v)
        _accumulate_lambda_stats(lambda_stats_sum, lambda_map)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'vmse': f'{loss_dict["vel_mse"].item():.5f}',
            'dice': f'{dice_score:.4f}'})

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'vel_mse': total_vel_mse / n,
        'dice': total_dice / n,
        'components': {k: v / n for k, v in loss_components_sum.items()},
        'lambda_stats': {k: v / n for k, v in lambda_stats_sum.items()},
    }


@torch.no_grad()
def ssl_validate_epoch(model, stn, dataloader, loss_fn, device, epoch,
                       config):
    """SSL validation: synthetic-warp procedure on held-out subjects.

    Metrics always score on synthetic warps (not GT labels), so the
    validation is fully self-supervised.  Model selection uses the
    lowest validation velocity MSE — the direct supervision signal.
    """
    model.eval()

    total_loss = 0
    total_vel_mse = 0
    total_dice = 0
    total_folding_pct = 0
    loss_components_sum = {}
    lambda_stats_sum = {'mean': 0.0, 'std': 0.0, 'p05': 0.0, 'p95': 0.0}

    D, H, W = config.target_size
    det_ref = (2.0 / D) * (2.0 / H) * (2.0 / W)

    pbar = tqdm(dataloader,
                desc=f"Epoch {epoch}/{config.num_epochs} [SSL Val]",
                leave=False, ncols=100)

    for batch in pbar:
        seg = batch['seg'].to(device)
        B = seg.shape[0]

        # Synthetic warp for validation
        v_synth, flow_synth_fw, _ = generate_synthetic_warp(
            B, config.target_size, stn, device,
            coarse_sizes=config.coarse_sizes,
            magnitude_range=config.magnitude_range)
        seg_warped = warp_seg_nearest(seg, flow_synth_fw, stn)

        # Forward pass
        flow_fw, flow_rv, lambda_map, affine_matrix, v_pred = model(
            seg, seg_warped, return_velocity=True)

        # Diffeomorphism diagnostics
        det_fw = jacobian_det(flow_fw) / det_ref
        total_folding_pct += (det_fw < 0).float().mean().item() * 100.0

        warped_seg_fw = _warp_template(
            seg, flow_fw, affine_matrix, stn)
        warped_seg_rv = _warp_sample_inverse(
            seg_warped, flow_rv, affine_matrix, stn)

        loss, loss_dict = loss_fn(
            warped_seg_fw, seg_warped,
            warped_seg_rv, seg,
            flow_fw, flow_rv, lambda_map,
            affine_matrix=affine_matrix,
            pred_vel=v_pred, target_vel=v_synth,
            return_components=True)

        total_loss += loss.item()
        total_vel_mse += loss_dict['vel_mse'].item()
        total_dice += 1 - (loss_dict['dice'].item() / 2.0)

        for k, v in loss_dict.items():
            loss_components_sum[k] = loss_components_sum.get(k, 0) + (
                v.item() if isinstance(v, torch.Tensor) else v)
        _accumulate_lambda_stats(lambda_stats_sum, lambda_map)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'vmse': f'{loss_dict["vel_mse"].item():.5f}'})

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'vel_mse': total_vel_mse / n,
        'dice': total_dice / n,
        'folding_pct': total_folding_pct / n,
        'components': {k: v / n for k, v in loss_components_sum.items()},
        'lambda_stats': {k: v / n for k, v in lambda_stats_sum.items()},
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SSL Pretraining via Equivariant Warp Regression')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config_ssl.yaml (default: '
                             'config_ssl.yaml in script directory)')
    args = parser.parse_args()

    config = SSLConfig(args.config)

    # Reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = not config.cudnn_benchmark

    for dir_path in [config.output_dir, config.checkpoint_dir, config.log_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    logger, log_file = setup_logger(config.log_dir, name="ssl_pretrain")

    logger.info("=" * 70)
    logger.info("Self-Supervised Pretraining — Equivariant Warp Regression")
    logger.info("=" * 70)
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Synthetic warps: coarse_sizes={config.coarse_sizes}, "
                f"magnitude_range={config.magnitude_range}")

    write_git_sha(config.output_dir, logger)
    config.save(os.path.join(config.output_dir, "config.json"))

    device = torch.device(config.device)
    if 'cuda' in config.device:
        logger.info(f"GPU: {torch.cuda.get_device_name()}")

    # =========================================================================
    # Data Loading
    # =========================================================================
    logger.info("Loading SSL datasets...")
    logger.info(f"  seg: {config.seg_filename}")

    train_dataset = SSLSegDataset(
        config.train_txt, config.target_size, config.seg_filename,
        num_warps=config.num_warps)
    val_dataset = SSLSegDataset(
        config.val_txt, config.target_size, config.seg_filename,
        num_warps=config.num_warps)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory)

    logger.info(f"Train: {len(train_dataset)} subjects | "
                f"Val: {len(val_dataset)} subjects")
    logger.info(f"Train batches: {len(train_loader)} | "
                f"Val batches: {len(val_loader)}")

    # =========================================================================
    # Model Setup
    # =========================================================================
    logger.info("Initializing model...")

    model = SegRegistrationNet(
        target_size=config.target_size,
        seg_channels=config.seg_channels,
        use_affine=config.use_affine,
        head=config.head,
        lowpass_size=config.lowpass_size,
    ).to(device)
    stn = SpatialTransformer(size=config.target_size, device=device).to(device)

    # Freeze the affine branch during SSL.  The synthetic warps have no
    # affine component, so the "correct" affine is identity (which is the
    # init).  Freezing ensures the velocity MSE target is exact: v_pred
    # should match v_synth without any affine-compensation shift.  The
    # affine net is small (64 FC params) and converges in the first few
    # epochs of supervised fine-tuning.
    if config.use_affine and config.freeze_affine:
        for p in model.affine_net.parameters():
            p.requires_grad_(False)
        logger.info("SSL: AffineNet FROZEN (identity affine throughout "
                    "pretraining — unfreezes during supervised fine-tuning)")

    num_params = sum(p.numel() for p in model.parameters()
                     if p.requires_grad)
    logger.info(f"Trainable parameters: {num_params:,}")
    logger.info(f"Flow head: {config.head}")
    logger.info(f"Affine: {'ENABLED (frozen)' if config.use_affine and config.freeze_affine else 'ENABLED' if config.use_affine else 'DISABLED'}")

    loss_fn = SegRegistrationLoss(
        weights=config.loss_weights,
        class_weights=config.class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr, weight_decay=config.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer, config.warmup_epochs, config.num_epochs, config.min_lr)
    scaler = GradScaler(enabled=config.use_amp)

    logger.info(f"Optimizer: AdamW (lr={config.lr}, "
                f"wd={config.weight_decay})")
    logger.info(f"Scheduler: Cosine with {config.warmup_epochs} warmup epochs")
    active_weights = {k: v for k, v in config.loss_weights.items() if v > 0}
    logger.info(f"Active loss weights: {active_weights}")

    # Resume (for continuing a broken SSL run)
    start_epoch = 1
    best_vel_mse = float('inf')

    if config.resume_from:
        if not os.path.exists(config.resume_from):
            raise FileNotFoundError(
                f"resume.checkpoint_path does not exist: "
                f"{config.resume_from}")
        logger.info(f"Resuming SSL from: {config.resume_from}")
        checkpoint = torch.load(
            config.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_vel_mse = checkpoint.get('best_vel_mse', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch - 1}, "
                    f"best vel_mse: {best_vel_mse:.6f}")

    # =========================================================================
    # Training Loop
    # =========================================================================
    logger.info("=" * 70)
    logger.info("Starting SSL pretraining...")
    logger.info("=" * 70)

    epochs_without_improvement = 0
    training_start = time.time()

    for epoch in range(start_epoch, config.num_epochs + 1):
        epoch_start = time.time()
        current_lr = scheduler.step(epoch - 1)

        train_metrics = ssl_train_epoch(
            model, stn, train_loader, loss_fn, optimizer, scaler,
            device, epoch, config)
        val_metrics = ssl_validate_epoch(
            model, stn, val_loader, loss_fn, device, epoch, config)

        epoch_time = time.time() - epoch_start

        # ---- Logging ----
        logger.info("-" * 70)
        mem = (f" | Peak mem: "
               f"{torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB"
               if 'cuda' in config.device else "")
        logger.info(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"Time: {epoch_time:.1f}s | LR: {current_lr:.2e}{mem}")
        logger.info(
            f"  Train - Loss: {train_metrics['loss']:.5f} | "
            f"Vel MSE: {train_metrics['vel_mse']:.6f} | "
            f"Dice: {train_metrics['dice']:.4f}")
        logger.info(
            f"  Val   - Loss: {val_metrics['loss']:.5f} | "
            f"Vel MSE: {val_metrics['vel_mse']:.6f} | "
            f"Dice: {val_metrics['dice']:.4f}")
        logger.info(
            f"  Val   - Folding: {val_metrics['folding_pct']:.4f}%")

        ls = val_metrics['lambda_stats']
        logger.info(
            f"  Val   - Lambda: mean={ls['mean']:.3f} "
            f"std={ls['std']:.3f} | "
            f"p05={ls['p05']:.3f} p95={ls['p95']:.3f} "
            f"(collapse signal: std→0 and p95-p05→0)")

        log_loss_components_artifact(
            epoch,
            train_metrics['components'], val_metrics['components'],
            loss_fn.weights, config.output_dir, logger,
            train_lambda_stats=train_metrics['lambda_stats'],
            val_lambda_stats=val_metrics['lambda_stats'])

        # ---- Model selection: lowest validation velocity MSE ----
        is_best = val_metrics['vel_mse'] < best_vel_mse
        if is_best:
            best_vel_mse = val_metrics['vel_mse']
            epochs_without_improvement = 0
            logger.info(
                f"  ★ New best! Vel MSE: {best_vel_mse:.6f}")
        else:
            epochs_without_improvement += 1

        # ---- Checkpointing ----
        if is_best or epoch % config.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_vel_mse': val_metrics['vel_mse'],
                'val_dice': val_metrics['dice'],
                'best_vel_mse': best_vel_mse,
            }
            if is_best:
                safe_save(
                    checkpoint,
                    os.path.join(config.checkpoint_dir, "best_model.pth"),
                    logger)
            if epoch % config.save_every == 0:
                safe_save(
                    checkpoint,
                    os.path.join(config.checkpoint_dir,
                                 f"checkpoint_epoch_{epoch:03d}.pth"),
                    logger)

        # ---- Early stopping ----
        if epochs_without_improvement >= config.patience:
            logger.warning(
                f"Early stopping: no improvement for "
                f"{config.patience} epochs.")
            break

        if 'cuda' in config.device:
            torch.cuda.empty_cache()

    total_time = time.time() - training_start

    logger.info("=" * 70)
    logger.info("SSL Pretraining Complete!")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time / 3600:.2f} hours")
    logger.info(f"Best validation vel MSE: {best_vel_mse:.6f}")
    logger.info(f"Checkpoints: {config.checkpoint_dir}")
    logger.info(f"Logs: {log_file}")
    logger.info("")
    logger.info("To fine-tune on labelled data, set in config.yaml:")
    logger.info("  resume:")
    logger.info(f"    checkpoint_path: \""
                f"{config.checkpoint_dir}/best_model.pth\"")
    logger.info("    weights_only: true")


if __name__ == "__main__":
    main()
