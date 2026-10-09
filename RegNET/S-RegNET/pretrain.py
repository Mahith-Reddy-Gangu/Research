"""
Self-Supervised Pretraining Script for Seg-only Registration

This script generates synthetic ground-truth warps on the fly using the SynthSeg
input segmentations. It trains the model to recover these generated warps.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import numpy as np
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse

from model import SegRegistrationNet, SpatialTransformer
from losses import (dice_loss, bending_energy_loss, jacobian_det_loss,
                    velocity_mse_loss, compute_dice_score)
from get_data import SegDataset
from train import Config, setup_logger, WarmupCosineScheduler, safe_save

# =============================================================================
# Synthetic Warp Generation
# =============================================================================

def generate_synthetic_batch(input_seg, stn, lowpass_size, sigma=0.1):
    """
    Generates a synthetic target by applying a random band-limited SVF.
    Matches the BandLimitedHead capacity exactly.
    """
    B, C, D, H, W = input_seg.shape
    device = input_seg.device
    
    # Sample coarse velocity field
    d = lowpass_size
    v_coarse = torch.randn(B, 3, d, d, d, device=device) * sigma
    
    # Upsample to full resolution
    v_gt = F.interpolate(v_coarse, size=(D, H, W), mode='trilinear', align_corners=False)
    
    # Integrate via scaling and squaring
    flow_gt = v_gt / (2 ** 7)
    for _ in range(7):
        flow_gt = flow_gt + stn(flow_gt, flow_gt)
        
    # Warp input segmentation to create synthetic target
    synthetic_target = stn(input_seg, flow_gt)
    
    return synthetic_target, v_gt


# =============================================================================
# Pretraining Loop
# =============================================================================

def pretrain_epoch(model, stn, dataloader, optimizer, scaler, device, epoch, config):
    model.train()
    total_loss = 0
    loss_components_sum = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs} [Pretrain]", leave=False, ncols=100)
    
    for batch in pbar:
        input_seg = batch['input_seg'].to(device)
        
        # Generate synthetic target on the fly
        with torch.no_grad():
            lowpass_size = config.lowpass_size or config.target_size[0] * 3 // 4
            synthetic_target, v_gt = generate_synthetic_batch(input_seg, stn, lowpass_size, sigma=0.05)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=config.use_amp):
            # Model predicts warp from input_seg -> synthetic_target
            flow_fw, flow_rv, lambda_map, affine_matrix, pred_vel = model(
                input_seg, synthetic_target, return_velocity=True
            )
            
            warped_seg_fw = stn(input_seg, flow_fw)
            
            # Compute Losses
            vel_mse = velocity_mse_loss(pred_vel, v_gt)
            dice = dice_loss(warped_seg_fw, synthetic_target, class_weights=config.class_weights)
            bending = bending_energy_loss(flow_fw)
            jacobian = jacobian_det_loss(flow_fw)
            
            # We weight Velocity MSE heavily as it is the primary self-supervised signal
            loss_dict = {
                'vel_mse': vel_mse * 100.0,
                'dice': dice * 1.0,
                'bending': bending * 0.05,
                'jacobian': jacobian * 0.15
            }
            
            loss = sum(loss_dict.values())
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        for k, v in loss_dict.items():
            loss_components_sum[k] = loss_components_sum.get(k, 0) + v.item()
            
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'vel_mse': f'{vel_mse.item():.4f}'})
        
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'components': {k: v / num_batches for k, v in loss_components_sum.items()},
    }


def main():
    parser = argparse.ArgumentParser(description='Self-Supervised Pretraining')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    args = parser.parse_args()

    config = Config(args.config)
    
    # Overwrite output dir for pretraining
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config.output_dir = f"{config.output_dir}_pretrain/{timestamp}"
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    config.log_dir = os.path.join(config.output_dir, "logs")

    # Reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)

    for dir_path in [config.output_dir, config.checkpoint_dir, config.log_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    logger, log_file = setup_logger(config.log_dir, name="ssl_pretrain")
    logger.info("=" * 70)
    logger.info("Self-Supervised Pretraining (On-the-fly Synthetic Warps)")
    logger.info("=" * 70)
    
    config.save(os.path.join(config.output_dir, "config.json"))

    device = torch.device(config.device)
    
    # Model
    model = SegRegistrationNet(
        target_size=config.target_size, 
        seg_channels=config.seg_channels, 
        use_affine=False, # Disable affine for pretraining dense field
        head='bandlimited', # Enforce tied bandlimited head
        lowpass_size=config.lowpass_size
    ).to(device)
    
    stn = SpatialTransformer(size=config.target_size, device=device).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler = GradScaler(enabled=config.use_amp)
    scheduler = WarmupCosineScheduler(optimizer, config.warmup_epochs, config.num_epochs, config.min_lr)

    # Dataset
    train_dataset = SegDataset(
        config.train_txt, config.template_seg_path,
        target_size=config.target_size,
        seg_filename=config.seg_filename,
        input_seg_filename=config.input_seg_filename,
        preload=config.pin_memory,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    
    # Loop
    best_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        lr = scheduler.step(epoch - 1)
        logger.info(f"\nEpoch {epoch}/{config.num_epochs} [LR: {lr:.2e}]")
        
        metrics = pretrain_epoch(model, stn, train_loader, optimizer, scaler, device, epoch, config)
        
        logger.info(f"  Train - Loss: {metrics['loss']:.4f}")
        for k, v in metrics['components'].items():
            logger.info(f"          {k}: {v:.4f}")
            
        # Save Best
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            save_path = os.path.join(config.checkpoint_dir, "best_pretrain_model.pth")
            safe_save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path, logger=logger)
            logger.info(f"  Saved best model to {save_path}")

if __name__ == "__main__":
    main()
