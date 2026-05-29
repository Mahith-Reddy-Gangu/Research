"""
Compute Self-Intersection (Folding) Statistics for Seg-only Registration Model

Self-intersection occurs when the Jacobian determinant of the deformation
field becomes negative — different source voxels map to the same target
location, so the transformation is no longer diffeomorphic.

What is measured:
    voxel-level folding rate: fraction of voxels with det(J) < 0
    plus per-volume min/mean/std of det(J), and the existing training-time
    jacobian_det_loss (linear-neg + top-K + pre-fold) for cross-reference.

What is NOT measured:
    mesh-level surface-flip (the actual diffeomorphism deliverable). A
    single voxel at det = -8 can flip many triangulated mesh elements;
    the two metrics can disagree by orders of magnitude. For that, run a
    separate tool that extracts iso-surfaces of the warped seg and counts
    inverted triangle orientations.

Usage:
    python compute_self_intersection.py \
        --which_timestamp 20260528_115247 --num_samples 10
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import yaml

from model import SegRegistrationNet, SpatialTransformer
from losses import compute_dice_score, jacobian_det_loss
from get_data import SegDataset


def _load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def compute_self_intersection_stats(flow):
    """
    Per-voxel Jacobian-determinant statistics for a deformation field.

    Args:
        flow: (B, 3, D, H, W) displacement field in normalized [-1,1] coords.
              Channel order (x, y, z) matching model.SpatialTransformer's
              id_grid = stack((xx, yy, zz)).

    Returns:
        dict with self-intersection loss, folding %, and Jacobian summary.
    """
    if isinstance(flow, np.ndarray):
        flow = torch.from_numpy(flow)
    if flow.ndim == 4:
        flow = flow.unsqueeze(0)

    B, _, D, H, W = flow.shape

    lin_z = (2 * torch.arange(D, device=flow.device, dtype=flow.dtype) + 1) / D - 1
    lin_y = (2 * torch.arange(H, device=flow.device, dtype=flow.dtype) + 1) / H - 1
    lin_x = (2 * torch.arange(W, device=flow.device, dtype=flow.dtype) + 1) / W - 1
    zz, yy, xx = torch.meshgrid(lin_z, lin_y, lin_x, indexing='ij')
    id_grid = torch.stack((xx, yy, zz), dim=0).unsqueeze(0)

    warped = id_grid + flow

    dW_d = warped[:, :, 1:, :, :] - warped[:, :, :-1, :, :]
    dW_h = warped[:, :, :, 1:, :] - warped[:, :, :, :-1, :]
    dW_w = warped[:, :, :, :, 1:] - warped[:, :, :, :, :-1]

    dW_d = F.pad(dW_d, (0, 0, 0, 0, 0, 1))
    dW_h = F.pad(dW_h, (0, 0, 0, 1, 0, 0))
    dW_w = F.pad(dW_w, (0, 1, 0, 0, 0, 0))

    a, b, c = dW_w[:, 0], dW_h[:, 0], dW_d[:, 0]
    d, e, f = dW_w[:, 1], dW_h[:, 1], dW_d[:, 1]
    g, h, i = dW_w[:, 2], dW_h[:, 2], dW_d[:, 2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    # Identity warp has det = (2/D)(2/H)(2/W); normalize so no-op → 1.
    ref = (2.0 / D) * (2.0 / H) * (2.0 / W)
    det_norm = det / ref

    self_intersection_loss = F.relu(-det_norm).mean()

    det_np = det_norm.detach().cpu().numpy()
    folding_mask = det_np < 0
    num_folding_voxels = int(folding_mask.sum())
    total_voxels = int(det_np.size)
    folding_percentage = (num_folding_voxels / total_voxels) * 100

    return {
        'loss': self_intersection_loss.item(),
        'folding_percentage': folding_percentage,
        'num_folding_voxels': num_folding_voxels,
        'total_voxels': total_voxels,
        'min_jacobian': float(det_np.min()),
        'max_jacobian': float(det_np.max()),
        'mean_jacobian': float(det_np.mean()),
        'std_jacobian': float(det_np.std()),
    }


def _warp_template(template_seg, final_flow, affine_matrix, stn):
    """Replay train.py's invariant: affine first, then dense flow."""
    if affine_matrix is not None:
        affine_grid = F.affine_grid(
            affine_matrix, template_seg.size(), align_corners=False
        )
        aligned_seg = F.grid_sample(
            template_seg, affine_grid, mode='bilinear',
            padding_mode='zeros', align_corners=False,
        )
        return stn(aligned_seg, final_flow)
    return stn(template_seg, final_flow)


def main():
    parser = argparse.ArgumentParser(description='Self-intersection / folding analysis (seg-only)')
    parser.add_argument('--which_timestamp', type=str, default=None,
                        help='Timestamp dir under output.base_dir (e.g. 20260528_115247)')
    parser.add_argument('--data_split', type=str, default='val',
                        choices=['train', 'val'], help='Data split to evaluate')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to analyze (-1 for all)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference')
    parser.add_argument('--target_size', type=int, nargs=3, default=None,
                        help='Override target volume size (defaults to config)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (defaults to ./config.yaml)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Direct checkpoint .pth path (overrides --which_timestamp)')
    args = parser.parse_args()

    # Config (single source of truth for paths and arch flags)
    config_path = args.config or (Path(__file__).parent / "config.yaml")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    cfg = _load_config(config_path)

    # Checkpoint resolution
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        if args.which_timestamp is None:
            raise ValueError("Provide --checkpoint or --which_timestamp")
        base = Path(cfg['output']['base_dir']) / args.which_timestamp
        checkpoint_path = str(base / cfg['output']['checkpoint_subdir'] / 'best_model.pth')

    target_size = tuple(args.target_size) if args.target_size else tuple(cfg['model']['target_size'])
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    data_list = cfg['data']['val_txt'] if args.data_split == 'val' else cfg['data']['train_txt']
    template_seg_path = cfg['data']['template_seg_path']
    seg_filename = cfg['data'].get('seg_filename', 'seg4_onehot.npy')

    use_affine = cfg.get('affine', {}).get('enabled', False)
    seg_channels = cfg['model'].get('seg_channels', 5)
    num_classes = cfg['model'].get('num_classes', 5)

    print("=" * 70)
    print("SELF-INTERSECTION ANALYSIS — seg-only registration")
    print("=" * 70)
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Device:      {device}")
    print(f"Data split:  {args.data_split}")
    print(f"use_affine:  {use_affine}")
    print(f"target_size: {target_size}")

    # Model + STN
    print("\nLoading model...")
    model = SegRegistrationNet(seg_channels=seg_channels, use_affine=use_affine).to(device)
    stn = SpatialTransformer(size=target_size, device=device).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_has_affine = any(k.startswith('affine_net.') for k in checkpoint['model_state_dict'])
    assert ckpt_has_affine == use_affine, (
        f"checkpoint/config affine mismatch: ckpt has affine_net={ckpt_has_affine}, "
        f"config affine.enabled={use_affine}"
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    print(f"Loaded epoch={checkpoint.get('epoch', 'unknown')}, "
          f"best_dice={checkpoint.get('best_dice', float('nan')):.4f}")

    # Dataset
    print("\nLoading dataset...")
    dataset = SegDataset(
        data_list, template_seg_path,
        target_size=target_size, seg_filename=seg_filename, preload=False,
    )
    num_samples = len(dataset) if args.num_samples == -1 else min(args.num_samples, len(dataset))
    print(f"Analyzing {num_samples} samples from {len(dataset)} total")

    all_results = []

    print("\n" + "=" * 70)
    print("RUNNING INFERENCE...")
    print("=" * 70)

    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Processing"):
            data = dataset[idx]
            template_seg = data['template_seg'].unsqueeze(0).to(device)
            sample_seg = data['sample_seg'].unsqueeze(0).to(device)

            final_flow, lambda_map, affine_matrix = model(template_seg, sample_seg)
            warped_seg = _warp_template(template_seg, final_flow, affine_matrix, stn)

            dice_per_class, mean_dice = compute_dice_score(
                warped_seg, sample_seg, num_classes=num_classes
            )

            flow_cpu = final_flow.cpu()
            stats = compute_self_intersection_stats(flow_cpu)
            jac_train_loss = jacobian_det_loss(flow_cpu).item()

            all_results.append({
                'sample_idx': idx,
                'mean_dice': mean_dice,
                'dice_per_class': dice_per_class,
                **stats,
                'jacobian_det_loss_training': jac_train_loss,
            })

            del template_seg, sample_seg, final_flow, lambda_map, affine_matrix
            del warped_seg, flow_cpu
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # Aggregate
    losses = [r['loss'] for r in all_results]
    folding_pcts = [r['folding_percentage'] for r in all_results]
    min_jacs = [r['min_jacobian'] for r in all_results]
    mean_jacs = [r['mean_jacobian'] for r in all_results]
    dice_scores = [r['mean_dice'] for r in all_results]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Self-intersection loss   mean={np.mean(losses):.6f}  std={np.std(losses):.6f}  max={np.max(losses):.6f}
Folding %                mean={np.mean(folding_pcts):.4f}%  std={np.std(folding_pcts):.4f}%  max={np.max(folding_pcts):.4f}%
Jacobian (mean of means) {np.mean(mean_jacs):.4f}
Jacobian (worst min)     {np.min(min_jacs):.4f}     ← single worst voxel across all samples
Dice                     mean={np.mean(dice_scores):.4f}  std={np.std(dice_scores):.4f}
""")

    mean_folding = float(np.mean(folding_pcts))
    if mean_folding < 0.1:
        quality, interp = "EXCELLENT", "Almost no folding — deformations are diffeomorphic at voxel resolution."
    elif mean_folding < 1.0:
        quality, interp = "GOOD", "Voxel-folding under the 1% threshold — verify mesh-level surface-flip separately."
    elif mean_folding < 5.0:
        quality, interp = "MODERATE", "Some folding present — may affect downstream surface analysis."
    else:
        quality, interp = "POOR", "Significant folding — consider increasing Jacobian / smoothness regularization."
    print(f"Quality assessment: {quality} — {interp}\n")

    # Per-sample table
    print("PER-SAMPLE RESULTS")
    print("-" * 92)
    print(f"{'Sample':<8}{'Dice':>8}{'Self-Int':>12}{'Folding %':>12}{'Min Jac':>10}{'Mean Jac':>10}{'Jac Loss':>12}")
    print("-" * 92)
    for r in all_results:
        print(f"{r['sample_idx']:<8}{r['mean_dice']:>8.4f}{r['loss']:>12.6f}"
              f"{r['folding_percentage']:>11.4f}%{r['min_jacobian']:>10.4f}"
              f"{r['mean_jacobian']:>10.4f}{r['jacobian_det_loss_training']:>12.6f}")
    print("-" * 92)
    print(f"{'MEAN':<8}{np.mean(dice_scores):>8.4f}{np.mean(losses):>12.6f}"
          f"{np.mean(folding_pcts):>11.4f}%{np.mean(min_jacs):>10.4f}"
          f"{np.mean(mean_jacs):>10.4f}")

    # Save JSON next to the checkpoint
    output_file = Path(checkpoint_path).parent.parent / "self_intersection_analysis.json"
    summary = {
        'config': {
            'timestamp': args.which_timestamp,
            'data_split': args.data_split,
            'num_samples': num_samples,
            'device': str(device),
            'use_affine': use_affine,
            'target_size': list(target_size),
        },
        'aggregate': {
            'self_intersection_loss': {
                'mean': float(np.mean(losses)),
                'std':  float(np.std(losses)),
                'min':  float(np.min(losses)),
                'max':  float(np.max(losses)),
            },
            'folding_percentage': {
                'mean': float(np.mean(folding_pcts)),
                'std':  float(np.std(folding_pcts)),
                'min':  float(np.min(folding_pcts)),
                'max':  float(np.max(folding_pcts)),
            },
            'jacobian': {
                'mean_of_means': float(np.mean(mean_jacs)),
                'worst_min':     float(np.min(min_jacs)),
            },
            'dice': {
                'mean': float(np.mean(dice_scores)),
                'std':  float(np.std(dice_scores)),
            },
            'quality_assessment': quality,
        },
        'per_sample': all_results,
    }
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
