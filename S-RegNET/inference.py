"""
Seg-only Registration Inference Script

Registers the template segmentation to a sample segmentation using a
trained checkpoint, computes all losses, and (optionally) writes
visualizations + a warped-seg NIfTI.

Usage:
    # Standard:
    python inference.py \\
        --input_seg /path/to/sample_seg.nii.gz \\
        --checkpoint /path/to/best_model.pth \\
        --output_dir /path/to/results

    # Custom config:
    python inference.py \\
        --input_seg /path/to/sample_seg.nii.gz \\
        --checkpoint /path/to/best_model.pth \\
        --config /path/to/config.yaml \\
        --output_dir /path/to/results

    # Losses only (skip visualizations + NIfTI):
    python inference.py \\
        --input_seg /path/to/sample_seg.nii.gz \\
        --checkpoint /path/to/best_model.pth \\
        --output_dir /path/to/results \\
        --losses_only
"""

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import argparse
import json
import os
import yaml
from pathlib import Path
from scipy.ndimage import distance_transform_edt, binary_erosion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from git_provenance import write_git_sha
from model import SegRegistrationNet, SpatialTransformer
from losses import (
    dice_loss, cross_entropy_loss,
    bending_energy_loss, jacobian_det_loss, displacement_loss,
    lambda_weighted_smoothness, lambda_prior_loss,
    affine_regularization_loss, affine_orthogonality_loss,
    compute_dice_score, jacobian_det,
)


# =============================================================================
# Data Loading
# =============================================================================

def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# FreeSurfer label → 5-class remapping (Background, Cortex, Subcortical GM,
# White Matter, CSF). Used when --input_seg is a .nii.gz integer-label
# volume; if the user passes a pre-converted .npy one-hot, this is skipped.
LABEL_MAPPING = {
    0: 0, 24: 0,
    3: 1, 42: 1,
    10: 2, 49: 2, 11: 3, 50: 2, 12: 2, 51: 2, 13: 2, 52: 2,
    17: 2, 53: 2, 18: 2, 54: 2, 26: 2, 58: 2, 60: 2, 8: 2, 47: 2,
    2: 3, 41: 3, 7: 3, 46: 3, 16: 3, 28: 3,
    4: 4, 43: 4, 5: 4, 44: 4, 14: 4, 15: 4,
}


def load_seg_input(path, target_size, num_classes=5):
    """
    Load a sample seg from disk. Supports:
      - .npy one-hot (C, D, H, W) — used as-is after resize.
      - .nii.gz integer label volume — FreeSurfer labels remapped to 0..C-1,
        unlisted labels become background (class 0), then one-hot encoded.

    Returns:
        seg: (C, D, H, W) float tensor.
        nifti_affine: 4x4 array (eye when input is .npy — used only for
                      saving the warped NIfTI to disk).
    """
    if path.endswith('.npy'):
        seg = np.load(path)
        seg = torch.tensor(seg, dtype=torch.float32)
        nifti_affine = np.eye(4)
    else:
        img = nib.load(path)
        data = img.get_fdata().astype(np.int64)
        remapped = np.zeros_like(data)
        for src, dst in LABEL_MAPPING.items():
            remapped[data == src] = dst
        seg_tensor = torch.tensor(remapped)
        seg = F.one_hot(seg_tensor.long(), num_classes).permute(3, 0, 1, 2).float()
        nifti_affine = img.affine

    seg = F.interpolate(seg.unsqueeze(0), size=target_size, mode='nearest').squeeze(0)
    return seg, nifti_affine


def load_template_seg(cfg, target_size):
    """Load the template seg from the path in config."""
    tseg = np.load(cfg['data']['template_seg_path'])
    tseg = torch.tensor(tseg, dtype=torch.float32)
    tseg = F.interpolate(tseg.unsqueeze(0), size=target_size, mode='nearest').squeeze(0)
    return tseg


# =============================================================================
# Loss Computation
# =============================================================================

def compute_avg_hausdorff_distance(seg_pred, seg_gt, num_classes, voxel_spacing=None):
    """
    Average Hausdorff Distance (AHD) per foreground class. For each class,
    extract surface voxels by mask-minus-erosion, then average the two
    directed distance-transform means.
    """
    if voxel_spacing is None:
        voxel_spacing = (1.0, 1.0, 1.0)

    pred_labels = seg_pred.squeeze(0).argmax(dim=0).cpu().numpy()
    gt_labels = seg_gt.squeeze(0).argmax(dim=0).cpu().numpy()

    ahd_per_class = []
    for c in range(1, num_classes):
        pred_mask = (pred_labels == c).astype(np.uint8)
        gt_mask = (gt_labels == c).astype(np.uint8)

        if pred_mask.sum() == 0 or gt_mask.sum() == 0:
            ahd_per_class.append(float('nan'))
            continue

        pred_b = pred_mask ^ binary_erosion(pred_mask).astype(np.uint8)
        gt_b = gt_mask ^ binary_erosion(gt_mask).astype(np.uint8)

        dt_gt = distance_transform_edt(~gt_b.astype(bool), sampling=voxel_spacing)
        dt_pred = distance_transform_edt(~pred_b.astype(bool), sampling=voxel_spacing)

        d_pred_to_gt = float(dt_gt[pred_b.astype(bool)].mean())
        d_gt_to_pred = float(dt_pred[gt_b.astype(bool)].mean())
        ahd_per_class.append((d_pred_to_gt + d_gt_to_pred) / 2.0)

    valid = [v for v in ahd_per_class if not np.isnan(v)]
    mean_ahd = float(np.mean(valid)) if valid else float('nan')
    return ahd_per_class, mean_ahd


def compute_all_losses(warped_seg, sample_seg, final_flow,
                       lambda_map, affine_matrix, num_classes=5):
    """Compute every available loss/metric for a single registration."""
    results = {}

    # Segmentation alignment
    results['dice'] = dice_loss(warped_seg, sample_seg).item()
    results['cross_entropy'] = cross_entropy_loss(warped_seg, sample_seg).item()

    # Geometric regularization
    results['bending'] = bending_energy_loss(final_flow).item()
    results['jacobian'] = jacobian_det_loss(final_flow).item()
    results['displacement'] = displacement_loss(final_flow).item()

    # Diffeomorphism diagnostics (matches train.py val loop)
    _, _, D_, H_, W_ = final_flow.shape
    det_ref = (2.0 / D_) * (2.0 / H_) * (2.0 / W_)
    det_norm = jacobian_det(final_flow) / det_ref
    results['folding_pct'] = (det_norm < 0).float().mean().item() * 100.0
    results['min_det'] = det_norm.min().item()
    results['fold_voxels'] = int((det_norm < 0).sum().item())

    # Lambda-based regularization
    if lambda_map is not None:
        results['lambda_smoothness'] = lambda_weighted_smoothness(final_flow, lambda_map).item()
        results['lambda_prior'] = lambda_prior_loss(lambda_map, sample_seg).item()

    # Affine
    if affine_matrix is not None:
        results['affine_reg'] = affine_regularization_loss(affine_matrix).item()
        results['affine_ortho'] = affine_orthogonality_loss(affine_matrix).item()

    # Per-class dice + AHD
    dice_per_class, mean_dice = compute_dice_score(warped_seg, sample_seg, num_classes)
    results['dice_score'] = mean_dice
    results['dice_per_class'] = dice_per_class

    ahd_per_class, mean_ahd = compute_avg_hausdorff_distance(
        warped_seg, sample_seg, num_classes
    )
    results['avg_hausdorff'] = mean_ahd
    results['avg_hausdorff_per_class'] = ahd_per_class

    # Flow statistics
    flow_np = final_flow.detach().cpu().numpy()
    results['flow_magnitude_mean'] = float(np.sqrt((flow_np ** 2).sum(axis=1)).mean())
    results['flow_magnitude_max'] = float(np.sqrt((flow_np ** 2).sum(axis=1)).max())

    return results


# =============================================================================
# Visualization
# =============================================================================

CLASS_COLORS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.3, 0.3],
    [0.4, 0.8, 0.4],
    [1.0, 0.9, 0.3],
    [0.2, 0.6, 1.0],
])


def _seg_panel(ax, labels_slice, title):
    """Draw a colored label-map slice with a title."""
    rgb = CLASS_COLORS[labels_slice]
    ax.imshow(rgb, aspect='equal')
    ax.set_title(title)
    ax.axis('off')


def plot_comparison(template_seg, sample_seg, warped_seg, save_path):
    """
    3×3 grid: rows = axial/coronal/sagittal mid-slice; columns = template,
    target (sample), warped result. Each panel is a colored label map.
    """
    t_labels = template_seg.squeeze().argmax(dim=0).cpu().numpy()
    s_labels = sample_seg.squeeze().argmax(dim=0).cpu().numpy()
    w_labels = warped_seg.squeeze().argmax(dim=0).cpu().numpy()
    D, H, W = t_labels.shape

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    views = [
        ('Axial',   lambda v: v[D // 2, :, :]),
        ('Coronal', lambda v: v[:, H // 2, :]),
        ('Sagittal', lambda v: v[:, :, W // 2]),
    ]

    for row_idx, (view_name, getter) in enumerate(views):
        _seg_panel(axes[row_idx, 0], getter(t_labels), f'{view_name} — Template')
        _seg_panel(axes[row_idx, 1], getter(s_labels), f'{view_name} — Target (Sample)')
        _seg_panel(axes[row_idx, 2], getter(w_labels), f'{view_name} — Warped Template')

    fig.suptitle('Seg Registration Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_deformation_field(flow, save_path):
    """Magnitude (3 views) + per-axis component maps at axial mid-slice."""
    D, H, W = flow.shape[1:]
    mid = D // 2
    magnitude = np.sqrt((flow ** 2).sum(axis=0))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[0, 0].imshow(magnitude[mid, :, :], cmap='hot');         axes[0, 0].set_title('Magnitude (Axial)')
    axes[0, 1].imshow(magnitude[:, H // 2, :], cmap='hot');      axes[0, 1].set_title('Magnitude (Coronal)')
    axes[0, 2].imshow(magnitude[:, :, W // 2], cmap='hot');      axes[0, 2].set_title('Magnitude (Sagittal)')

    component_names = ['X (L-R)', 'Y (A-P)', 'Z (S-I)']
    for i in range(3):
        im = axes[1, i].imshow(flow[i, mid, :, :], cmap='RdBu_r',
                               vmin=-flow.max(), vmax=flow.max())
        axes[1, i].set_title(f'Flow {component_names[i]}')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)

    for ax in axes.flat:
        ax.axis('off')

    fig.suptitle('Deformation Field Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_warped_nifti(warped_seg, nifti_affine, save_path):
    """Save the warped seg as a NIfTI label volume (argmax → int16)."""
    labels = warped_seg.squeeze().argmax(dim=0).cpu().numpy().astype(np.int16)
    img = nib.Nifti1Image(labels, nifti_affine)
    nib.save(img, save_path)


# =============================================================================
# Inference
# =============================================================================

def _print_losses(losses, class_names, has_affine):
    print()
    print("=" * 70)
    print("LOSS VALUES")
    print("=" * 70)

    print("\n  Segmentation Losses:")
    print(f"    Dice loss         : {losses['dice']:.6f}")
    print(f"    Cross entropy     : {losses['cross_entropy']:.6f}")

    print("\n  Geometric Regularization:")
    print(f"    Bending energy    : {losses['bending']:.6f}")
    print(f"    Jacobian det      : {losses['jacobian']:.6f}")
    print(f"    Displacement      : {losses['displacement']:.6f}")
    print(f"    Folding %         : {losses['folding_pct']:.4f}%  "
          f"(min det = {losses['min_det']:.4f}, fold voxels = {losses['fold_voxels']})")

    if 'lambda_smoothness' in losses:
        print(f"    Lambda smoothness : {losses['lambda_smoothness']:.6f}")
        print(f"    Lambda prior      : {losses['lambda_prior']:.6f}")

    if has_affine:
        print("\n  Affine Regularization:")
        print(f"    Affine identity   : {losses['affine_reg']:.6f}")
        print(f"    Affine ortho      : {losses['affine_ortho']:.6f}")

    print(f"\n  Dice Scores:")
    print(f"    Mean Dice         : {losses['dice_score']:.4f}")
    for i, name in enumerate(class_names):
        print(f"    {name:18s}: {losses['dice_per_class'][i]:.4f}")

    print(f"\n  Avg Hausdorff Distance (AHD, voxels):")
    print(f"    Mean AHD          : {losses['avg_hausdorff']:.4f}")
    for i, name in enumerate(class_names[1:], start=1):
        v = losses['avg_hausdorff_per_class'][i - 1]
        v_str = f"{v:.4f}" if not np.isnan(v) else "  N/A"
        print(f"    {name:18s}: {v_str}")

    print(f"\n  Flow Statistics:")
    print(f"    Mean magnitude    : {losses['flow_magnitude_mean']:.6f}")
    print(f"    Max magnitude     : {losses['flow_magnitude_max']:.6f}")


def setup_inference(checkpoint_path, config_path=None, device='cuda:0',
                    use_affine=False, verbose=True):
    """Load config, template seg, and model once; reuse across samples."""
    cfg = load_config(config_path)
    target_size = tuple(cfg['model']['target_size'])
    num_classes = cfg['model']['num_classes']
    class_names = cfg['visualization']['class_names']
    dev = torch.device(device)

    if verbose:
        print("Loading template seg...")
    template_seg = load_template_seg(cfg, target_size).unsqueeze(0).to(dev)

    if verbose:
        print("Loading model...")
    model = SegRegistrationNet(seg_channels=num_classes, use_affine=use_affine).to(dev)
    stn = SpatialTransformer(size=target_size, device=dev).to(dev)

    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    if verbose:
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
              f"best_dice {ckpt.get('best_dice', '?')})")

    return {
        'cfg': cfg,
        'target_size': target_size,
        'num_classes': num_classes,
        'class_names': class_names,
        'device': dev,
        'use_affine': use_affine,
        'template_seg': template_seg,
        'model': model,
        'stn': stn,
    }


@torch.no_grad()
def run_inference_on_sample(ctx, input_seg_path, output_dir,
                            losses_only=False, verbose=False):
    """Run inference on a single sample. Returns the losses dict."""
    target_size = ctx['target_size']
    num_classes = ctx['num_classes']
    class_names = ctx['class_names']
    device = ctx['device']
    template_seg = ctx['template_seg']
    model = ctx['model']
    stn = ctx['stn']

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_git_sha(output_dir)

    if verbose:
        print(f"Loading sample seg: {input_seg_path}")
    sample_seg, nifti_affine = load_seg_input(input_seg_path, target_size, num_classes)
    sample_seg = sample_seg.unsqueeze(0).to(device)

    # Forward + affine-then-flow warp replay (critical invariant; matches train.py:_warp_template)
    final_flow, lambda_map, affine_matrix = model(template_seg, sample_seg)
    if affine_matrix is not None:
        affine_grid = F.affine_grid(affine_matrix, template_seg.size(), align_corners=False)
        aligned_seg = F.grid_sample(template_seg, affine_grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=False)
        warped_seg = stn(aligned_seg, final_flow)
    else:
        warped_seg = stn(template_seg, final_flow)

    losses = compute_all_losses(
        warped_seg, sample_seg, final_flow,
        lambda_map, affine_matrix, num_classes=num_classes,
    )

    if verbose:
        _print_losses(losses, class_names, has_affine=affine_matrix is not None)

    with open(output_dir / "losses.json", 'w') as f:
        json.dump(losses, f, indent=2)
    if verbose:
        print(f"\n  Losses saved to: {output_dir / 'losses.json'}")

    if losses_only:
        return losses

    if verbose:
        print("\nGenerating visualizations...")

    plot_comparison(template_seg, sample_seg, warped_seg,
                    output_dir / 'comparison.png')
    plot_deformation_field(final_flow.squeeze(0).cpu().numpy(),
                           output_dir / 'deformation_field.png')
    save_warped_nifti(warped_seg, nifti_affine,
                      str(output_dir / 'warped_seg.nii.gz'))

    if verbose:
        print(f"\nAll outputs saved to: {output_dir}")
    return losses


def run_inference(args):
    print("=" * 70)
    print("Seg-only Registration Inference")
    print("=" * 70)
    print(f"Input Seg       : {args.input_seg}")
    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Config          : {args.config}")
    print(f"Device          : {args.device}")
    print(f"Output dir      : {args.output_dir}")
    print(f"Use affine      : {args.use_affine}")
    print(f"Losses only     : {args.losses_only}")
    print()

    ctx = setup_inference(args.checkpoint, args.config, args.device,
                          use_affine=args.use_affine, verbose=True)
    run_inference_on_sample(
        ctx, args.input_seg, args.output_dir,
        losses_only=args.losses_only, verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(description='Seg-only Registration Inference')
    parser.add_argument('--input_seg', type=str, required=True,
                        help='Path to sample segmentation (.nii.gz integer labels or .npy one-hot)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (default: config.yaml in script directory)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (e.g. cuda:0, cpu)')
    parser.add_argument('--use_affine', action='store_true',
                        help='Use the affine pre-alignment stage (must match the checkpoint)')
    parser.add_argument('--losses_only', action='store_true',
                        help='Skip visualizations and NIfTI; only save losses.json')
    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()
