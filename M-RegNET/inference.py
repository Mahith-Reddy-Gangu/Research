"""
MRI Registration Inference Script

Registers a sample MRI to the template using a trained model checkpoint.
Computes all available loss values and generates visualizations.

Usage:
    # Basic (MRI only - computes regularization + NCC losses):
    python inference.py \
        --input /path/to/sample.nii.gz \
        --checkpoint /path/to/best_model.pth \
        --output_dir /path/to/results

    # With ground truth segmentation (additionally computes dice/boundary losses):
    python inference.py \
        --input /path/to/sample.nii.gz \
        --input_seg /path/to/sample_seg.nii.gz \
        --checkpoint /path/to/best_model.pth \
        --output_dir /path/to/results

    # With custom config:
    python inference.py \
        --input /path/to/sample.nii.gz \
        --checkpoint /path/to/best_model.pth \
        --config /path/to/config.yaml \
        --output_dir /path/to/results

    # Losses only (skip all visualizations and NIfTI output):
    python inference.py \
        --input /path/to/sample.nii.gz \
        --checkpoint /path/to/best_model.pth \
        --output_dir /path/to/results \
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

# Set to "fomo60k" when running inference on fomo60K targets (fixes OASIS→fomo60K orientation mismatch).
# Set to "" or any other value to skip the fix.
# FLAG = ""
FLAG = "fomo60k"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from model_mri import MRIRegistrationNet
from losses_mri import (
    multi_scale_ncc_loss, mse_loss, dice_loss, focal_loss, boundary_loss,
    smoothness_loss, bending_energy_loss, jacobian_det_loss, displacement_loss,
    lambda_weighted_smoothness, lambda_prior_loss, multi_scale_consistency_loss,
    affine_regularization_loss, affine_orthogonality_loss,
    compute_dice_score, jacobian_det,
)
from get_data_mri import detect_and_correct_inversion


# =============================================================================
# Data Loading
# =============================================================================

def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_nifti(path, target_size):
    """
    Load a NIfTI (.nii.gz) volume, normalize to [0,1], resize, and correct inversion.

    Returns:
        mri: (1, D, H, W) tensor
        affine: NIfTI affine matrix (for saving outputs)
        header: NIfTI header
    """
    if path.endswith(".npy"):
        data = np.load(path).astype(np.float32)
        nifti_affine = np.eye(4)
        nifti_header = None
    else:
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        nifti_affine = img.affine
        nifti_header = img.header

    mri = torch.tensor(data, dtype=torch.float32)
    if mri.ndim == 3:
        mri = mri.unsqueeze(0)  # (1, D, H, W)

    # Normalize to [0, 1]
    mri_min, mri_max = mri.min(), mri.max()
    if mri_max - mri_min > 0:
        mri = (mri - mri_min) / (mri_max - mri_min)

    # Resize
    mri = F.interpolate(
        mri.unsqueeze(0), size=target_size, mode='trilinear', align_corners=False
    ).squeeze(0)

    mri, was_inverted = detect_and_correct_inversion(mri)
    return mri, nifti_affine, nifti_header


LABEL_MAPPING = {
    0: 0, 24: 0,
    3: 1, 42: 1,
    10: 2, 49: 2, 11: 3, 50: 2, 12: 2, 51: 2, 13: 2, 52: 2,
    17: 2, 53: 2, 18: 2, 54: 2, 26: 2, 58: 2, 60: 2, 8: 2, 47: 2,
    2: 3, 41: 3, 7: 3, 46: 3, 16: 3, 28: 3,
    4: 4, 43: 4, 5: 4, 44: 4, 14: 4, 15: 4,
}

def load_seg_nifti(path, target_size, num_classes=5):
    """
    Load a segmentation NIfTI and convert to one-hot (C, D, H, W).

    Supports:
    - Integer label volumes (D, H, W) -> one-hot (FreeSurfer labels remapped via LABEL_MAPPING)
    - Pre-existing one-hot .npy files (C, D, H, W)
    """
    if path.endswith('.npy'):
        seg = np.load(path)
        seg = torch.tensor(seg, dtype=torch.float32)
    else:
        img = nib.load(path)
        data = img.get_fdata().astype(np.int64)
        # Remap FreeSurfer labels to 0..num_classes-1; unlisted labels -> 0
        remapped = np.zeros_like(data)
        for src, dst in LABEL_MAPPING.items():
            remapped[data == src] = dst
        seg_tensor = torch.tensor(remapped)
        seg = F.one_hot(seg_tensor.long(), num_classes).permute(3, 0, 1, 2).float()

    seg = F.interpolate(
        seg.unsqueeze(0), size=target_size, mode='nearest'
    ).squeeze(0)

    return seg


def load_template(cfg, target_size):
    """Load template MRI and segmentation from config paths."""
    template_mri_path = cfg['data']['template_mri_path']
    template_seg_path = cfg['data']['template_seg_path']

    # Template MRI
    tmri = np.load(template_mri_path)
    tmri = torch.tensor(tmri, dtype=torch.float32)
    if tmri.ndim == 3:
        tmri = tmri.unsqueeze(0)
    tmri_min, tmri_max = tmri.min(), tmri.max()
    if tmri_max - tmri_min > 0:
        tmri = (tmri - tmri_min) / (tmri_max - tmri_min)
    tmri = F.interpolate(
        tmri.unsqueeze(0), size=target_size, mode='trilinear', align_corners=False
    ).squeeze(0)
    tmri, _ = detect_and_correct_inversion(tmri)

    # Template segmentation
    tseg = np.load(template_seg_path)
    tseg = torch.tensor(tseg, dtype=torch.float32)
    tseg = F.interpolate(
        tseg.unsqueeze(0), size=target_size, mode='nearest'
    ).squeeze(0)

    return tmri, tseg


# =============================================================================
# Loss Computation
# =============================================================================

def compute_avg_hausdorff_distance(seg_pred, seg_gt, num_classes, voxel_spacing=None):
    """
    Compute Average Hausdorff Distance (AVD) per foreground class.

    For each class, boundaries are extracted via morphological erosion, then
    distance transforms give the minimum distance from every boundary voxel to
    the opposing boundary.  The two directed averages are averaged to yield AHD.

    Args:
        seg_pred: (1, C, D, H, W) or (C, D, H, W) one-hot tensor (predicted / warped)
        seg_gt:   (1, C, D, H, W) or (C, D, H, W) one-hot tensor (ground truth)
        num_classes: total number of classes (including background at index 0)
        voxel_spacing: (dz, dy, dx) voxel size in mm; defaults to isotropic 1 mm

    Returns:
        ahd_per_class: list of AHD values for classes 1..num_classes-1
                       (nan when a class is absent in either mask)
        mean_ahd: mean over valid (non-nan) foreground classes
    """
    if voxel_spacing is None:
        voxel_spacing = (1.0, 1.0, 1.0)

    pred_labels = seg_pred.squeeze(0).argmax(dim=0).cpu().numpy()   # (D, H, W)
    gt_labels   = seg_gt.squeeze(0).argmax(dim=0).cpu().numpy()     # (D, H, W)

    ahd_per_class = []

    for c in range(1, num_classes):  # skip background (class 0)
        pred_mask = (pred_labels == c).astype(np.uint8)
        gt_mask   = (gt_labels   == c).astype(np.uint8)

        if pred_mask.sum() == 0 or gt_mask.sum() == 0:
            ahd_per_class.append(float('nan'))
            continue

        # Extract surface voxels: mask minus its erosion
        pred_boundary = pred_mask ^ binary_erosion(pred_mask).astype(np.uint8)
        gt_boundary   = gt_mask   ^ binary_erosion(gt_mask).astype(np.uint8)

        # Distance transforms: distance from every voxel to the nearest
        # boundary surface of the *other* segmentation
        dt_gt_bnd   = distance_transform_edt(~gt_boundary.astype(bool),   sampling=voxel_spacing)
        dt_pred_bnd = distance_transform_edt(~pred_boundary.astype(bool), sampling=voxel_spacing)

        # Directed average distances
        d_pred_to_gt = float(dt_gt_bnd[pred_boundary.astype(bool)].mean())
        d_gt_to_pred = float(dt_pred_bnd[gt_boundary.astype(bool)].mean())

        ahd_per_class.append((d_pred_to_gt + d_gt_to_pred) / 2.0)

    valid = [v for v in ahd_per_class if not np.isnan(v)]
    mean_ahd = float(np.mean(valid)) if valid else float('nan')
    return ahd_per_class, mean_ahd


def compute_all_losses(warped_mri, sample_mri, final_flow, intermediate_flows,
                       lambda_maps, affine_matrix,
                       warped_seg=None, sample_seg=None, num_classes=5,
                       final_velocity=None):
    """
    Compute all available losses and metrics.

    Intensity and regularization losses are always computed.
    Segmentation losses are computed only when ground truth seg is available.

    Args:
        final_velocity: (B, 3, D, H, W) or None. When provided (SVF mode),
            smoothness / bending / displacement / lambda_smoothness are applied
            to v instead of phi — mirroring MRIRegistrationLoss. jacobian
            stays on the integrated phi.

    Returns:
        dict of {loss_name: float}
    """
    results = {}

    # Target for velocity-domain regularizers (matches MRIRegistrationLoss).
    reg_target = final_velocity if final_velocity is not None else final_flow

    # MRI intensity losses
    results['ncc'] = multi_scale_ncc_loss(warped_mri, sample_mri).item()
    results['mse'] = mse_loss(warped_mri, sample_mri).item()

    # Geometric regularization (smoothness / bending / displacement on v in SVF mode).
    results['smoothness'] = smoothness_loss(reg_target).item()
    results['bending'] = bending_energy_loss(reg_target).item()
    results['jacobian'] = jacobian_det_loss(final_flow).item()
    results['displacement'] = displacement_loss(reg_target).item()

    # Diffeomorphism diagnostics (matches train_mri.py validation loop).
    _, _, D_, H_, W_ = final_flow.shape
    det_ref = (2.0 / D_) * (2.0 / H_) * (2.0 / W_)
    det_norm = jacobian_det(final_flow) / det_ref
    results['folding_pct'] = (det_norm < 0).float().mean().item() * 100.0
    results['min_det'] = det_norm.min().item()
    results['fold_voxels'] = int((det_norm < 0).sum().item())

    # Lambda-based regularization (lambda_smoothness on v in SVF mode)
    if lambda_maps is not None and len(lambda_maps) > 0:
        final_lambda = lambda_maps[-1]
        results['lambda_smoothness'] = lambda_weighted_smoothness(reg_target, final_lambda).item()
        results['lambda_prior'] = lambda_prior_loss(final_lambda, sample_mri).item()

    # Multi-scale consistency
    if intermediate_flows is not None and len(intermediate_flows) > 1:
        results['multi_scale'] = multi_scale_consistency_loss(intermediate_flows).item()

    # Affine regularization
    if affine_matrix is not None:
        results['affine_reg'] = affine_regularization_loss(affine_matrix).item()
        results['affine_ortho'] = affine_orthogonality_loss(affine_matrix).item()

    # Segmentation losses (require ground truth)
    if warped_seg is not None and sample_seg is not None:
        results['dice'] = dice_loss(warped_seg, sample_seg).item()
        results['focal'] = focal_loss(warped_seg, sample_seg).item()
        results['boundary'] = boundary_loss(warped_seg, sample_seg).item()

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
    [0.0, 0.0, 0.0],    # Background - black
    [1.0, 0.3, 0.3],    # Cortex (Gray Matter) - red
    [0.4, 0.8, 0.4],    # Subcortical GM - green
    [1.0, 0.9, 0.3],    # White Matter - yellow
    [0.2, 0.6, 1.0],    # CSF - blue
])


def seg_to_rgb(seg_volume, alpha=0.6):
    """
    Convert a one-hot segmentation (C, D, H, W) to an RGB volume (D, H, W, 3).
    Background is transparent (returns alpha mask too).
    """
    labels = seg_volume.argmax(dim=0).cpu().numpy()  # (D, H, W)
    rgb = CLASS_COLORS[labels]  # (D, H, W, 3)
    mask = (labels > 0).astype(np.float32) * alpha
    return rgb, mask


def plot_slices(mri, seg_rgb, seg_mask, title, save_path, slice_indices=None):
    """
    Plot axial, coronal, sagittal slices with MRI and segmentation overlay.

    Args:
        mri: (D, H, W) numpy array
        seg_rgb: (D, H, W, 3) numpy array
        seg_mask: (D, H, W) numpy array - alpha mask for overlay
        title: figure title
        save_path: where to save the figure
        slice_indices: dict with 'axial', 'coronal', 'sagittal' indices (optional)
    """
    D, H, W = mri.shape
    if slice_indices is None:
        slice_indices = {'axial': D // 2, 'coronal': H // 2, 'sagittal': W // 2}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    views = [
        ('Axial', mri[slice_indices['axial'], :, :],
         seg_rgb[slice_indices['axial'], :, :],
         seg_mask[slice_indices['axial'], :, :]),
        ('Coronal', mri[:, slice_indices['coronal'], :],
         seg_rgb[:, slice_indices['coronal'], :],
         seg_mask[:, slice_indices['coronal'], :]),
        ('Sagittal', mri[:, :, slice_indices['sagittal']],
         seg_rgb[:, :, slice_indices['sagittal']],
         seg_mask[:, :, slice_indices['sagittal']]),
    ]

    for ax, (view_name, mri_slice, seg_slice, mask_slice) in zip(axes, views):
        # MRI background
        ax.imshow(mri_slice, cmap='gray', aspect='equal')
        # Segmentation overlay with alpha blending
        overlay = np.zeros((*mri_slice.shape, 4))
        overlay[..., :3] = seg_slice
        overlay[..., 3] = mask_slice
        ax.imshow(overlay, aspect='equal')
        ax.set_title(f'{view_name} (z={slice_indices["axial"]}, y={slice_indices["coronal"]}, x={slice_indices["sagittal"]})'
                     if view_name == 'Axial' else view_name)
        ax.axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_deformation_field(flow, save_path):
    """
    Visualize deformation field magnitude and component maps.

    Args:
        flow: (3, D, H, W) numpy array
    """
    D, H, W = flow.shape[1:]
    mid = D // 2

    magnitude = np.sqrt((flow ** 2).sum(axis=0))  # (D, H, W)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Top row: magnitude in 3 views
    axes[0, 0].imshow(magnitude[mid, :, :], cmap='hot')
    axes[0, 0].set_title('Deformation Magnitude (Axial)')
    axes[0, 1].imshow(magnitude[:, H // 2, :], cmap='hot')
    axes[0, 1].set_title('Deformation Magnitude (Coronal)')
    axes[0, 2].imshow(magnitude[:, :, W // 2], cmap='hot')
    axes[0, 2].set_title('Deformation Magnitude (Sagittal)')

    # Bottom row: x, y, z components at axial mid-slice
    component_names = ['X (Left-Right)', 'Y (Anterior-Posterior)', 'Z (Superior-Inferior)']
    for i in range(3):
        im = axes[1, i].imshow(flow[i, mid, :, :], cmap='RdBu_r', vmin=-flow.max(), vmax=flow.max())
        axes[1, i].set_title(f'Flow {component_names[i]}')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)

    for ax in axes.flat:
        ax.axis('off')

    fig.suptitle('Deformation Field Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _seg_overlay(ax, mri_slice, labels_slice, alpha=0.6):
    """Draw MRI background then overlay segmentation colours."""
    ax.imshow(mri_slice, cmap='gray', aspect='equal')
    overlay = np.zeros((*mri_slice.shape, 4))
    overlay[..., :3] = CLASS_COLORS[labels_slice]
    overlay[..., 3] = (labels_slice > 0).astype(np.float32) * alpha
    ax.imshow(overlay, aspect='equal')


def plot_comparison(template_mri, sample_mri, warped_mri, warped_seg, sample_seg,
                    save_path, num_classes=5, template_seg=None):
    """
    Side-by-side comparison: template, sample, warped result.
    Row 1: MRI (template | sample | warped).
    Row 2 (when seg available): seg overlaid on MRI
            (template seg on template MRI | GT seg on sample MRI | warped seg on sample MRI).
    """
    t_mri = template_mri.squeeze().cpu().numpy()
    s_mri = sample_mri.squeeze().cpu().numpy()
    w_mri = warped_mri.squeeze().cpu().numpy()
    D = t_mri.shape[0]
    mid = D // 2

    has_seg = sample_seg is not None
    nrows = 2 if has_seg else 1

    fig, axes = plt.subplots(nrows, 3, figsize=(18, 6 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    # Row 1: MRI comparison
    axes[0, 0].imshow(t_mri[mid], cmap='gray')
    axes[0, 0].set_title('Template MRI')
    axes[0, 1].imshow(s_mri[mid], cmap='gray')
    axes[0, 1].set_title('Sample MRI (Target)')
    axes[0, 2].imshow(w_mri[mid], cmap='gray')
    axes[0, 2].set_title('Warped Template MRI')

    # Row 2: Segmentation overlaid on MRI (so background voxels show anatomy, not black)
    if has_seg:
        w_labels = warped_seg.squeeze().argmax(dim=0).cpu().numpy()
        s_labels = sample_seg.squeeze().argmax(dim=0).cpu().numpy()

        if template_seg is not None:
            t_labels = template_seg.squeeze().argmax(dim=0).cpu().numpy()
            _seg_overlay(axes[1, 0], t_mri[mid], t_labels[mid])
            axes[1, 0].set_title('Template Segmentation')
        else:
            axes[1, 0].imshow(t_mri[mid], cmap='gray')
            axes[1, 0].set_title('Template MRI')

        _seg_overlay(axes[1, 1], s_mri[mid], s_labels[mid])
        axes[1, 1].set_title('Ground Truth Segmentation')
        _seg_overlay(axes[1, 2], s_mri[mid], w_labels[mid])
        axes[1, 2].set_title('Warped Template Segmentation')

    for ax in axes.flat:
        ax.axis('off')

    fig.suptitle('Registration Result Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_warped_nifti(warped_seg, nifti_affine, save_path, target_size):
    """Save the warped segmentation as a NIfTI label volume."""
    labels = warped_seg.squeeze().argmax(dim=0).cpu().numpy().astype(np.int16)
    img = nib.Nifti1Image(labels, nifti_affine)
    nib.save(img, save_path)


# =============================================================================
# Main Inference
# =============================================================================

def _print_losses(losses, class_names, has_affine):
    print()
    print("=" * 70)
    print("LOSS VALUES")
    print("=" * 70)

    print("\n  MRI Intensity Losses:")
    print(f"    NCC (multi-scale) : {losses['ncc']:.6f}")
    print(f"    MSE               : {losses['mse']:.6f}")

    print("\n  Geometric Regularization:")
    print(f"    Smoothness        : {losses['smoothness']:.6f}")
    print(f"    Bending energy    : {losses['bending']:.6f}")
    print(f"    Jacobian det      : {losses['jacobian']:.6f}")
    print(f"    Displacement      : {losses['displacement']:.6f}")
    print(f"    Folding %         : {losses['folding_pct']:.4f}%  "
          f"(min det = {losses['min_det']:.4f}, fold voxels = {losses['fold_voxels']})")

    if 'lambda_smoothness' in losses:
        print(f"    Lambda smoothness : {losses['lambda_smoothness']:.6f}")
        print(f"    Lambda prior      : {losses['lambda_prior']:.6f}")
    if 'multi_scale' in losses:
        print(f"    Multi-scale       : {losses['multi_scale']:.6f}")

    if has_affine:
        print("\n  Affine Regularization:")
        print(f"    Affine identity   : {losses['affine_reg']:.6f}")
        print(f"    Affine ortho      : {losses['affine_ortho']:.6f}")

    if 'dice_score' in losses:
        print("\n  Segmentation Losses:")
        print(f"    Dice loss         : {losses['dice']:.6f}")
        print(f"    Focal loss        : {losses['focal']:.6f}")
        print(f"    Boundary loss     : {losses['boundary']:.6f}")
        print(f"\n  Dice Scores:")
        print(f"    Mean Dice         : {losses['dice_score']:.4f}")
        for i, name in enumerate(class_names):
            print(f"    {name:18s}: {losses['dice_per_class'][i]:.4f}")

        print(f"\n  Avg Hausdorff Distance (AVD, mm):")
        print(f"    Mean AVD          : {losses['avg_hausdorff']:.4f}")
        for i, name in enumerate(class_names[1:], start=1):
            val = losses['avg_hausdorff_per_class'][i - 1]
            val_str = f"{val:.4f}" if not np.isnan(val) else "  N/A"
            print(f"    {name:18s}: {val_str}")

    print(f"\n  Flow Statistics:")
    print(f"    Mean magnitude    : {losses['flow_magnitude_mean']:.6f}")
    print(f"    Max magnitude     : {losses['flow_magnitude_max']:.6f}")


def setup_inference(checkpoint_path, config_path=None, device='cuda:0',
                    use_affine=False, use_acm=None, verbose=True):
    """Load config, template, and model once; reuse the returned context across samples.

    use_acm: pass True/False to override; pass None to read from config.acm.enabled.
    """
    cfg = load_config(config_path)
    target_size = tuple(cfg['model']['target_size'])
    num_classes = cfg['model']['num_classes']
    class_names = cfg['visualization']['class_names']
    dev = torch.device(device)

    if use_acm is None:
        use_acm = cfg.get('acm', {}).get('enabled', True)

    flow_cfg = cfg.get('flow', {})
    flow_parameterization = flow_cfg.get('parameterization', 'displacement')
    integration_steps = int(flow_cfg.get('integration_steps', 7))
    velocity_bound = float(flow_cfg.get('velocity_bound', 0.0))
    velocity_field = flow_cfg.get('velocity_field', 'dense')
    cp_spacing = int(flow_cfg.get('cp_spacing', 4))
    cascade_cfg = cfg.get('cascade', {})
    num_stages = int(cascade_cfg.get('num_stages', 1))

    if verbose:
        print("Loading template...")
    template_mri, template_seg = load_template(cfg, target_size)
    template_mri = template_mri.unsqueeze(0).to(dev)
    template_seg = template_seg.unsqueeze(0).to(dev)

    if verbose:
        print("Loading model...")
        print(f"  use_affine={use_affine}, use_acm={use_acm}, "
              f"flow_parameterization={flow_parameterization}")
    model = MRIRegistrationNet(
        seg_channels=num_classes, use_affine=use_affine, use_acm=use_acm,
        target_size=target_size,
        flow_parameterization=flow_parameterization,
        velocity_bound=velocity_bound,
        velocity_field=velocity_field,
        cp_spacing=cp_spacing,
    ).to(dev)

    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    # strict=False: handles pre-SVF checkpoints that lack stn_image / stn_flow
    # id_grid buffers (buffers are recomputed from target_size at init).
    # But a bspline config silently dropping cp_head from a dense checkpoint (or
    # vice versa) would produce garbage warps, so guard the architecture explicitly.
    sd = ckpt['model_state_dict']
    has_cp_head = any(k.startswith('decoder.cp_head') for k in sd)
    if velocity_field == 'bspline':
        assert has_cp_head, (
            "config sets flow.velocity_field='bspline' but the checkpoint has no "
            "decoder.cp_head weights — it was trained with a dense velocity field. "
            "Architecture mismatch; refusing to load silently."
        )
    else:
        assert not has_cp_head, (
            "checkpoint has decoder.cp_head weights (trained with B-spline velocity) "
            "but config requests velocity_field='dense'. Set flow.velocity_field: bspline."
        )
    result = model.load_state_dict(sd, strict=False)
    if verbose:
        print(f"  MISSING:    {result.missing_keys}")
        print(f"  UNEXPECTED: {result.unexpected_keys}")
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
        'use_acm': use_acm,
        'flow_parameterization': flow_parameterization,
        'integration_steps': integration_steps,
        'velocity_bound': velocity_bound,
        'num_stages': num_stages,
        'template_mri': template_mri,
        'template_seg': template_seg,
        'model': model,
    }


@torch.no_grad()
def run_inference_on_sample(ctx, input_path, input_seg_path=None, output_dir=None,
                            losses_only=False, verbose=False):
    """Run inference for a single sample using a pre-built context. Returns the losses dict."""
    target_size  = ctx['target_size']
    num_classes  = ctx['num_classes']
    class_names  = ctx['class_names']
    device       = ctx['device']
    template_mri = ctx['template_mri']
    template_seg = ctx['template_seg']
    model        = ctx['model']
    flow_parameterization = ctx['flow_parameterization']
    integration_steps     = ctx['integration_steps']
    num_stages            = ctx['num_stages']

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Loading input MRI...")
    sample_mri, nifti_affine, _ = load_nifti(input_path, target_size)

    sample_seg = None
    if input_seg_path:
        if verbose:
            print("Loading input segmentation...")
        sample_seg = load_seg_nifti(input_seg_path, target_size, num_classes)

    if FLAG == "fomo60k":
        # Bring fomo60K sample into OASIS template space.
        sample_mri = torch.flip(sample_mri, dims=[3])
        sample_mri = sample_mri.permute(0, 1, 3, 2).contiguous()
        if sample_seg is not None:
            sample_seg = torch.flip(sample_seg, dims=[3])
            sample_seg = sample_seg.permute(0, 1, 3, 2).contiguous()

    sample_mri = sample_mri.unsqueeze(0).to(device)
    if sample_seg is not None:
        sample_seg = sample_seg.unsqueeze(0).to(device)

    # Forward (single source of truth — same path as train_mri.py val).
    from cascade_utils import run_cascade_forward
    out = run_cascade_forward(
        model, model.stn_image, model.stn_flow,
        template_mri, template_seg, sample_mri,
        num_stages=num_stages,
        n_integration_steps=integration_steps,
        flow_parameterization=flow_parameterization,
    )
    final_flow = out['final_flow']
    final_velocity = out['final_velocity']
    intermediate_flows = out['intermediate_velocities']
    lambda_maps = out['lambda_maps']
    affine_matrix = out['affine_matrix']
    warped_mri = out['warped_mri']
    warped_seg = out['warped_seg']

    losses = compute_all_losses(
        warped_mri, sample_mri, final_flow, intermediate_flows,
        lambda_maps, affine_matrix,
        warped_seg=warped_seg, sample_seg=sample_seg, num_classes=num_classes,
        final_velocity=final_velocity,
    )

    if verbose:
        _print_losses(losses, class_names, has_affine=affine_matrix is not None)

    losses_path = output_dir / "losses.json"
    with open(losses_path, 'w') as f:
        json.dump(losses, f, indent=2)
    if verbose:
        print(f"\n  Losses saved to: {losses_path}")

    if losses_only:
        return losses

    if verbose:
        print("\nGenerating visualizations...")

    seg_rgb, seg_mask = seg_to_rgb(warped_seg.squeeze(0))
    s_mri_np = sample_mri.squeeze().cpu().numpy()
    plot_slices(
        s_mri_np, seg_rgb, seg_mask,
        'Warped Template Segmentation on Sample MRI',
        output_dir / 'warped_seg_overlay.png'
    )

    if sample_seg is not None:
        gt_rgb, gt_mask = seg_to_rgb(sample_seg.squeeze(0))
        plot_slices(
            s_mri_np, gt_rgb, gt_mask,
            'Ground Truth Segmentation on Sample MRI',
            output_dir / 'ground_truth_seg_overlay.png'
        )

    flow_np = final_flow.squeeze(0).cpu().numpy()
    plot_deformation_field(flow_np, output_dir / 'deformation_field.png')

    plot_comparison(
        template_mri, sample_mri, warped_mri, warped_seg,
        sample_seg, output_dir / 'comparison.png', num_classes=num_classes,
        template_seg=template_seg
    )

    save_warped_nifti(warped_seg, nifti_affine, str(output_dir / 'warped_seg.nii.gz'), target_size)

    if verbose:
        print(f"\nAll outputs saved to: {output_dir}")

    return losses


def run_inference(args):
    print("=" * 70)
    print("MRI Registration Inference")
    print("=" * 70)
    print(f"Input MRI       : {args.input}")
    print(f"Input Seg       : {args.input_seg or 'None (seg losses will be skipped)'}")
    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Config          : {args.config}")
    print(f"Device          : {args.device}")
    print(f"Output dir      : {args.output_dir}")
    print(f"Losses only     : {args.losses_only}")
    print()

    ctx = setup_inference(args.checkpoint, args.config, args.device,
                          use_affine=True, verbose=True)
    run_inference_on_sample(
        ctx, args.input, args.input_seg, args.output_dir,
        losses_only=args.losses_only, verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(description='MRI Registration Inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input sample MRI (.nii.gz)')
    parser.add_argument('--input_seg', type=str, default=None,
                        help='Path to input sample segmentation (.nii.gz or .npy, optional)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (default: config.yaml in script directory)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (e.g. cuda:0, cpu)')
    parser.add_argument('--losses_only', action='store_true',
                        help='Compute and save losses only; skip all visualizations and NIfTI output')
    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()


