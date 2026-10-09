"""
Loss Functions for Seg-only Registration

Composite (sum of weighted terms; weights live in config.yaml -> loss):

    Segmentation alignment
      - dice (excludes background channel 0)
      - cross_entropy (vs argmax of one-hot target)

    Geometric regularization
      - bending energy (second-order smoothness)
      - Jacobian determinant (linear-neg + top-K + pre-fold barrier;
        bulk-mean ReLU(-det) was found insufficient to suppress fold voxels)
      - displacement magnitude

    Lambda-based adaptive regularization (linear-λ + anatomy prior)
      - lambda_weighted_smoothness:  mean( λ_avg · |∇φ|² )       [linear in λ]
      - lambda_prior:                anatomy-aware Gaussian pulling λ toward
                                     (1 - dilated_boundary(sample_seg))

    Affine regularization (only active when affine_matrix is not None)
      - affine_reg:   MSE against the 3x4 identity affine
      - affine_ortho: MSE of RᵀR vs I on the rotation submatrix

Notes:
- The lambda formulations were rewritten after the test_1 baseline collapsed
  to a constant λ ≈ 0.5 over 70 epochs (see dump/2026-05-28_pre-R1/). The
  linear-λ smoothness gives the head a data-dependent gradient; the
  anatomy-aware prior gives it a meaningful target. Together they are the
  minimum change required to break collapse — neither alone is sufficient.
- The Jacobian formulation is the current top-K + pre-fold one, calibrated
  with config weight 0.5 (not the original 0.01 on plain mean(ReLU(-det))).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Segmentation Alignment Losses
# =============================================================================

def dice_loss(y_pred, y_true, smooth=1e-5, class_weights=None):
    """
    Dice loss — foreground classes only (skips class 0 = background).

    Background dominates the volume; including it inflates dice and washes
    out the gradient signal for the tissues we actually care about.
    """
    vol_axes = list(range(2, y_pred.ndim))

    y_pred = y_pred[:, 1:]
    y_true = y_true[:, 1:]

    intersection = (y_pred * y_true).sum(dim=vol_axes)
    union = y_pred.sum(dim=vol_axes) + y_true.sum(dim=vol_axes)

    dice_score = (2. * intersection + smooth) / (union + smooth)

    if class_weights is not None:
        # class_weights has shape (C,). We only want foreground weights
        fg_weights = class_weights[1:].to(dice_score.device)
        weighted_dice = (dice_score * fg_weights).sum(dim=-1) / (fg_weights.sum() + 1e-8)
        return 1 - weighted_dice.mean()
    else:
        return 1 - dice_score.mean()


def cross_entropy_loss(y_pred, y_true, class_weights=None):
    """Cross entropy against the argmax of a one-hot target."""
    if class_weights is not None:
        class_weights = class_weights.to(y_pred.device)
    return F.cross_entropy(y_pred, y_true.argmax(dim=1), weight=class_weights)


# =============================================================================
# Geometric Regularization Losses
# =============================================================================

def bending_energy_loss(flow):
    """Second-order smoothness (bending energy)."""
    d2x = flow[:, :, 2:, :, :] - 2 * flow[:, :, 1:-1, :, :] + flow[:, :, :-2, :, :]
    d2y = flow[:, :, :, 2:, :] - 2 * flow[:, :, :, 1:-1, :] + flow[:, :, :, :-2, :]
    d2z = flow[:, :, :, :, 2:] - 2 * flow[:, :, :, :, 1:-1] + flow[:, :, :, :, :-2]
    return torch.mean(d2x ** 2) + torch.mean(d2y ** 2) + torch.mean(d2z ** 2)


def jacobian_det(flow):
    """
    Per-voxel Jacobian determinant of the warp map T(p) = p + flow(p).

    flow is (B, 3, D, H, W) with channel order (x, y, z) matching the
    SpatialTransformer's id_grid = stack((xx, yy, zz)) — i.e. channel 0
    varies along the W axis, channel 1 along H, channel 2 along D. We
    build warped_grid = id_grid + flow and take per-voxel-step finite
    differences along each spatial axis. Because id_grid already steps
    by (2/W, 2/H, 2/D) per voxel, the identity term and per-axis units
    fall out automatically — det's sign reflects true invertibility.
    """
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
    return det


def jacobian_det_loss(flow):
    """
    Anti-folding penalty on the Jacobian determinant of the warp.

    Three components on det_norm = det / ref where ref = (2/D)(2/H)(2/W)
    is the identity-warp determinant:

    - Linear ReLU on negatives: mean pressure across all folded voxels.
    - Top-K linear hard-example mining: bypasses the volume-mean
      dilution (1/N over millions of voxels) that makes neg.mean()
      contribute negligible per-voxel gradient. K = 0.1% of voxels;
      neg clamped to ≤10 to bound a single catastrophic outlier.
    - Pre-fold barrier: pulls det away from 0 while still positive,
      so the warp doesn't drift into folding in the first place.
    """
    det = jacobian_det(flow)

    _, _, D, H, W = flow.shape
    ref = (2.0 / D) * (2.0 / H) * (2.0 / W)
    det_norm = det / ref

    neg = F.relu(-det_norm)
    neg_linear = neg.mean()

    k = max(1, int(0.001 * neg.numel()))
    topk_neg, _ = torch.topk(neg.flatten(), k)
    neg_topk = topk_neg.sum()

    # Relaxed pre-fold barrier threshold from 0.1 to 0.05
    # Allows sharper local deformations (higher Dice) before penalty kicks in
    near_zero = F.relu(0.05 - det_norm) * (det_norm > 0).float()
    pre_fold = (near_zero ** 2).mean()

    return neg_linear + 0.1 * neg_topk + 0.05 * pre_fold


def displacement_loss(flow):
    """Penalises voxels moving far from origin (mean of squared flow)."""
    return torch.mean(flow ** 2)


# =============================================================================
# Lambda-based Adaptive Regularization (linear-λ + anatomy prior)
# =============================================================================
#
# History: the original (test_1) formulations were *quadratic in λ* with a
# constant-0.5 Gaussian prior (σ=0.1). Visualisation of a 70-epoch run
# (see dump/2026-05-28_pre-R1/) confirmed total collapse — global std of λ
# ≈ 0.003, r(λ, |∇φ|) ≈ 0 with the wrong sign on the worst-folding subjects.
# Two coupled root causes:
#   1. Quadratic-in-λ smoothness has ∂L/∂λ ∝ λ — gradient vanishes precisely
#      at the constant-0.5 equilibrium the prior pins it to.
#   2. The constant-0.5 prior with σ=0.1 has effective coefficient
#      0.05·1/(2·0.01) = 2.5 on (λ−0.5)² — far larger than any λ-dependent
#      term it competes with at init.
#
# The two functions below address both: linear-in-λ smoothness gives the
# head a data-dependent gradient, and an anatomy-derived prior gives it a
# meaningful target instead of a degenerate constant.


def _dilated_boundary(sample_seg, dilate=2):
    """Binary boundary of a one-hot seg, dilated by `dilate` voxels (3D max-pool).

    Args:
        sample_seg: (B, C, D, H, W) one-hot.
        dilate: half-width of the dilation kernel (kernel size = 2·dilate+1).

    Returns:
        (B, 1, D, H, W) float in {0, 1}. 1 = within `dilate` voxels of any
        inter-class transition; 0 = strict interior.

    Implementation: per-axis |Δseg| summed across channels gives a non-zero
    value exactly at class transitions (one-hot → one channel goes +1, another
    −1, abs catches both). max_pool3d with kernel 2·dilate+1 dilates.
    """
    dz = (sample_seg[:, :, 1:, :, :] - sample_seg[:, :, :-1, :, :]).abs()
    dy = (sample_seg[:, :, :, 1:, :] - sample_seg[:, :, :, :-1, :]).abs()
    dx = (sample_seg[:, :, :, :, 1:] - sample_seg[:, :, :, :, :-1]).abs()
    dz = F.pad(dz, (0, 0, 0, 0, 0, 1))
    dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
    dx = F.pad(dx, (0, 1, 0, 0, 0, 0))
    grad = (dz + dy + dx).sum(dim=1, keepdim=True)
    boundary = (grad > 0).float()
    k = 2 * dilate + 1
    return F.max_pool3d(boundary, kernel_size=k, stride=1, padding=dilate)


def lambda_weighted_smoothness(flow, lambda_map):
    """
    Spatially-adaptive smoothness, *linear* in λ:

        loss = mean( λ_avg · |∇φ|² )

    where λ_avg is the per-edge average of adjacent voxels' λ values and
    |∇φ|² is the sum-of-squares of the per-edge flow difference across the
    3 flow components. Critical property vs. the quadratic form:

        ∂L/∂λ = |∇φ|²    (data-dependent, independent of λ)

    so the head receives a real gradient signal everywhere — in particular
    at the local minimum where the quadratic form's gradient vanishes.
    """
    dz = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

    lz = 0.5 * (lambda_map[:, :, 1:, :, :] + lambda_map[:, :, :-1, :, :])
    ly = 0.5 * (lambda_map[:, :, :, 1:, :] + lambda_map[:, :, :, :-1, :])
    lx = 0.5 * (lambda_map[:, :, :, :, 1:] + lambda_map[:, :, :, :, :-1])

    sz = (dz ** 2).sum(dim=1, keepdim=True)
    sy = (dy ** 2).sum(dim=1, keepdim=True)
    sx = (dx ** 2).sum(dim=1, keepdim=True)

    return (lz * sz).mean() + (ly * sy).mean() + (lx * sx).mean()


def lambda_prior_loss(lambda_map, sample_seg, std_val=0.3, dilate=2):
    """
    Anatomy-aware Gaussian prior on λ.

        target = 1 - dilated_boundary(sample_seg)        # interior=1, boundary=0
        loss   = mean( (λ - target)² / (2 · std_val²) )

    The target encodes the actual job we want λ to do: be smooth in tissue
    interiors (λ→1 → full smoothness penalty), allow deformation at tissue
    boundaries (λ→0 → no smoothness penalty there). The constant-0.5 prior
    it replaces gave the head no anatomical signal whatsoever.

    σ widened 0.1 → 0.3 vs. the previous form: effective coefficient
    1/(2σ²) drops 50 → 5.5, so with config weight 0.05 the prior is a soft
    guide (eff ≈ 0.28) rather than the dominant force it was (eff ≈ 2.5).
    """
    with torch.no_grad():
        target = 1.0 - _dilated_boundary(sample_seg, dilate=dilate)
    return torch.mean((lambda_map - target) ** 2 / (2 * std_val ** 2))


# =============================================================================
# Affine Regularization Losses
# =============================================================================

def affine_regularization_loss(affine_matrix):
    """Penalises deviation of the predicted affine from identity."""
    identity = torch.eye(3, 4, device=affine_matrix.device).unsqueeze(0)
    return F.mse_loss(affine_matrix, identity.expand_as(affine_matrix))


def affine_orthogonality_loss(affine_matrix):
    """Encourages the 3x3 rotation submatrix to be orthogonal."""
    R = affine_matrix[:, :3, :3]
    RtR = torch.bmm(R.transpose(1, 2), R)
    I = torch.eye(3, device=R.device).unsqueeze(0).expand_as(RtR)
    return F.mse_loss(RtR, I)


# =============================================================================
# Cycle Consistency
# =============================================================================

def cycle_consistency_loss(flow_fw, flow_rv, stn):
    """Cycle consistency between forward and reverse flows."""
    fw_rv = flow_fw + stn(flow_rv, flow_fw)
    rv_fw = flow_rv + stn(flow_fw, flow_rv)
    return torch.mean(fw_rv ** 2) + torch.mean(rv_fw ** 2)


# =============================================================================
# Comprehensive Loss Class
# =============================================================================

class SegRegistrationLoss(nn.Module):
    """
    Composite loss for seg-only registration.

    Terms (all summed with weights from `weights`):
      - dice, cross_entropy (symmetric)
      - bending, jacobian, displacement (symmetric)
      - lambda_smoothness, lambda_prior
      - cycle
      - affine_reg, affine_ortho  (active only when affine_matrix is not None)

    `weights` is required — config.yaml is the single source of truth. Any
    key missing from `weights` is treated as 0.0 at sum time.
    """
    def __init__(self, weights, class_weights=None):
        super().__init__()
        if weights is None:
            raise ValueError(
                "SegRegistrationLoss requires `weights` (typically config.loss). "
                "Default weights were removed because they drift from config.yaml "
                "and silently caused regressions when config loading was skipped."
            )
        self.weights = weights
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32)

    def forward(self, warped_seg_fw, sample_seg, warped_seg_rv, template_seg,
                flow_fw, flow_rv, lambda_map, stn,
                affine_matrix=None, return_components=False):
        loss_dict = {}

        # Segmentation alignment (symmetric)
        loss_dict['dice'] = dice_loss(warped_seg_fw, sample_seg, class_weights=self.class_weights) + dice_loss(warped_seg_rv, template_seg, class_weights=self.class_weights)
        loss_dict['cross_entropy'] = cross_entropy_loss(warped_seg_fw, sample_seg, class_weights=self.class_weights) + cross_entropy_loss(warped_seg_rv, template_seg, class_weights=self.class_weights)

        # Geometric regularization (symmetric)
        loss_dict['bending'] = bending_energy_loss(flow_fw) + bending_energy_loss(flow_rv)
        loss_dict['jacobian'] = jacobian_det_loss(flow_fw) + jacobian_det_loss(flow_rv)
        loss_dict['displacement'] = displacement_loss(flow_fw) + displacement_loss(flow_rv)

        # Lambda-adaptive (linear-λ smoothness applied to both + anatomy prior)
        loss_dict['lambda_smoothness'] = lambda_weighted_smoothness(flow_fw, lambda_map) + lambda_weighted_smoothness(flow_rv, lambda_map)
        loss_dict['lambda_prior'] = lambda_prior_loss(lambda_map, sample_seg)

        # Cycle consistency
        loss_dict['cycle'] = cycle_consistency_loss(flow_fw, flow_rv, stn)

        # Affine
        if affine_matrix is not None:
            loss_dict['affine_reg'] = affine_regularization_loss(affine_matrix)
            loss_dict['affine_ortho'] = affine_orthogonality_loss(affine_matrix)
        else:
            loss_dict['affine_reg'] = torch.tensor(0.0, device=flow_fw.device)
            loss_dict['affine_ortho'] = torch.tensor(0.0, device=flow_fw.device)

        total_loss = sum(self.weights.get(k, 0.0) * loss_dict[k] for k in loss_dict)

        if return_components:
            return total_loss, loss_dict
        return total_loss


# =============================================================================
# Utility: Dice Score (for evaluation)
# =============================================================================

def compute_dice_score(y_pred, y_true, num_classes=5, epsilon=1e-5):
    """
    Per-class Dice using hard labels (argmax → one-hot).

    Bilinear warping during training produces soft seg values, so we
    convert to hard one-hot here before scoring.
    """
    y_pred_hard = F.one_hot(
        y_pred.argmax(dim=1), num_classes
    ).permute(0, 4, 1, 2, 3).float()

    dice_per_class = []
    vol_axes = [2, 3, 4]

    for c in range(num_classes):
        pred_c = y_pred_hard[:, c:c + 1, ...]
        true_c = y_true[:, c:c + 1, ...]

        inter = 2 * (pred_c * true_c).sum(dim=vol_axes)
        union = pred_c.sum(dim=vol_axes) + true_c.sum(dim=vol_axes)
        dice = (inter + epsilon) / (union + epsilon)
        dice_per_class.append(dice.mean().item())

    fg_mean = sum(dice_per_class[1:]) / (num_classes - 1) if num_classes > 1 else dice_per_class[0]
    return dice_per_class, fg_mean
