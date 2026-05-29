"""
Modular Loss Functions for MRI-based Registration

Combines:
1. MRI intensity losses (NCC, MSE)
2. Segmentation alignment losses (Dice, Focal)
3. Geometric regularization (Smoothness, Jacobian, Bending)
4. Lambda-based adaptive regularization
5. Boundary alignment losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MRI Intensity Losses
# =============================================================================

def ncc_loss(y_pred, y_true, win=9):
    """
    Normalized Cross Correlation loss for MRI intensity matching.
    
    Args:
        y_pred: (B, 1, D, H, W) - warped MRI
        y_true: (B, 1, D, H, W) - target MRI
        win: window size for local NCC
    """
    ndims = 3
    sum_filt = torch.ones([1, 1, win, win, win], device=y_pred.device)
    pad_size = win // 2
    
    I = y_true
    J = y_pred
    
    I2 = I * I
    J2 = J * J
    IJ = I * J
    
    I_sum = F.conv3d(I, sum_filt, padding=pad_size)
    J_sum = F.conv3d(J, sum_filt, padding=pad_size)
    I2_sum = F.conv3d(I2, sum_filt, padding=pad_size)
    J2_sum = F.conv3d(J2, sum_filt, padding=pad_size)
    IJ_sum = F.conv3d(IJ, sum_filt, padding=pad_size)
    
    win_size = win ** ndims
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    
    cc = cross * cross / (I_var * J_var + 1e-5)
    
    return 1 - torch.mean(cc)


def multi_scale_ncc_loss(y_pred, y_true, windows=(5, 9, 13)):
    """Multi-scale NCC capturing fine details (win=5) and global patterns (win=13)."""
    return sum(ncc_loss(y_pred, y_true, win=w) for w in windows) / len(windows)


def mse_loss(y_pred, y_true):
    """Mean Squared Error for MRI intensity matching."""
    return F.mse_loss(y_pred, y_true)


# =============================================================================
# Segmentation Alignment Losses
# =============================================================================

def dice_loss(y_pred, y_true, smooth=1e-5):
    """
    Dice loss for segmentation overlap — foreground classes only (skips class 0).

    Args:
        y_pred: (B, C, D, H, W) - warped segmentation
        y_true: (B, C, D, H, W) - target segmentation
    """
    ndims = len(y_pred.shape) - 2
    vol_axes = list(range(2, ndims + 2))

    # Exclude background (class 0) from both prediction and target
    y_pred = y_pred[:, 1:]
    y_true = y_true[:, 1:]

    intersection = (y_pred * y_true).sum(dim=vol_axes)
    union = y_pred.sum(dim=vol_axes) + y_true.sum(dim=vol_axes)

    dice_score = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice_score.mean()


def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0):
    """
    Focal loss for handling class imbalance.
    """
    y_pred = torch.clamp(y_pred, min=1e-7, max=1-1e-7)
    ce_loss = -y_true * torch.log(y_pred)
    focal_weight = (1 - y_pred) ** gamma
    return (alpha * focal_weight * ce_loss).mean()


def boundary_loss(y_pred, y_true):
    """
    Boundary alignment loss — penalizes positional misalignment of edges.

    MSE on edge-magnitude maps is satisfied by any pair of segmentations with
    similar gradient-magnitude *distributions*, even at offset positions, so
    it under-weights translation/shape errors at boundaries. NCC on the same
    maps with a small window penalizes positional drift because two strong
    edges at offset positions correlate poorly even if their magnitudes match.

    Per-class boundary magnitudes are concatenated along the channel axis and
    summed against an all-ones NCC filter — the existing ncc_loss is built for
    single-channel inputs, so we reduce per-class first.
    """
    def compute_boundary(seg):
        grad_x = seg[:, :, :, :, 1:] - seg[:, :, :, :, :-1]
        grad_y = seg[:, :, :, 1:, :] - seg[:, :, :, :-1, :]
        grad_z = seg[:, :, 1:, :, :] - seg[:, :, :-1, :, :]

        grad_x = F.pad(grad_x, (0, 1, 0, 0, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1, 0, 0))
        grad_z = F.pad(grad_z, (0, 0, 0, 0, 0, 1))

        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-8)

    pred_boundary = compute_boundary(y_pred).sum(dim=1, keepdim=True)
    true_boundary = compute_boundary(y_true).sum(dim=1, keepdim=True)

    return ncc_loss(pred_boundary, true_boundary, win=5)


# =============================================================================
# Geometric Regularization Losses
# =============================================================================

def smoothness_loss(flow):
    """First-order smoothness regularization."""
    dx = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

    return torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)


def bending_energy_loss(flow):
    """Second-order smoothness (bending energy)."""
    d2x = flow[:, :, 2:, :, :] - 2*flow[:, :, 1:-1, :, :] + flow[:, :, :-2, :, :]
    d2y = flow[:, :, :, 2:, :] - 2*flow[:, :, :, 1:-1, :] + flow[:, :, :, :-2, :]
    d2z = flow[:, :, :, :, 2:] - 2*flow[:, :, :, :, 1:-1] + flow[:, :, :, :, :-2]
    
    return torch.mean(d2x**2) + torch.mean(d2y**2) + torch.mean(d2z**2)


def jacobian_det(flow):
    """
    Per-voxel Jacobian determinant of the warp map T(p) = p + flow(p).

    flow is (B, 3, D, H, W) with channel order (x, y, z) matching the
    SpatialTransformer's id_grid = stack((xx, yy, zz)) — i.e. channel 0
    varies along the W axis, channel 1 along H, channel 2 along D.
    We build warped_grid = id_grid + flow and take per-voxel-step finite
    differences along each spatial axis. Because id_grid already steps by
    (2/W, 2/H, 2/D) per voxel, the identity term and the per-axis units
    fall out automatically — det's sign reflects true invertibility.
    """
    B, _, D, H, W = flow.shape

    # Build id_grid stepping (xx, yy, zz) -> channel order matches model_mri.py
    lin_z = (2 * torch.arange(D, device=flow.device, dtype=flow.dtype) + 1) / D - 1
    lin_y = (2 * torch.arange(H, device=flow.device, dtype=flow.dtype) + 1) / H - 1
    lin_x = (2 * torch.arange(W, device=flow.device, dtype=flow.dtype) + 1) / W - 1
    zz, yy, xx = torch.meshgrid(lin_z, lin_y, lin_x, indexing='ij')
    id_grid = torch.stack((xx, yy, zz), dim=0).unsqueeze(0)    # (1, 3, D, H, W)

    warped = id_grid + flow                                     # (B, 3, D, H, W)

    # Finite differences along each spatial axis of the warped grid.
    # dW_d[c] = ∂warped[c] / ∂(D-axis index)  — column for d/dz_position
    # dW_h[c] = ∂warped[c] / ∂(H-axis index)  — column for d/dy_position
    # dW_w[c] = ∂warped[c] / ∂(W-axis index)  — column for d/dx_position
    dW_d = warped[:, :, 1:, :, :] - warped[:, :, :-1, :, :]
    dW_h = warped[:, :, :, 1:, :] - warped[:, :, :, :-1, :]
    dW_w = warped[:, :, :, :, 1:] - warped[:, :, :, :, :-1]

    dW_d = F.pad(dW_d, (0, 0, 0, 0, 0, 1))     # pad last D slice
    dW_h = F.pad(dW_h, (0, 0, 0, 1, 0, 0))     # pad last H slice
    dW_w = F.pad(dW_w, (0, 1, 0, 0, 0, 0))     # pad last W slice

    # 3x3 determinant. Rows = warp output channel (x, y, z),
    # columns = spatial direction (d/dz, d/dy, d/dx).
    # Reversing column order flips sign of det, so we reorder columns to
    # (d/dx, d/dy, d/dz) by listing dW_w, dW_h, dW_d below.
    a, b, c = dW_w[:, 0], dW_h[:, 0], dW_d[:, 0]   # ∂x_out / ∂(x,y,z)
    d, e, f = dW_w[:, 1], dW_h[:, 1], dW_d[:, 1]   # ∂y_out / ∂(x,y,z)
    g, h, i = dW_w[:, 2], dW_h[:, 2], dW_d[:, 2]   # ∂z_out / ∂(x,y,z)

    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return det


def jacobian_det_loss(flow):
    """
    Anti-folding penalty on the Jacobian determinant of the warp.

    Computed on the normalised determinant det_norm = det / ref where
    ref = (2/D)(2/H)(2/W) is the identity-warp determinant, so det_norm = 1.0
    at identity and folded voxels have det_norm < 0.

    Three components:

    - Linear ReLU on negatives: broad mean pressure across all folded voxels.
    - Top-K linear (hard-example mining): mean over the worst K folded voxels
      — bypasses the volume-mean dilution that makes a plain `neg.mean()`
      contribute negligible per-voxel gradient (1/N over a 2M-voxel volume).
      Top-K concentrates gradient on actual offenders at 1/k per voxel.

      Earlier code squared `clamp(neg, max=10)` before the top-K. The forward
      cap bounded the loss *value* but not the gradient — `∂L/∂flow` flows
      through `det`'s 3x3 cofactor expansion, whose entries scale as flow².
      A runaway voxel with flow ~5000 produced cofactor ~25M and per-voxel
      gradient ~250k, blowing up training (see epoch-19 worst_det=-5321 in
      the failed run). Linear keeps gradient bounded as `(1/k) × cofactor`,
      which combined with smoothness/bending and `clip_grad_norm_` gives a
      real bound. Cap on `neg` is kept so a single catastrophic outlier
      can't dominate the top-K mean.
    - Pre-fold barrier on positive-but-small det: pulls det away from 0 while
      it's still positive, so the warp doesn't drift into folding in the
      first place. Active only for 0 < det_norm < 0.1.
    """
    det = jacobian_det(flow)

    _, _, D, H, W = flow.shape
    ref = (2.0 / D) * (2.0 / H) * (2.0 / W)
    det_norm = det / ref

    neg = F.relu(-det_norm)
    neg_linear = neg.mean()

    # Top-K linear hard-example mining on the worst folded voxels.
    # K = 0.1% of volume — ~2k voxels at 128^3. Cap at 10 bounds outlier
    # influence on the mean; gradient stays 1/k below the cap, 0 above.
    k = max(1, int(0.001 * neg.numel()))
    topk_neg, _ = torch.topk(neg.flatten(), k)
    neg_topk = topk_neg.clamp(max=10.0).mean()

    near_zero = F.relu(0.1 - det_norm) * (det_norm > 0).float()
    pre_fold = (near_zero ** 2).mean()

    return neg_linear + 0.1 * neg_topk + 0.05 * pre_fold


def displacement_loss(flow):
    """Penalizes large displacements."""
    return torch.mean(flow ** 2)


# =============================================================================
# Lambda-based Adaptive Regularization
# =============================================================================

def lambda_weighted_smoothness(flow, lambda_map):
    """
    Spatially-adaptive smoothness using lambda map.

    Linear in lambda: loss = mean(lambda * |grad_phi|^2). This is the standard
    literature formulation. Earlier code applied lambda inside the gradient
    (`(lambda * grad_phi)^2`), making the penalty quadratic in lambda — which
    introduced a vanishing-gradient pathology at edges (gradient w.r.t. lambda
    scales with lambda itself, so lambda was slow to relax to 0 in exactly the
    regions where we wanted it to). Linear form fixes that.

    High lambda = more smoothness, low lambda = more flexibility.
    """
    dx2 = (flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]) ** 2
    dy2 = (flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]) ** 2
    dz2 = (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]) ** 2

    # Mean across flow channels (x,y,z components), preserve spatial shape.
    dx2 = dx2.mean(dim=1, keepdim=True)
    dy2 = dy2.mean(dim=1, keepdim=True)
    dz2 = dz2.mean(dim=1, keepdim=True)

    # Lambda averaged across each pair of neighbouring voxels — matches the
    # location of the corresponding finite-difference value.
    lx = 0.5 * (lambda_map[:, :, 1:, :, :] + lambda_map[:, :, :-1, :, :])
    ly = 0.5 * (lambda_map[:, :, :, 1:, :] + lambda_map[:, :, :, :-1, :])
    lz = 0.5 * (lambda_map[:, :, :, :, 1:] + lambda_map[:, :, :, :, :-1])

    return (lx * dx2).mean() + (ly * dy2).mean() + (lz * dz2).mean()


_SOBEL_KERNEL_CACHE = {}


def _sobel_gaussian_edge_map(image, sigma=1.5):
    """
    Convolution-derived edge-magnitude map in [0,1] per batch element.

    Sobel-3D along each spatial axis, then a separable Gaussian smooth. Used
    to derive an anatomical target for lambda: edges should have low lambda
    (warp allowed to deform sharply), interiors high lambda (smooth).

    Implementation note: kernels are fixed (no learnable params) and cached
    per (device, dtype) pair to avoid reallocation each forward pass.
    """
    device, dtype = image.device, image.dtype
    key = (device, dtype, float(sigma))
    if key not in _SOBEL_KERNEL_CACHE:
        # 3D Sobel kernels (3, 3, 3) along x, y, z (separable; built explicitly
        # for clarity). Magnitudes match standard 3D Sobel conventions.
        s = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
        d = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype)
        kx = d[None, :].T @ s[None, :]
        kx = kx[:, :, None] * s[None, None, :]
        ky = s[None, :].T @ d[None, :]
        ky = ky[:, :, None] * s[None, None, :]
        kz = s[None, :].T @ s[None, :]
        kz = kz[:, :, None] * d[None, None, :]
        # Stack into (3, 1, 3, 3, 3) so a single conv3d gives 3 channels (gx, gy, gz)
        sobel = torch.stack([kx, ky, kz], dim=0).unsqueeze(1)

        # 1D Gaussian for separable smoothing, length 2*ceil(3*sigma)+1
        radius = int(round(3.0 * sigma))
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        g = g / g.sum()
        _SOBEL_KERNEL_CACHE[key] = (sobel, g, radius)

    sobel, g, radius = _SOBEL_KERNEL_CACHE[key]

    # Edge magnitude
    grads = F.conv3d(image, sobel, padding=1)
    mag = torch.sqrt((grads ** 2).sum(dim=1, keepdim=True) + 1e-8)

    # Separable Gaussian smoothing along each axis
    g_x = g.view(1, 1, 1, 1, -1)
    g_y = g.view(1, 1, 1, -1, 1)
    g_z = g.view(1, 1, -1, 1, 1)
    mag = F.conv3d(mag, g_x, padding=(0, 0, radius))
    mag = F.conv3d(mag, g_y, padding=(0, radius, 0))
    mag = F.conv3d(mag, g_z, padding=(radius, 0, 0))

    # Per-batch [0,1] normalization
    flat = mag.flatten(1)
    lo = flat.amin(dim=1).view(-1, 1, 1, 1, 1)
    hi = flat.amax(dim=1).view(-1, 1, 1, 1, 1)
    return (mag - lo) / (hi - lo + 1e-8)


def lambda_prior_loss(lambda_map, fixed_image):
    """
    Structured prior pulling lambda toward (1 - E) where E is the image edge map.

    The old prior pulled lambda toward a constant 0.5 everywhere, which made
    "adaptive smoothness" uniform smoothness in disguise (lambda collapsed to
    ~0.5 with negligible spatial variance). The new prior is anatomy-aware:

      - High lambda (heavy smoothing) where the image is homogeneous.
      - Low lambda (relaxed smoothing) at boundaries where the warp legitimately
        needs to deform sharply.

    Bilinearly resizes the edge map to the lambda spatial shape when they differ
    (lambda may be produced at a coarser scale than the input image).
    """
    edges = _sobel_gaussian_edge_map(fixed_image)
    if edges.shape[2:] != lambda_map.shape[2:]:
        edges = F.interpolate(edges, size=lambda_map.shape[2:], mode='trilinear',
                              align_corners=False)
    lambda_target = 1.0 - edges
    return F.mse_loss(lambda_map, lambda_target)


# =============================================================================
# Affine Regularization Losses
# =============================================================================

def affine_regularization_loss(affine_matrix):
    """
    Penalizes deviation of the predicted affine from identity.
    Keeps the affine transform small so the deformable stage handles fine detail.
    """
    identity = torch.eye(3, 4, device=affine_matrix.device).unsqueeze(0)
    return F.mse_loss(affine_matrix, identity.expand_as(affine_matrix))


def affine_orthogonality_loss(affine_matrix):
    """
    Encourages the rotation component (3x3 submatrix) to be orthogonal,
    preventing shearing and non-rigid distortions in the affine stage.
    """
    R = affine_matrix[:, :3, :3]
    RtR = torch.bmm(R.transpose(1, 2), R)
    I = torch.eye(3, device=R.device).unsqueeze(0).expand_as(RtR)
    return F.mse_loss(RtR, I)


# =============================================================================
# Multi-scale Consistency
# =============================================================================

def multi_scale_consistency_loss(intermediate_flows):
    """
    Ensures consistency between multi-scale flow/velocity heads.

    In SVF mode this enforces velocity-pyramid (not displacement-pyramid)
    consistency — same MSE-on-upsampled-coarse-vs-fine math, applied to the
    pre-tanh per-scale velocity outputs (postmortem §6.5). The B-spline velocity
    parameterisation has no pyramid, so this is a logged-only zero there.
    """
    if len(intermediate_flows) == 0:
        return torch.tensor(0.0)
    if len(intermediate_flows) < 2:
        return torch.tensor(0.0, device=intermediate_flows[0].device)
    
    total_loss = 0.0
    for i in range(len(intermediate_flows) - 1):
        coarse = intermediate_flows[i]
        fine = intermediate_flows[i + 1]
        
        upsampled = F.interpolate(coarse, size=fine.shape[2:], mode='trilinear', align_corners=False)
        total_loss += F.mse_loss(upsampled, fine)
    
    return total_loss / (len(intermediate_flows) - 1)


# =============================================================================
# Comprehensive Loss Class
# =============================================================================

class MRIRegistrationLoss(nn.Module):
    """
    Comprehensive loss for MRI-based registration.

    Combines MRI intensity matching with segmentation evaluation.
    """
    # Every loss term that appears in `forward`'s loss_dict. Used to catch
    # config typos at __init__ time — a misspelled key (e.g. `boundery`)
    # would otherwise silently weight to 0.0 via `weights.get(k, 0.0)` and
    # only surface as a quiet metric regression days later.
    KNOWN_WEIGHTS = frozenset({
        'ncc', 'mse',
        'dice', 'focal', 'boundary',
        'smoothness', 'bending', 'jacobian', 'displacement',
        'lambda_smoothness', 'lambda_prior',
        'multi_scale',
        'affine_reg', 'affine_ortho',
    })

    def __init__(self, weights):
        super().__init__()
        # `weights` is required — config.yaml is the single source of truth.
        if weights is None:
            raise ValueError(
                "MRIRegistrationLoss requires `weights` (typically config.loss)."
            )
        unknown = set(weights) - self.KNOWN_WEIGHTS
        if unknown:
            raise ValueError(
                f"Unknown loss weight keys in config: {sorted(unknown)}. "
                f"Known keys: {sorted(self.KNOWN_WEIGHTS)}"
            )
        self.weights = weights
    
    def forward(self, warped_mri, sample_mri, warped_seg, sample_seg,
                final_flow, intermediate_flows, lambda_maps,
                affine_matrix=None, final_velocity=None, return_components=False):
        """
        Compute comprehensive loss.

        Args:
            warped_mri: (B, 1, D, H, W) - warped template MRI
            sample_mri: (B, 1, D, H, W) - target MRI
            warped_seg: (B, 5, D, H, W) - warped template segmentation
            sample_seg: (B, 5, D, H, W) - target segmentation (for evaluation)
            final_flow: (B, 3, D, H, W) - integrated displacement (phi)
            intermediate_flows: list of multi-scale heads (velocities in SVF mode,
                displacements in displacement mode)
            lambda_maps: list of lambda maps
            affine_matrix: (B, 3, 4) or None - predicted affine matrix
            final_velocity: (B, 3, D, H, W) or None - predicted velocity v in SVF
                mode. When provided, smoothness/bending/displacement/lambda_smoothness
                are applied to v (not phi). jacobian stays on phi because folding
                is a property of the warp the data actually sees.
            return_components: whether to return individual losses
        """
        loss_dict = {}

        # Target for velocity-domain regularizers: v if available (SVF mode), else phi.
        reg_target = final_velocity if final_velocity is not None else final_flow

        # MRI intensity losses (multi-scale NCC for better detail capture)
        loss_dict['ncc'] = multi_scale_ncc_loss(warped_mri, sample_mri)
        loss_dict['mse'] = mse_loss(warped_mri, sample_mri)

        # Segmentation evaluation losses
        loss_dict['dice'] = dice_loss(warped_seg, sample_seg)
        loss_dict['focal'] = focal_loss(warped_seg, sample_seg)
        loss_dict['boundary'] = boundary_loss(warped_seg, sample_seg)

        # Geometric regularization
        # Smoothness/bending/displacement go on reg_target (v in SVF, phi otherwise).
        # Jacobian stays on the integrated phi — folding is a property of the warp.
        loss_dict['smoothness'] = smoothness_loss(reg_target)
        loss_dict['bending'] = bending_energy_loss(reg_target)
        loss_dict['jacobian'] = jacobian_det_loss(final_flow)
        loss_dict['displacement'] = displacement_loss(reg_target)

        # Lambda-based adaptive regularization (smoothness on reg_target weighted by lambda)
        if lambda_maps is not None and len(lambda_maps) > 0:
            final_lambda = lambda_maps[-1]
            loss_dict['lambda_smoothness'] = lambda_weighted_smoothness(reg_target, final_lambda)
            loss_dict['lambda_prior'] = lambda_prior_loss(final_lambda, sample_mri)
        else:
            loss_dict['lambda_smoothness'] = torch.tensor(0.0, device=final_flow.device)
            loss_dict['lambda_prior'] = torch.tensor(0.0, device=final_flow.device)

        # Multi-scale consistency (velocity pyramid in SVF mode, displacement pyramid otherwise)
        loss_dict['multi_scale'] = multi_scale_consistency_loss(intermediate_flows)

        # Affine regularization
        if affine_matrix is not None:
            loss_dict['affine_reg'] = affine_regularization_loss(affine_matrix)
            loss_dict['affine_ortho'] = affine_orthogonality_loss(affine_matrix)
        else:
            loss_dict['affine_reg'] = torch.tensor(0.0, device=final_flow.device)
            loss_dict['affine_ortho'] = torch.tensor(0.0, device=final_flow.device)

        # Total weighted loss — zero-weight terms are SKIPPED in the sum so that
        # `0.0 * inf = NaN` from a monitored-but-not-penalised term (e.g.
        # jacobian_det_loss producing inf in early epochs) cannot poison the
        # total. The loss_dict still carries the raw value for logging.
        total_loss = sum(
            float(self.weights.get(k, 0.0)) * loss_dict[k]
            for k in loss_dict
            if float(self.weights.get(k, 0.0)) != 0.0
        )

        if return_components:
            return total_loss, loss_dict
        return total_loss


# =============================================================================
# Utility: Dice Score (for evaluation)
# =============================================================================

def compute_dice_score(y_pred, y_true, num_classes=5, epsilon=1e-5):
    """
    Compute per-class Dice scores for evaluation using hard labels.
    
    Bilinear warping produces soft segmentation values, so we convert
    to hard one-hot labels via argmax before computing Dice.
    
    Returns:
        dice_per_class: list of dice scores for each class
        mean_dice: average dice score
    """
    y_pred_hard = F.one_hot(
        y_pred.argmax(dim=1), num_classes
    ).permute(0, 4, 1, 2, 3).float()

    dice_per_class = []
    vol_axes = [2, 3, 4]
    
    for c in range(num_classes):
        pred_c = y_pred_hard[:, c:c+1, ...]
        true_c = y_true[:, c:c+1, ...]
        
        intersection = 2 * (pred_c * true_c).sum(dim=vol_axes)
        union = pred_c.sum(dim=vol_axes) + true_c.sum(dim=vol_axes)
        dice = (intersection + epsilon) / (union + epsilon)
        dice_per_class.append(dice.mean().item())
    
    # Mean over foreground classes only (skip background at index 0)
    fg_mean = sum(dice_per_class[1:]) / (num_classes - 1) if num_classes > 1 else dice_per_class[0]
    return dice_per_class, fg_mean