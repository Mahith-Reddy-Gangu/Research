"""
Synthetic Warp Generator for Self-Supervised Pretraining

Generates random, smooth, diffeomorphic deformation fields with known
ground-truth velocity for equivariance-based SSL pretraining of S-RegNET.

The velocity field is band-limited by construction: random coefficients
are drawn on a coarse grid and trilinearly upsampled to the target
resolution.  This matches the BandLimitedHead's function class — the
network can only emit velocities with the same frequency content that
SSL trained it on, which is precisely the property that keeps mesh
self-intersections near zero.

Integration uses the same 7-step scaling-and-squaring as model.py
(lines 339–343), so exp(+v) and exp(−v) are exact inverses.

The key mathematical guarantee: a velocity whose maximum spatial
frequency is capped at 1/(2·cell_size) voxels⁻¹ is too smooth to fold
the warp for any realistic magnitude — the Jacobian stays positive
everywhere by construction.  This is the same insight behind
bandlimit_opt/instance_opt_bandlimited.py's lowpass() and
BandLimitedHead's trilinear down-up filter.
"""

import torch
import torch.nn.functional as F


def generate_synthetic_velocity(batch_size, target_size, device,
                                 coarse_sizes=(16, 24, 32),
                                 magnitude_range=(0.02, 0.08),
                                 seed=None):
    if seed is not None:
        g = torch.Generator().manual_seed(int(seed))
    else:
        g = None
    d = coarse_sizes[torch.randint(len(coarse_sizes), (1,), generator=g).item()]
    lo, hi = magnitude_range
    sigma = lo + (hi - lo) * torch.rand(1, generator=g).item()
    vel_coarse = torch.randn(batch_size, 3, d, d, d, generator=g) * sigma
    vel_coarse = vel_coarse.to(device)
    vel = F.interpolate(vel_coarse, size=target_size,
                         mode='trilinear', align_corners=False)
    vel = vel - vel.mean(dim=(2, 3, 4), keepdim=True)
    return vel
    """Generate a random band-limited velocity field.

    Args:
        batch_size:      number of independent velocity fields to generate.
        target_size:     (D, H, W) target volume dimensions.
        device:          torch device.
        coarse_sizes:    tuple of possible coarse-grid sizes; one is chosen
                         uniformly at random per call.  Multiple scales expose
                         the network to multi-frequency deformations:
                           16 → minimum wavelength 16 vox (very smooth)
                           24 → minimum wavelength ~11 vox
                           32 → minimum wavelength  8 vox
        magnitude_range: (lo, hi) — the standard deviation σ of the coarse-grid
                         Gaussian is drawn uniformly from this range.  At 128³:
                           σ = 0.02 → ~1.3 voxel displacement (gentle)
                           σ = 0.05 → ~3.2 voxel displacement (medium)
                           σ = 0.08 → ~5.1 voxel displacement (moderate)

    Returns:
        (B, 3, D, H, W) float32 velocity field at raw UNet scale (i.e. before
        the /2^7 of scaling-and-squaring).  Zero-mean in each spatial component
        so no bulk translation is encoded — any global translation is the
        affine branch's job, not the dense field's.
    """
    # Randomly choose coarse grid size for multi-scale diversity
    d = coarse_sizes[torch.randint(len(coarse_sizes), (1,)).item()]

    # Random magnitude within range
    lo, hi = magnitude_range
    sigma = lo + (hi - lo) * torch.rand(1, device=device).item()

    # Random Gaussian coefficients on coarse grid
    vel_coarse = torch.randn(batch_size, 3, d, d, d, device=device) * sigma

    # Trilinear upsample to target resolution — band-limited by construction.
    # align_corners=False matches the SpatialTransformer and the
    # BandLimitedHead's lowpass/upsample operators.
    vel = F.interpolate(vel_coarse, size=target_size,
                        mode='trilinear', align_corners=False)

    # Remove per-component spatial mean to eliminate bulk translation.
    # Any global rigid component is the AffineNet's job; the dense field
    # should only encode local deformations.
    vel = vel - vel.mean(dim=(2, 3, 4), keepdim=True)
    return vel


def integrate_velocity(vel, stn, n_steps=7):
    """Integrate a velocity field via scaling-and-squaring.

    Byte-for-byte the integration in model.SegRegistrationNet.forward
    (lines 339–343)::

        flow = vel / 2^7
        for _ in range(7):
            flow = flow + stn(flow, flow)

    Args:
        vel:     (B, 3, D, H, W) raw UNet-scale velocity.
        stn:     model.SpatialTransformer instance (same target_size).
        n_steps: number of squaring steps (must be 7 to match the model).

    Returns:
        (B, 3, D, H, W) integrated displacement field.
    """
    flow = vel / (2 ** n_steps)
    for _ in range(n_steps):
        flow = flow + stn(flow, flow)
    return flow


@torch.no_grad()
def generate_synthetic_warp(batch_size, target_size, stn, device,
                             coarse_sizes=(16, 24, 32),
                             magnitude_range=(0.02, 0.08),
                             seed=None):
    vel = generate_synthetic_velocity(batch_size, target_size, device,
                                       coarse_sizes=coarse_sizes,
                                       magnitude_range=magnitude_range,
                                       seed=seed)
    flow_fw = integrate_velocity(vel, stn)
    flow_rv = integrate_velocity(-vel, stn)
    return vel, flow_fw, flow_rv
    """Generate a complete synthetic warp with known ground-truth velocity.

    The returned velocity is the supervision target for
    losses.velocity_mse_loss; the forward flow is used to create the
    synthetic training pair by warping a subject's segmentation.

    All outputs are detached (no grad): the synthetic warp is a fixed
    target, not a learnable parameter.  Gradients flow through the
    *predicted* velocity in the loss, not through the target.

    Args:
        batch_size:      number of independent warps.
        target_size:     (D, H, W) volume dimensions.
        stn:             model.SpatialTransformer (same target_size).
        device:          torch device.
        coarse_sizes:    see generate_synthetic_velocity.
        magnitude_range: see generate_synthetic_velocity.

    Returns:
        velocity: (B, 3, D, H, W) ground-truth velocity (supervision signal).
        flow_fw:  (B, 3, D, H, W) forward flow  = exp(+v).
        flow_rv:  (B, 3, D, H, W) reverse flow  = exp(−v).
    """
    vel = generate_synthetic_velocity(batch_size, target_size, device,
                                      coarse_sizes=coarse_sizes,
                                      magnitude_range=magnitude_range)
    flow_fw = integrate_velocity(vel, stn)
    flow_rv = integrate_velocity(-vel, stn)
    return vel, flow_fw, flow_rv


def warp_seg_nearest(seg, flow, stn):
    """Warp a one-hot segmentation with nearest-neighbor interpolation.

    Preserves the one-hot property (each output voxel maps to exactly one
    class), matching the role of ``sample_seg`` in supervised training — a
    clean, discrete target.  The model's _warp_template uses bilinear (for
    gradient flow in the dice loss); this function is for the *inverse*
    direction: creating the SSL training target, which must be one-hot so
    that:

    1. The dice/CE losses see the same input distribution as supervised
       training (bilinear prediction vs. one-hot target).
    2. The lambda_prior's boundary detection (_dilated_boundary) operates
       on a clean seg with sharp transitions, not blurred ones.

    Uses the same grid convention as SpatialTransformer (id_grid +
    flow, align_corners=False, padding_mode='zeros') so the warp is
    geometrically identical — only the interpolation mode differs.

    Args:
        seg:  (B, C, D, H, W) one-hot segmentation.
        flow: (B, 3, D, H, W) displacement field (from generate_synthetic_warp).
        stn:  model.SpatialTransformer whose id_grid defines the coordinate
              system.

    Returns:
        (B, C, D, H, W) one-hot segmentation warped by flow.
    """
    flow_perm = flow.permute(0, 2, 3, 4, 1)            # (B, D, H, W, 3)
    grid = stn.id_grid.expand(seg.shape[0], -1, -1, -1, -1)
    warped_grid = grid + flow_perm
    return F.grid_sample(seg, warped_grid, mode='nearest',
                         padding_mode='zeros', align_corners=False)
