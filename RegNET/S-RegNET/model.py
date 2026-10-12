"""
Seg-only Registration Network

Both inputs are 5-channel one-hot segmentations. Architecture:

    template_seg (5ch) ─┐
    sample_seg   (5ch) ─┼─► [AffineNet] ──► 3×4 affine ──► warps template_seg
                        │                                          │
                        ▼                                          ▼
                   concat (10ch) ──► UNet ──► flow (3ch) + lambda (1ch)
                                                     │
                                                     ▼
                                      SpatialTransformer warps template_seg

Critical invariant: the warp sequence is *affine first, then dense flow*.
Every consumer (train.py, inference.py) must replay both stages in this
order — skipping the affine warp when use_affine=True produces silently
wrong outputs.

The lambda head produces a per-voxel adaptive-smoothness weight
(Sigmoid, λ ∈ [0, 1]); losses.lambda_weighted_smoothness consumes it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building Blocks
# =============================================================================

class AffineNet(nn.Module):
    """
    Predicts a 3x4 affine matrix from concat(template_seg, sample_seg).
    Identity-initialised so training starts from a no-op affine.
    """
    def __init__(self, in_channels=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(64, 12)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
        ], dtype=torch.float))

    def forward(self, template_seg, sample_seg):
        x = torch.cat([template_seg, sample_seg], dim=1)
        f = self.conv(x).view(x.size(0), -1)
        return self.fc(f).view(-1, 3, 4)


class ConvBlock(nn.Module):
    """Conv → InstanceNorm → LeakyReLU, twice."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BandLimitedHead(nn.Module):
    """
    Two-branch band-limited velocity head —
    the per-subject "combo" optimizer's function class, built into the net:

        v = lp96(base_conv(d1)) + up(res_conv(d2))

      base     1x1 conv on the final decoder features, through a FIXED
               parameterless trilinear down-up filter at lowpass_size³ = 3/4 of
               the grid (96³ at 128³ input). Content above that band — the
               band that carries most of the mesh self-intersection — cannot
               be emitted at all, on any input.
      residual 1x1 conv on the second decoder level (dec2, already 64³),
               ZERO-initialised and trilinearly upsampled: the learned stand-in
               for the combo's optimized 64³ delta, which also starts at zero.

    Both resamplings use mode='trilinear', align_corners=False, matching
    bandlimit_opt/instance_opt_bandlimited.py's lowpass() and
    build_velocities() exactly — the distillation targets were built with those
    two operators, so any other choice biases the regression.
    """
    def __init__(self, base_channels=32, res_channels=64, lowpass_size=96):
        super().__init__()
        self.lowpass_size = lowpass_size

        # Near-zero init like the old flow head, so v ≈ 0 → warp ≈ identity.
        self.base_conv = nn.Conv3d(base_channels, 3, kernel_size=1)
        nn.init.normal_(self.base_conv.weight, 0, 1e-3)
        nn.init.zeros_(self.base_conv.bias)

        # Exactly zero: the combo's delta starts at zero, and a warm-started
        # net must reproduce the low-passed checkpoint before it learns.
        self.res_conv = nn.Conv3d(res_channels, 3, kernel_size=1)
        nn.init.zeros_(self.res_conv.weight)
        nn.init.zeros_(self.res_conv.bias)

    def forward(self, d1, d2):
        size = d1.shape[2:]
        d = self.lowpass_size
        base = F.interpolate(self.base_conv(d1), size=(d, d, d),
                             mode='trilinear', align_corners=False)
        base = F.interpolate(base, size=size, mode='trilinear', align_corners=False)
        res = F.interpolate(self.res_conv(d2), size=size,
                            mode='trilinear', align_corners=False)
        return base + res


# =============================================================================
# UNet (single-scale flow + lambda head)
# =============================================================================

class UNet(nn.Module):
    """
    Standard 4-level UNet on concat(template_seg, sample_seg) (10 channels).

    Two heads on the final decoder features:
      - flow head (1x1 Conv3d → 3 channels) with near-zero init so the
        initial warp is identity.
      - lambda head (Conv3d → Sigmoid → 1 channel) producing a per-voxel
        adaptive-smoothness weight in [0, 1].

    head='bandlimited' swaps the flow head for BandLimitedHead, which emits a
    single 3-channel band-limited velocity from dec1 + dec2 instead of the
    6-channel (fw, rv) pair.
    """
    def __init__(self, in_channels=10, out_channels=6, head='default',
                 lowpass_size=96):
        super().__init__()
        self.head = head

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        # Flow head — near-zero init so initial flow ≈ 0 → warp starts at identity.
        if head == 'bandlimited':
            self.vel_head = BandLimitedHead(base_channels=32, res_channels=64,
                                             lowpass_size=lowpass_size)
        else:
            self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)
            nn.init.normal_(self.out_conv.weight, 0, 1e-3)
            nn.init.zeros_(self.out_conv.bias)

        # Lambda head — per-voxel adaptive-smoothness weight in [0, 1].
        #
        # Previous form was a single 1×1 conv → Sigmoid. That collapsed to a
        # constant λ ≈ 0.5 (see dump/2026-05-28_pre-R1/): without a 3D-context
        # stage, each voxel's λ is just a linear projection of its own 32-d
        # feature vector, and the loss landscape (constant-0.5 prior +
        # quadratic-in-λ smoothness) has no incentive to make those vectors
        # encode spatial structure beyond what the flow head already needs.
        #
        # 3×3 ConvBlock → 1×1 → Sigmoid gives ~30k params of spatial context.
        # Final 1×1 is biased so initial λ ≈ 0.8 (sigmoid(1.386)) — i.e.
        # interior-like everywhere — so the head learns to *drop* λ at
        # boundaries from a high start rather than starting collapsed at the
        # weakest-gradient point of the sigmoid.
        self.lambda_head = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.lambda_head[3].weight, 0, 1e-3)
        nn.init.constant_(self.lambda_head[3].bias, 1.386)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        vel = (self.vel_head(d1, d2) if self.head == 'bandlimited'
               else self.out_conv(d1))
        lambda_map = self.lambda_head(d1)
        return vel, lambda_map


# =============================================================================
# Spatial Transformer
# =============================================================================

class SpatialTransformer(nn.Module):
    """
    Dense STN for 3D volumes. Identity grid uses pixel-centre coords
    consistent with align_corners=False; padding_mode='zeros' avoids the
    directional "streak" artifact that border-replication produces when
    flow at the volume boundary points outward.
    """
    def __init__(self, size, device='cpu'):
        super().__init__()
        D, H, W = size

        # Pixel-centre coordinates under align_corners=False: (2i+1)/N - 1.
        # linspace(-1, 1, N) would be the align_corners=True grid; using
        # the wrong one introduces a sub-voxel sampling shift.
        lin_z = (2 * torch.arange(D, device=device).float() + 1) / D - 1
        lin_y = (2 * torch.arange(H, device=device).float() + 1) / H - 1
        lin_x = (2 * torch.arange(W, device=device).float() + 1) / W - 1
        zz, yy, xx = torch.meshgrid(lin_z, lin_y, lin_x, indexing='ij')

        id_grid = torch.stack((xx, yy, zz), dim=-1)
        self.register_buffer('id_grid', id_grid.unsqueeze(0))

    def forward(self, moving, flow):
        """
        Args:
            moving: (B, C, D, H, W)
            flow:   (B, 3, D, H, W) — channel order (x, y, z)
        Returns:
            warped: (B, C, D, H, W)
        """
        B = moving.shape[0]
        flow = flow.permute(0, 2, 3, 4, 1)
        grid = self.id_grid.expand(B, -1, -1, -1, -1)
        warped_grid = grid + flow

        # padding_mode='zeros' (not 'border'): for one-hot seg this maps
        # off-grid regions to all-zeros → argmax picks class 0 (background),
        # which is the correct semantics; 'border' would replicate edge
        # voxels and produce directional streak artifacts.
        return F.grid_sample(
            moving, warped_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )


# =============================================================================
# Main Model
# =============================================================================

class SegRegistrationNet(nn.Module):
    """
    Seg-only registration network. Optionally includes affine pre-alignment.

    Input:
        - template_seg: (B, 5, D, H, W) one-hot
        - sample_seg:   (B, 5, D, H, W) one-hot

    Output:
        - flow_fw:       (B, 3, D, H, W) forward deformation field
        - flow_rv:       (B, 3, D, H, W) reverse deformation field
                         (in affine-aligned space when use_affine=True)
        - lambda_map:    (B, 1, D, H, W) per-voxel adaptive-smoothness
                         weight (Sigmoid output, λ ∈ [0, 1])
        - affine_matrix: (B, 3, 4) or None — predicted affine

    head='bandlimited' (config model.head) ties the two velocities: the UNet
    emits a single band-limited v and the reverse field is -v, so exp(-v) is
    the exact inverse of exp(+v) by construction — the property the mesh push
    rides on. The default 6-channel head predicts the two fields independently.
    """
    def __init__(self, target_size, seg_channels=5, use_affine=False,
                 head='default', lowpass_size=None):
        super().__init__()
        self.use_affine = use_affine
        self.head = head
        if use_affine:
            self.affine_net = AffineNet(in_channels=2 * seg_channels)

        # band limit = 3/4 of the grid unless set: 96 at 128³ (all existing
        # checkpoints), 144 at 192³ (white_r192_ft kept 96, model.lowpass_size)
        self.unet = UNet(in_channels=2 * seg_channels, out_channels=6, head=head,
                          lowpass_size=lowpass_size or target_size[0] * 3 // 4)
        self.stn = SpatialTransformer(size=target_size)

    def forward(self, template_seg, sample_seg, return_velocity=False):
        affine_matrix = None

        if self.use_affine:
            affine_matrix = self.affine_net(template_seg, sample_seg)
            affine_grid = F.affine_grid(affine_matrix, template_seg.size(), align_corners=False)
            template_seg = F.grid_sample(
                template_seg, affine_grid, mode='nearest',
                padding_mode='zeros', align_corners=False,
            )

        x = torch.cat([template_seg, sample_seg], dim=1)
        vel, lambda_map = self.unet(x)
        if self.head == 'bandlimited':
            vel_fw, vel_rv = vel, -vel          # tied: exp(-v) inverts exp(+v)
        else:
            vel_fw = vel[:, :3, ...]
            vel_rv = vel[:, 3:, ...]

        flow_fw = vel_fw / (2 ** 7)
        flow_rv = vel_rv / (2 ** 7)
        for _ in range(7):
            flow_fw = flow_fw + self.stn(flow_fw, flow_fw)
            flow_rv = flow_rv + self.stn(flow_rv, flow_rv)

        # return_velocity exposes the raw UNet-scale forward velocity for the
        # distillation regression term (losses.velocity_mse_loss); every other
        # caller keeps the historical 4-tuple.
        if return_velocity:
            return flow_fw, flow_rv, lambda_map, affine_matrix, vel_fw
        return flow_fw, flow_rv, lambda_map, affine_matrix


# =============================================================================
# Warm start
# =============================================================================

def load_bandlimited_warm_start(model, state_dict):
    """
    Load an OLD 6-channel-head checkpoint into a head='bandlimited' model.

    Trunk, lambda head and affine net load unchanged. `unet.out_conv` has no
    counterpart: its FORWARD half is the velocity the distillation targets were
    built from, so weight[:3]/bias[:3] become the base branch and the residual
    branch keeps its zero init — at which point the net reproduces the
    low-passed checkpoint. The reverse half is discarded; the tied head
    derives it as -v.

    Strict loading cannot survive that shape change, so the surgery is
    explicit and anything else out of place is raised, not swallowed.
    """
    sd = dict(state_dict)
    out_w = sd.pop('unet.out_conv.weight', None)
    out_b = sd.pop('unet.out_conv.bias', None)
    if out_w is not None:
        sd['unet.vel_head.base_conv.weight'] = out_w[:3].clone()
        sd['unet.vel_head.base_conv.bias'] = out_b[:3].clone()

    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith('unet.vel_head.res_conv')]
    if missing or unexpected:
        raise RuntimeError(
            f"band-limited warm start: unexpected checkpoint layout — "
            f"missing {missing}, unexpected {unexpected}")



