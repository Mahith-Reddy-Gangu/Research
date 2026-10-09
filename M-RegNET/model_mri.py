"""
MRI-Guided Registration Network with Anatomical Correction Module

Architecture:
- Input: Template MRI (1ch) + Template Seg (5ch) + Sample MRI (1ch)
- Dual-stream encoder: MRI stream + Segmentation attention stream
- Anatomical Correction Module (ACM) in decoder
- Multi-scale deformation with progressive refinement

Key difference from segmentation-only approach:
- MRI provides continuous intensity information
- Segmentation provides anatomical structure guidance
- Combined for robust registration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building Blocks
# =============================================================================

class AffineNet(nn.Module):
    """
    Predicts a 3x4 affine transformation matrix for coarse alignment.

    Takes concatenated [template_mri, sample_mri] as input and predicts
    an affine matrix initialized to identity. Used as a pre-alignment
    step before the deformable registration network.
    """
    def __init__(self, in_channels=2):
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
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(64, 12)
        # Initialize to identity transform
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0
        ], dtype=torch.float))

    def forward(self, template_mri, sample_mri):
        """
        Args:
            template_mri: (B, 1, D, H, W)
            sample_mri: (B, 1, D, H, W)
        Returns:
            affine_matrix: (B, 3, 4)
        """
        x = torch.cat([template_mri, sample_mri], dim=1)
        features = self.conv(x).view(x.size(0), -1)
        affine_params = self.fc(features)
        return affine_params.view(-1, 3, 4)


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """Basic conv block with InstanceNorm and LeakyReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class SegmentationAttentionModule(nn.Module):
    """
    Segmentation Attention Module (SAM)

    Uses template segmentation to generate attention weights that focus
    on anatomically important regions and boundaries.
    """
    def __init__(self, feature_channels, seg_channels):
        super().__init__()

        # Boundary detection from segmentation
        self.boundary_conv = nn.Sequential(
            nn.Conv3d(seg_channels, seg_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(seg_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention generation
        self.attention_conv = nn.Sequential(
            nn.Conv3d(feature_channels + seg_channels, feature_channels, kernel_size=1),
            nn.InstanceNorm3d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Attention weights in [0, 1]
        )
        
        # Feature refinement
        self.refine_conv = nn.Sequential(
            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features, seg_map):
        """
        Args:
            features: (B, C, D, H, W) - MRI feature map
            seg_map: (B, 5, D, H, W) - segmentation map
        Returns:
            refined_features, attention_weights
        """
        if seg_map.shape[2:] != features.shape[2:]:
            seg_map = F.interpolate(seg_map, size=features.shape[2:], mode='nearest')

        # Detect boundaries
        seg_boundaries = self.boundary_conv(seg_map)
        
        # Generate attention
        combined = torch.cat([features, seg_boundaries], dim=1)
        attention_weights = self.attention_conv(combined)
        
        # Apply attention and refine
        attended = features * attention_weights
        refined = self.refine_conv(attended)
        
        return refined, attention_weights


class AnatomicalCorrectionModule(nn.Module):
    """
    Anatomical Correction Module (ACM)
    
    Ensures deformations respect anatomical boundaries by:
    1. Incorporating segmentation structure into decoder features
    2. Encouraging smooth deformations within structures
    3. Allowing flexible deformations at boundaries
    """
    def __init__(self, channels, seg_channels=5):
        super().__init__()
        
        self.correction = nn.Sequential(
            nn.Conv3d(channels + seg_channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features, seg_map):
        """
        Args:
            features: (B, C, D, H, W) - decoder features
            seg_map: (B, 5, D, H, W) - template segmentation
        Returns:
            corrected_features: (B, C, D, H, W)
        """
        # Resize seg_map if needed
        if seg_map.shape[2:] != features.shape[2:]:
            seg_map = F.interpolate(seg_map, size=features.shape[2:], mode='nearest')

        combined = torch.cat([features, seg_map], dim=1)
        return self.correction(combined)


# =============================================================================
# Encoder
# =============================================================================

class DualStreamEncoder(nn.Module):
    """
    Dual-stream encoder:
    - MRI stream: processes [template_mri, sample_mri] (2 channels)
    - Segmentation attention at each scale using template_seg
    """
    def __init__(self, mri_channels=2, seg_channels=5):
        super().__init__()
        
        # MRI encoder stream
        self.enc1 = ConvBlock(mri_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        
        # Segmentation encoder (for attention)
        self.seg_enc1 = ConvBlock(seg_channels, 32)
        self.seg_enc2 = ConvBlock(32, 64)
        self.seg_enc3 = ConvBlock(64, 128)
        self.seg_enc4 = ConvBlock(128, 256)
        
        # Segmentation Attention Modules
        self.sam1 = SegmentationAttentionModule(32, 32)
        self.sam2 = SegmentationAttentionModule(64, 64)
        self.sam3 = SegmentationAttentionModule(128, 128)
        self.sam4 = SegmentationAttentionModule(256, 256)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)
    
    def forward(self, mri_input, seg_input):
        """
        Args:
            mri_input: (B, 2, D, H, W) - concatenated [template_mri, sample_mri]
            seg_input: (B, 5, D, H, W) - template segmentation
        Returns:
            skip_connections, bottleneck_features, attention_maps
        """
        skip_connections = []
        attention_maps = []
        
        # Scale 1
        mri_e1 = self.enc1(mri_input)
        seg_e1 = self.seg_enc1(seg_input)
        fused_e1, attn1 = self.sam1(mri_e1, seg_e1)
        skip_connections.append(fused_e1)
        attention_maps.append(attn1)
        
        # Scale 2
        mri_p1 = self.pool(fused_e1)
        seg_p1 = self.pool(seg_e1)
        mri_e2 = self.enc2(mri_p1)
        seg_e2 = self.seg_enc2(seg_p1)
        fused_e2, attn2 = self.sam2(mri_e2, seg_e2)
        skip_connections.append(fused_e2)
        attention_maps.append(attn2)
        
        # Scale 3
        mri_p2 = self.pool(fused_e2)
        seg_p2 = self.pool(seg_e2)
        mri_e3 = self.enc3(mri_p2)
        seg_e3 = self.seg_enc3(seg_p2)
        fused_e3, attn3 = self.sam3(mri_e3, seg_e3)
        skip_connections.append(fused_e3)
        attention_maps.append(attn3)
        
        # Scale 4
        mri_p3 = self.pool(fused_e3)
        seg_p3 = self.pool(seg_e3)
        mri_e4 = self.enc4(mri_p3)
        seg_e4 = self.seg_enc4(seg_p3)
        fused_e4, attn4 = self.sam4(mri_e4, seg_e4)
        skip_connections.append(fused_e4)
        attention_maps.append(attn4)
        
        # Bottleneck
        bottleneck_in = self.pool(fused_e4)
        bottleneck_out = self.bottleneck(bottleneck_in)
        
        return skip_connections, bottleneck_out, attention_maps


# =============================================================================
# Decoder with ACM
# =============================================================================

def _cubic_bspline_kernel_1d(stride):
    """1D uniform cubic B-spline kernel sampled for integer upsampling `stride`.

    Returns a length-(4*stride - 1) tensor w with w[m] = B(m/stride), B = uniform
    cubic B-spline basis (support |t| < 2). Convolving a stride-spaced lattice of
    control points with this kernel realises cubic B-spline interpolation. The
    kernel has partition-of-unity over the lattice (Σ_k B(t - k) = 1), so a constant
    control grid maps to a constant dense field; it also reproduces linear fields
    exactly (Σ_k k·B(t - k) = t).
    """
    radius = 2 * stride - 1
    m = torch.arange(-radius, radius + 1, dtype=torch.float32)
    t = (m / stride).abs()
    w = torch.zeros_like(t)
    near = t < 1
    far = (t >= 1) & (t < 2)
    w[near] = (3 * t[near] ** 3 - 6 * t[near] ** 2 + 4) / 6
    w[far] = (2 - t[far]) ** 3 / 6
    return w


class CubicBSplineUpsample3d(nn.Module):
    """Upsample a coarse control-point grid to a dense field by cubic B-spline interp.

    Control points sit on a `stride`-spaced lattice of the full-resolution grid; the
    dense field is the cubic B-spline interpolant, computed as a fixed (non-learnable)
    separable convolution on the lattice-stuffed tensor. Output size equals the full
    resolution exactly — no crop bookkeeping.

    The interpolant has spatial gradient bounded by construction
    (‖∇v‖ ≤ C·max|c_i − c_j| / stride, nothing representable below ~2·stride voxels),
    which is what bounds ‖J_v‖ for SVF integration (deep dive §12.4): fold control is
    structural, not a loss penalty. Volume boundaries (last ~2·stride voxels) taper
    toward zero because the lattice support truncates there — benign for registration
    since the volume edge is background and an identity warp there is desired.
    """

    def __init__(self, stride):
        super().__init__()
        self.stride = int(stride)
        k1d = _cubic_bspline_kernel_1d(self.stride)
        L = k1d.numel()
        self.pad = (L - 1) // 2
        k3d = k1d[:, None, None] * k1d[None, :, None] * k1d[None, None, :]  # (L, L, L)
        weight = k3d[None, None].repeat(3, 1, 1, 1, 1)  # (3, 1, L, L, L) — groups=3
        self.register_buffer('weight', weight)

    def forward(self, control_points, out_size):
        """control_points: (B, 3, nd, nh, nw) on the stride lattice.
        out_size: (D, H, W). Returns the dense field (B, 3, D, H, W)."""
        s = self.stride
        stuffed = control_points.new_zeros(control_points.shape[0], 3, *out_size)
        stuffed[..., ::s, ::s, ::s] = control_points
        return F.conv3d(stuffed, self.weight, padding=self.pad, groups=3)


# cp_spacing → (decoder feature attribute used as control-point source, its channels)
_BSPLINE_SOURCE = {2: ('d2', 64), 4: ('d3', 128), 8: ('d4', 256)}


class MultiScaleDecoder(nn.Module):
    """
    Multi-scale decoder with optional Anatomical Correction Module at each scale.
    Progressive deformation refinement from coarse to fine.

    When use_acm=False, ACM modules are not instantiated and decoder features
    flow directly into the flow + lambda heads. Use this to ablate ACM and
    measure whether re-injecting template_seg at each decoder scale actually
    helps beyond what SAM in the encoder already provides via skip connections.

    flow_parameterization='svf' switches the per-scale heads to predict
    velocity components that are summed across scales (coarse-to-fine
    residual stack) and bounded by a soft tanh on the summed velocity. The
    bound is applied once on the summed final velocity, not per scale —
    per-scale tanh would compound saturation through the residual chain
    (deep dive §12.2, postmortem §6.5).

    velocity_bound: B in `v = B·tanh(v_raw/B)`. Set to 0 to disable the
    bound (raw v passed to integrate_svf; only used for ablation).
    """
    def __init__(self, seg_channels=5, use_acm=True,
                 flow_parameterization='displacement', velocity_bound=0.5,
                 velocity_field='dense', cp_spacing=4):
        super().__init__()

        self.use_acm = use_acm
        self.flow_parameterization = flow_parameterization
        self.velocity_bound = float(velocity_bound)
        self.velocity_field = velocity_field
        self.cp_spacing = int(cp_spacing)

        # Upsampling + decoder blocks
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)  # 256 up + 256 skip

        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)  # 128 up + 128 skip

        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)   # 64 up + 64 skip

        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)    # 32 up + 32 skip

        if use_acm:
            self.acm4 = AnatomicalCorrectionModule(256, seg_channels)
            self.acm3 = AnatomicalCorrectionModule(128, seg_channels)
            self.acm2 = AnatomicalCorrectionModule(64, seg_channels)
            self.acm1 = AnatomicalCorrectionModule(32, seg_channels)
        
        # Multi-scale flow outputs
        self.flow4 = nn.Conv3d(256, 3, kernel_size=3, padding=1)
        self.flow3 = nn.Conv3d(128, 3, kernel_size=3, padding=1)
        self.flow2 = nn.Conv3d(64, 3, kernel_size=3, padding=1)
        self.flow1 = nn.Conv3d(32, 3, kernel_size=3, padding=1)

        # B-spline velocity parameterisation: a single coarse control-point head
        # at the decoder scale matching cp_spacing, upsampled to the dense velocity
        # by fixed cubic B-spline interpolation. Replaces the free-form dense field
        # (and the soft tanh bound) with a structurally fold-bounded one (§12.4).
        if velocity_field == 'bspline':
            assert flow_parameterization == 'svf', (
                "velocity_field='bspline' is only defined for the SVF parameterisation"
            )
            assert self.cp_spacing in _BSPLINE_SOURCE, (
                f"cp_spacing must be one of {sorted(_BSPLINE_SOURCE)}; got {self.cp_spacing}"
            )
            self.cp_source, cp_in_ch = _BSPLINE_SOURCE[self.cp_spacing]
            self.cp_head = nn.Conv3d(cp_in_ch, 3, kernel_size=3, padding=1)
            self.bspline_up = CubicBSplineUpsample3d(self.cp_spacing)

        self.max_disp = 0.25
        
        # Lambda maps for adaptive regularization
        self.lambda4 = nn.Sequential(nn.Conv3d(256, 1, kernel_size=3, padding=1), nn.Sigmoid())
        self.lambda3 = nn.Sequential(nn.Conv3d(128, 1, kernel_size=3, padding=1), nn.Sigmoid())
        self.lambda2 = nn.Sequential(nn.Conv3d(64, 1, kernel_size=3, padding=1), nn.Sigmoid())
        self.lambda1 = nn.Sequential(nn.Conv3d(32, 1, kernel_size=3, padding=1), nn.Sigmoid())
        
        # Initialize flow layers to near-zero
        flow_layers = [self.flow1, self.flow2, self.flow3, self.flow4]
        if velocity_field == 'bspline':
            flow_layers.append(self.cp_head)
        for flow_layer in flow_layers:
            nn.init.normal_(flow_layer.weight, 0, 1e-3)
            nn.init.zeros_(flow_layer.bias)
    
    def forward(self, bottleneck, skip_connections, template_seg):
        """
        Args:
            bottleneck: (B, 512, D/16, H/16, W/16)
            skip_connections: [e1, e2, e3, e4] from encoder
            template_seg: (B, 5, D, H, W) - for ACM
        Returns:
            final_head: (B, 3, D, H, W) — velocity v in SVF mode, displacement phi otherwise
            intermediate_heads: list of per-scale outputs (same kind as final_head)
            lambda_maps: list of lambda maps
        """
        # Decoder scale 4
        up4 = self.up4(bottleneck)
        d4 = self.dec4(torch.cat([up4, skip_connections[3]], dim=1))
        if self.use_acm:
            d4 = self.acm4(d4, template_seg)
        lambda4 = self.lambda4(d4)

        # Decoder scale 3
        up3 = self.up3(d4)
        d3 = self.dec3(torch.cat([up3, skip_connections[2]], dim=1))
        if self.use_acm:
            d3 = self.acm3(d3, template_seg)
        lambda3 = self.lambda3(d3)

        # Decoder scale 2
        up2 = self.up2(d3)
        d2 = self.dec2(torch.cat([up2, skip_connections[1]], dim=1))
        if self.use_acm:
            d2 = self.acm2(d2, template_seg)
        lambda2 = self.lambda2(d2)

        # Decoder scale 1 (finest)
        up1 = self.up1(d2)
        d1 = self.dec1(torch.cat([up1, skip_connections[0]], dim=1))
        if self.use_acm:
            d1 = self.acm1(d1, template_seg)
        lambda1 = self.lambda1(d1)

        lambda_maps = [lambda4, lambda3, lambda2, lambda1]

        if self.flow_parameterization == 'svf' and self.velocity_field == 'bspline':
            # B-spline velocity: predict control points at the cp_spacing scale,
            # interpolate to the dense velocity. The cubic B-spline bounds ‖J_v‖
            # by construction, so no tanh bound and no multi-scale velocity sum;
            # the velocity pyramid (and its consistency loss) is therefore empty.
            cp_source = {'d2': d2, 'd3': d3, 'd4': d4}[self.cp_source]
            control_points = self.cp_head(cp_source)
            final_head = self.bspline_up(control_points, out_size=d1.shape[2:])
            intermediate_heads = []
            return final_head, intermediate_heads, lambda_maps

        if self.flow_parameterization == 'svf':
            # Sum-then-bound velocity (deep dive §12.1, postmortem §6.5).
            # Raw per-scale velocities are summed via coarse-to-fine residuals;
            # a single soft tanh bound is applied to the summed final v. Per-
            # scale tanh would compound saturation through the residual chain
            # and break the multi_scale_consistency_loss (which is now on v).
            v4 = self.flow4(d4)
            v3 = self.flow3(d3) + F.interpolate(
                v4, size=d3.shape[2:], mode='trilinear', align_corners=False
            )
            v2 = self.flow2(d2) + F.interpolate(
                v3, size=d2.shape[2:], mode='trilinear', align_corners=False
            )
            v1 = self.flow1(d1) + F.interpolate(
                v2, size=d1.shape[2:], mode='trilinear', align_corners=False
            )

            if self.velocity_bound > 0:
                B = self.velocity_bound
                final_head = B * torch.tanh(v1 / B)
            else:
                final_head = v1

            # Pre-tanh intermediates at native resolutions — keeps
            # multi_scale_consistency_loss in the linear regime.
            intermediate_heads = [v4, v3, v2, v1]
            return final_head, intermediate_heads, lambda_maps

        # Legacy displacement branch — bounded by tanh at every scale.
        m = self.max_disp
        flow4 = m * torch.tanh(self.flow4(d4) / m)
        flow3 = m * torch.tanh(
            (self.flow3(d3) + F.interpolate(flow4, size=d3.shape[2:], mode='trilinear', align_corners=False)) / m
        )
        flow2 = m * torch.tanh(
            (self.flow2(d2) + F.interpolate(flow3, size=d2.shape[2:], mode='trilinear', align_corners=False)) / m
        )
        final_flow = m * torch.tanh(
            (self.flow1(d1) + F.interpolate(flow2, size=d1.shape[2:], mode='trilinear', align_corners=False)) / m
        )
        intermediate_flows = [flow4, flow3, flow2, final_flow]
        return final_flow, intermediate_flows, lambda_maps


# =============================================================================
# Main Model
# =============================================================================

class MRIRegistrationNet(nn.Module):
    """
    MRI-Guided Registration Network

    Optionally includes an affine pre-alignment stage that coarsely aligns the
    template to the sample before predicting a dense deformation field.

    Input:
        - template_mri: (B, 1, D, H, W) - template MRI scan
        - template_seg: (B, 5, D, H, W) - template segmentation (for guidance)
        - sample_mri: (B, 1, D, H, W) - sample MRI to register to

    Output:
        - final_flow: (B, 3, D, H, W) - deformation field (in affine-aligned space when affine is used)
        - intermediate_flows: list of flows at each scale
        - lambda_maps: list of lambda maps for adaptive regularization
        - attention_maps: list of attention maps from SAM
        - affine_matrix: (B, 3, 4) or None - predicted affine matrix
    """
    def __init__(self, seg_channels=5, use_affine=False, use_acm=True,
                 target_size=(128, 128, 128),
                 flow_parameterization='displacement', velocity_bound=0.5,
                 velocity_field='dense', cp_spacing=4):
        super().__init__()

        self.use_affine = use_affine
        self.use_acm = use_acm
        self.flow_parameterization = flow_parameterization
        if use_affine:
            self.affine_net = AffineNet(in_channels=2)

        self.encoder = DualStreamEncoder(mri_channels=2, seg_channels=seg_channels)
        self.decoder = MultiScaleDecoder(
            seg_channels=seg_channels, use_acm=use_acm,
            flow_parameterization=flow_parameterization,
            velocity_bound=velocity_bound,
            velocity_field=velocity_field,
            cp_spacing=cp_spacing,
        )

        # Two STN instances:
        #   stn_image — image/seg warps (out-of-grid → 0).
        #   stn_flow  — flow-on-flow composition inside cascade_utils.integrate_svf
        #               and compose_flows (out-of-grid → border to avoid fake folds).
        # Held on the model so checkpoints carry id_grid buffers and there is
        # no risk of training/eval mismatch.
        self.stn_image = SpatialTransformer(target_size, padding_mode='zeros')
        self.stn_flow = SpatialTransformer(target_size, padding_mode='border')

    def forward(self, template_mri, template_seg, sample_mri):
        """
        Args:
            template_mri: (B, 1, D, H, W)
            template_seg: (B, 5, D, H, W)
            sample_mri: (B, 1, D, H, W)
        """
        affine_matrix = None

        if self.use_affine:
            affine_matrix = self.affine_net(template_mri, sample_mri)
            affine_grid = F.affine_grid(affine_matrix, template_mri.size(), align_corners=False)
            template_mri = F.grid_sample(
                template_mri, affine_grid, mode='bilinear',
                padding_mode='zeros', align_corners=False
            )
            template_seg = F.grid_sample(
                template_seg, affine_grid, mode='nearest',
                padding_mode='zeros', align_corners=False
            )

        # Concatenate MRI inputs
        mri_input = torch.cat([template_mri, sample_mri], dim=1)  # (B, 2, D, H, W)

        # Encode
        skip_connections, bottleneck, attention_maps = self.encoder(mri_input, template_seg)

        # Decode with anatomical correction
        final_flow, intermediate_flows, lambda_maps = self.decoder(
            bottleneck, skip_connections, template_seg
        )

        return final_flow, intermediate_flows, lambda_maps, attention_maps, affine_matrix


# =============================================================================
# Spatial Transformer
# =============================================================================

class SpatialTransformer(nn.Module):
    """
    Spatial Transformer Network for warping volumes.

    padding_mode:
      'zeros'  — used for image/seg warps (`stn_image`). Out-of-grid regions
                 render as background (correct for one-hot seg → class 0 via
                 argmax, and for brain MRI background = 0).
      'border' — used for flow-on-flow composition (`stn_flow`) inside
                 `integrate_svf` and `compose_flows`. Zero-padding on flow
                 composition makes the composed flow jump discontinuously to
                 0 at the boundary and the finite-difference Jacobian sees
                 fake folds there (deep dive §10.1). Border extrapolation
                 keeps the composition smooth.

    Both STNs share the same id_grid math; only the padding behaviour at
    `grid_sample` differs.
    """
    def __init__(self, size, device='cpu', padding_mode='zeros'):
        super().__init__()
        D, H, W = size

        if padding_mode not in ('zeros', 'border'):
            raise ValueError(
                f"SpatialTransformer.padding_mode must be 'zeros' or 'border', got {padding_mode!r}"
            )
        self.padding_mode = padding_mode

        # Pixel-centre coordinates under align_corners=False:  (2*i + 1)/N - 1
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
            flow: (B, 3, D, H, W)
        Returns:
            warped: (B, C, D, H, W)
        """
        B = moving.shape[0]
        flow = flow.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)

        grid = self.id_grid.expand(B, -1, -1, -1, -1)
        warped_grid = grid + flow

        warped = F.grid_sample(
            moving, warped_grid,
            mode='bilinear',
            padding_mode=self.padding_mode,
            align_corners=False
        )

        return warped