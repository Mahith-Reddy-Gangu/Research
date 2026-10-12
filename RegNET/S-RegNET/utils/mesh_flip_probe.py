"""
Why does 2% of the mesh flip when only 0.009% of voxels fold?

This probe does NOT fix anything — it decides which of four mechanisms produces
the mesh-flip / voxel-folding gap, because each one needs a different fix and
three of them are not fixable by damping voxels at all:

  (1) FEW BAD VOXELS, RIPPLE           the working hypothesis. A handful of
      voxels at det << 0 sit under a much larger patch of mesh. Signature:
      flipped triangles cluster tightly around the negative-det set
      (capture_within_2vox high), few negative-det components, min_det very
      negative. Fix = local damper (measured here, --damp).
  (2) SUB-VOXEL NON-INJECTIVITY        the folding metric only checks det on the
      128³ lattice; the mesh is pushed through the *trilinear interpolant* of
      that field and has ~1 mm triangles, i.e. sub-voxel. A field can be
      lattice-positive and still non-injective inside cells. Signature:
      fold_pct rises with --refine (2x/4x lattice), flips are diffuse, and
      flipped triangles sit at small-but-POSITIVE det. Fix = stronger pre-fold
      barrier / coarser velocity, not a damper on negative voxels.
  (3) INTEGRATION / PUSH NUMERICS      scaling-and-squaring with 7 steps, or the
      composed-displacement push, is the thing that folds — not the model.
      Signature: flip falls when --steps rises, or the RK4 ODE push (which never
      composes displacement fields) disagrees with the field push. Fix is free:
      change inference only.
  (4) METRIC ARTIFACT                  dot(n_before, n_after) < 0 flags any
      triangle whose normal rotated past 90°, which a fold-free diffeomorphism
      can do (pure local rotation), and near-degenerate slivers flip on noise.
      Signature: flipped triangles are slivers (low quality) or their rotation
      angle piles up just past 90°. Fix = report a defensible metric.

  (5) SELF-INTERSECTION                the only one of these that is a real
      topology failure, and the only thing the genus-0 deliverable actually
      needs. Moving vertices cannot change V-E+F, so the pushed mesh is
      combinatorially genus-0 no matter how violent the warp; what a warp can
      destroy is the EMBEDDING, i.e. the surface passing through itself.
      dot(n)<0 measures neither. Reported per push, with the un-pushed template
      as the baseline to subtract.

Before any of that it prints a CONTROL: the same mesh pushed by a warp that
provably cannot fold, at the model's own displacement magnitude, evaluated both
exactly and through the 128³ grid. Whatever flip % the control scores is a floor
that belongs to the mesh and the pipeline, not to the model — subtract it first.
The control warp is separable, so its exact endpoint is known in closed form:
each push is also scored against that GROUND TRUTH in voxels, which is what
decides whether the ODE push or the composed-field push is the accurate one when
the two disagree on the model's own field.

Per subject it prints one block; read it in this order:

    control identity/analytic/grid  (4) and (2) floors, before blaming the model
    control err vs exact push      (3) which push to believe, in voxels
    fold@1x / fold@2x / fold@4x    (2) if these climb
    field vs ode push flip         (3) if these disagree
    steps=7 vs 10 flip             (3) if this falls
    flip / flip_nonsliver / angle  (4) if flips are slivers or ~90°
    capture_within_2vox, det@flip  (1) if capture is high and det@flip < 0
    self-intersection              (5) the only line that is a topology failure
    squaring steps sweep           (6) whether s=7 is anywhere near the minimum

--damp then measures the proposed fix without retraining: velocity is scaled
down towards `floor` where det_norm < thresh, Gaussian-smoothed by `sigma`
voxels (a hard scaling would create its own folds), re-integrated, and re-scored
for WM Dice + folding + flip. A global scale is swept alongside it so a local
damper's Dice cost can be judged against the trivial baseline of shrinking the
whole field.

Run from the S-RegNET directory (needs torch + nibabel + scipy):
    python utils/mesh_flip_probe.py --model <run_dir> --device cuda:0 \\
        --num_samples 3 --damp
    # --refine 1,2,4     sub-voxel folding lattices (2 => ~2 GB VRAM, 4 => ~30 GB)
    # --steps 7,10       squaring steps for the inverse (7 = what the model used)
    # --damp_thresh 0.05,0.2  --damp_sigma 1,2  --damp_floor 0.0
    # --global_scale 0.95     velocity scaled everywhere, the honesty baseline
"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root modules

from git_provenance import write_git_sha
from get_data import SegDataset
from inference import load_config, setup_inference
from losses import compute_dice_score, jacobian_det
from visualize_run import resolve_checkpoint, detect_affine
from visualize_mesh import (WM_LABEL, integrate_svf, load_template_mesh, norm_to_world,
                            ref_for, sample_field, undo_affine)

SLIVER_Q = 0.1          # triangle quality below this is a sliver (1 = equilateral)


# =============================================================================
# Field diagnostics
# =============================================================================

def det_norm(flow):
    """Normalized Jacobian det of T(x) = x + flow(x): 1 = volume preserved,
    <= 0 = folded. Same normalization train.py / folding_stats use."""
    _, _, D, H, W = flow.shape
    return jacobian_det(flow) / ((2.0 / D) * (2.0 / H) * (2.0 / W))


def fold_at(flow, factor):
    """(fold %, min det) of the field's TRILINEAR INTERPOLANT on a lattice
    `factor` times finer than the field itself. factor=1 reproduces the reported
    folding number; factor>1 asks whether that number is only an artifact of
    checking injectivity at 2 mm spacing when the mesh is sub-voxel.
    align_corners=False matches grid_sample, so this is the same interpolant the
    mesh push samples."""
    fl = flow if factor == 1 else F.interpolate(flow, scale_factor=factor,
                                                mode='trilinear', align_corners=False)
    d = det_norm(fl)
    return float((d < 0).float().mean() * 100), float(d.min())


def neg_voxel_geometry(det, n):
    """The negative-det set: how many voxels, how many connected blobs, and their
    centres in normalized (x,y,z). 'Few voxels, ripple effect' predicts a small
    count in few blobs."""
    d = det[0].detach().cpu().numpy()
    mask = d < 0
    _lab, n_comp = ndimage.label(mask)
    idx = np.argwhere(mask)                                   # (K,3) as (z,y,x)
    pts = np.stack([(2 * idx[:, 2] + 1) / n - 1,
                    (2 * idx[:, 1] + 1) / n - 1,
                    (2 * idx[:, 0] + 1) / n - 1], axis=1) if len(idx) else np.zeros((0, 3))
    return {'neg_voxels': int(mask.sum()), 'neg_components': int(n_comp),
            'min_det': float(d.min()), 'pct': float(mask.mean() * 100)}, pts.astype(np.float32)


def sample_scalar(vol, pts, device):
    """Trilinear sample of a (1,D,H,W) scalar volume at normalized (x,y,z) pts."""
    g = torch.as_tensor(pts, dtype=torch.float32, device=device).view(1, -1, 1, 1, 3)
    s = F.grid_sample(vol.unsqueeze(1), g, mode='bilinear', padding_mode='border',
                      align_corners=False)
    return s.view(-1).cpu().numpy()


# =============================================================================
# Pushes
# =============================================================================

def ode_push(vel, verts, n_steps=32, sampler=sample_field):
    """Push affine-aligned template verts into subject space by integrating
    dy/dt = -v(y) over unit time with RK4.

    For a stationary velocity field this is the exact inverse map, and unlike
    `v_a + flow_inv(v_a)` it never samples a composed displacement field, so it
    carries no scaling-and-squaring or field-composition error. Disagreement
    between the two pushes is numerics, not the model. Returns (verts, max RK4
    step in voxel units — must stay well under 1).

    `sampler` is the velocity lookup; the control swaps in the analytic warp so
    the integrator can be scored without the grid's discretization on top."""
    h, y, step = 1.0 / n_steps, verts.clone(), 0.0
    for _ in range(n_steps):
        k1 = -sampler(vel, y)
        k2 = -sampler(vel, y + 0.5 * h * k1)
        k3 = -sampler(vel, y + 0.5 * h * k2)
        k4 = -sampler(vel, y + h * k3)
        d = (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        step = max(step, float(d.norm(dim=1).max()))
        y = y + d
    return y, step


def field_push(flow_inv, v_a):
    """The push evaluate_all.py scores: one composed displacement lookup."""
    return v_a + sample_field(flow_inv, v_a)


# =============================================================================
# Triangle analysis
# =============================================================================

def tri_geom(v, faces):
    """Per-triangle (raw normal, area, quality, centroid). quality = 1 for an
    equilateral triangle, -> 0 for a sliver; slivers reverse on interpolation
    noise alone, so the flip % must be reported with and without them."""
    v0, v1, v2 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    l2 = (((v1 - v0) ** 2).sum(1) + ((v2 - v1) ** 2).sum(1) + ((v0 - v2) ** 2).sum(1))
    q = 4 * np.sqrt(3) * area / np.maximum(l2, 1e-20)
    return nrm, area, q, (v0 + v1 + v2) / 3.0


def flip_mask(v_before, v_after, faces):
    nb, _a, _q, _c = tri_geom(v_before, faces)
    na, _a2, _q2, _c2 = tri_geom(v_after, faces)
    cos = (nb * na).sum(1) / np.maximum(np.linalg.norm(nb, axis=1) * np.linalg.norm(na, axis=1), 1e-20)
    return cos < 0, np.degrees(np.arccos(np.clip(cos, -1, 1)))


def control_flips(v_a, v_a_np, faces, n, amp, cycles, ode_steps, stn, steps, base):
    """Flip % from a warp that PROVABLY cannot fold: u_i = amp·sin(pi·c·x_i), whose
    Jacobian is diagonal with entries 1 + amp·c·pi·cos(...) > 0 whenever
    amp·c·pi < 1. Pushed three ways at the model's own displacement magnitude:

      identity   the metric's zero point — must come out exactly 0.
      analytic   u evaluated exactly at the vertices. Any flip here is the METRIC
                 (slivers, or normals rotating past 90 degrees at det > 0) or the
                 mesh itself, and no field-side fix can remove it.
      grid       the SAME field rasterized to the n^3 lattice and trilinearly
                 sampled — the model's own pipeline carrying a known-good field.
                 analytic -> grid is the discretization floor every push pays on
                 a 2 mm grid with ~1 mm triangles.

    Subtract these from the model's flip % before blaming the model."""
    cycles = min(cycles, 0.9 / (amp * np.pi))            # keep the Jacobian margin
    ident, _ = flip_mask(v_a_np, v_a_np, faces)
    an, _ = flip_mask(v_a_np, v_a_np + amp * np.sin(np.pi * cycles * v_a_np), faces)

    lin = (2 * torch.arange(n, device=v_a.device, dtype=torch.float32) + 1) / n - 1
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing='ij')
    u = (amp * torch.sin(np.pi * cycles * torch.stack((xx, yy, zz)))).unsqueeze(0)
    gr, _ = flip_mask(v_a_np, (v_a + sample_field(u, v_a)).cpu().numpy(), faces)
    # Same field read as a VELOCITY: its flow is a diffeomorphism whatever the
    # amplitude, so any flip the ODE push scores here is the ODE path's own noise
    # (32 trilinear velocity lookups per vertex), not the model.
    v_ode, _s = ode_push(u, v_a, ode_steps)
    od, _ = flip_mask(v_a_np, v_ode.cpu().numpy(), faces)

    # --- ground truth -------------------------------------------------------
    # u is separable and axis-aligned, so dy/dt = -u(y) decouples into three
    # copies of dx/dt = -a sin(Kx), whose exact solution is
    #     tan(K x(t)/2) = tan(K x0/2) e^{-aKt},
    # the branch pinned by the fixed point (multiple of pi) that x0 sits next to.
    # Flip % alone can only say the two pushes DISAGREE; this says which one is
    # wrong, in voxels, on a field where the answer is known.
    K = np.pi * cycles
    phi = K * v_a_np.astype(np.float64) / 2.0
    m = np.floor(phi / np.pi + 0.5)
    exact = (2.0 / K) * (np.arctan(np.tan(phi - m * np.pi) * np.exp(-amp * K)) + m * np.pi)

    def err(p):                                          # (max, mean) vertex error, voxels
        d = np.linalg.norm(p - exact, axis=1) * (n / 2.0)
        return float(d.max()), float(d.mean())

    # analytic sampler = the integrator alone; grid sampler adds the 128^3
    # discretization of the velocity; field adds scaling-and-squaring on top.
    v_ode_an, _ = ode_push(None, v_a, ode_steps,
                           sampler=lambda _f, y: amp * torch.sin(np.pi * cycles * y))
    # Squaring sweep against the closed form. Scaling-and-squaring balances two
    # errors that move in opposite directions: the truncation of exp(v/2^s) by
    # its first order term, which falls as s rises, against one trilinear
    # resampling of the whole field per squaring, which accumulates as s rises.
    # So the total need not be monotone in s, and the conventional s=7 is a
    # convention, not a minimum. This measures where the minimum actually is.
    sweep = {}
    for s in sorted(set(steps) | {base}):                # base = the model's own s
        vf = (v_a + sample_field(integrate_svf(-u, stn, s), v_a)).cpu().numpy()
        mx, mn = err(vf.astype(np.float64))
        sweep[str(s)] = {'max_vox': mx, 'mean_vox': mn}
    return {'amp': float(amp), 'cycles': float(cycles),
            'jac_margin': float(1 - amp * cycles * np.pi),
            'identity_flip_pct': float(ident.mean() * 100),
            'analytic_flip_pct': float(an.mean() * 100),
            'grid_flip_pct': float(gr.mean() * 100),
            'ode_flip_pct': float(od.mean() * 100),
            'err_ode_analytic_vox': err(v_ode_an.cpu().numpy().astype(np.float64))[0],
            'err_ode_grid_vox': err(v_ode.cpu().numpy().astype(np.float64))[0],
            'err_field_vox': sweep[str(base)]['max_vox'],
            'err_field_base_steps': int(base),
            'squaring_sweep': sweep}


def cluster_count(faces, sel):
    """Connected components of the SELECTED triangles (adjacency = shared vertex).
    A ripple from a few bad voxels shows up as few large clusters; a diffuse
    sub-voxel problem shows up as thousands of singletons."""
    f = faces[sel]
    if len(f) == 0:
        return 0, 0
    rows = np.repeat(np.arange(len(f)), 3)
    inc = coo_matrix((np.ones(rows.size), (rows, f.ravel())),
                     shape=(len(f), int(faces.max()) + 1))
    n_c, lab = connected_components(inc @ inc.T, directed=False)
    return int(n_c), int(np.bincount(lab).max())


# =============================================================================
# Self-intersection — the real topology test
# =============================================================================

def seg_hits_tri(P, Q, T, eps=1e-14):
    """Möller-Trumbore: does the segment P->Q pierce triangle T? All (N,3)/(N,3,3)."""
    D, e1, e2 = Q - P, T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    pv = np.cross(D, e2)
    det = (e1 * pv).sum(1)
    ok = np.abs(det) >= eps                       # parallel segment: no crossing
    inv = np.zeros_like(det)
    np.divide(1.0, det, out=inv, where=ok)
    tv = P - T[:, 0]
    qv = np.cross(tv, e1)
    u = (tv * pv).sum(1) * inv                    # barycentric in T
    w = (D * qv).sum(1) * inv
    t = (e2 * qv).sum(1) * inv                    # position along the segment
    return ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t >= 0) & (t <= 1)


def tri_tri_cross(A, B):
    """Do the paired triangles A[i], B[i] cross? Two triangles intersect iff an
    edge of one pierces the other, so six segment tests decide it. Coplanar
    overlap is measure-zero and is not detected."""
    hit = np.zeros(len(A), dtype=bool)
    for X, Y in ((A, B), (B, A)):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            hit |= seg_hits_tri(X[:, a], X[:, b], Y)
    return hit


# Broadphase lives in repair_template_mesh (numpy only, no torch) so there is ONE
# implementation of it. The radius-split KD-tree version that used to be here still
# made every triangle pay for the query radius of the most stretched one: on a
# pushed cortical mesh that cost 311M candidates for 954K real narrowphase pairs
# (99.7% waste, ~5 GB of pair array, which is what made this pass thrash rather
# than finish). Inserting each triangle into only the cells its own AABB covers
# needs 72M for the identical result — 4.3x fewer, 1.16 GB, 17s.
from repair_template_mesh import cell_pairs                      # noqa: E402


def self_intersections(v, faces, label='', chunk=500_000):
    """Non-adjacent triangle pairs of this mesh that actually cross.

    This is what the genus-0 deliverable needs and what dot(n_before,n_after)<0
    does not measure. Moving vertices cannot change V-E+F, so the pushed mesh is
    combinatorially genus-0 however violent the warp; the failure a warp CAN
    cause is the surface passing through itself. A triangle whose normal merely
    rotated past 90° under shear is a perfectly embedded triangle and shows up
    here as clean — which is the point of measuring it.

    Broadphase is a uniform grid over triangle AABBs (see cell_pairs) narrowed by
    the boxes themselves; pairs sharing a vertex are dropped, since neighbours
    always touch at it."""
    tri = v[faces].astype(np.float64)
    cen, lo, hi = tri.mean(1), tri.min(1), tri.max(1)
    rad = np.linalg.norm(tri - cen[:, None, :], axis=2).max(1)
    pairs = cell_pairs(lo, hi, 2.0 * float(np.percentile(rad, 99.0)))
    print(f"      [self-int {label}] {len(pairs):,} candidates", end='', flush=True)

    hit, n_pairs, n_narrow = np.zeros(len(faces), dtype=bool), 0, 0
    for s in range(0, len(pairs), chunk):
        i, j = pairs[s:s + chunk].T
        m = (lo[i] <= hi[j]).all(1) & (lo[j] <= hi[i]).all(1)          # boxes overlap
        i, j = i[m], j[m]
        m = ~(faces[i][:, :, None] == faces[j][:, None, :]).any((1, 2))  # not neighbours
        i, j = i[m], j[m]
        n_narrow += len(i)
        x = tri_tri_cross(tri[i], tri[j])
        n_pairs += int(x.sum())
        hit[i[x]] = True
        hit[j[x]] = True
    print(f" -> {n_narrow:,} narrowphase -> {n_pairs:,} crossings", flush=True)

    n_c, big = cluster_count(faces, hit)
    return {'si_candidates': int(len(pairs)), 'si_narrowphase': n_narrow,
            'si_pairs': n_pairs, 'si_faces': int(hit.sum()),
            'si_faces_pct': float(hit.mean() * 100),
            'si_clusters': n_c, 'si_largest': big}, hit


# =============================================================================
# PLY export
# =============================================================================

def write_ply(path, verts, faces, color):
    """Binary little-endian PLY with per-VERTEX colour. Vertex colour, not face
    colour: self-intersection is a per-face property, but face colour is the
    half of the spec that online viewers routinely ignore, so a vertex is
    painted if any triangle it belongs to is flagged."""
    vd = np.empty(len(verts), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for k, i in (('x', 0), ('y', 1), ('z', 2)):
        vd[k] = verts[:, i]
    for k, i in (('red', 0), ('green', 1), ('blue', 2)):
        vd[k] = color[:, i]
    fd = np.empty(len(faces), dtype=[('n', 'u1'), ('a', '<i4'), ('b', '<i4'), ('c', '<i4')])
    fd['n'] = 3
    for k, i in (('a', 0), ('b', 1), ('c', 2)):
        fd[k] = faces[:, i]
    assert vd.itemsize == 15 and fd.itemsize == 13, 'structured dtype got padded'
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {len(faces)}\nproperty list uchar int vertex_indices\n"
            "end_header\n")
    with open(path, 'wb') as f:
        f.write(head.encode('ascii'))
        f.write(vd.tobytes())
        f.write(fd.tobytes())


def selfint_colors(faces, hit, base, n_verts):
    """Vertex colours for a self-intersection PLY.

        grey   clean
        amber  crossed itself in the TEMPLATE too — the template subject's own
               surface defect, carried through the warp rather than caused by it
        red    the warp created this crossing

    Red is painted second so a vertex touching both reads as red."""
    col = np.full((n_verts, 3), 200, np.uint8)
    for sel, rgb in ((hit & base, (255, 170, 0)), (hit & ~base, (220, 30, 30))):
        col[np.unique(faces[sel])] = rgb
    return col


def write_selfint_ply(path, verts_norm, faces, colors, ref):
    """Coordinates go through the template's reference volume, so the model is in
    mm and the right shape; it is not registered to a subject's own volume."""
    write_ply(path, norm_to_world(verts_norm, ref), faces, colors)
    print(f"      [ply] {path.name}  ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def analyse_flips(v_before, v_after, faces, det_vol, neg_pts, n, device):
    """Everything needed to separate the four mechanisms, for one push."""
    flip, angle = flip_mask(v_before, v_after, faces)
    _nb, area_b, q, cen = tri_geom(v_before, faces)
    _na, area_a, _q2, _c = tri_geom(v_after, faces)

    det_c = sample_scalar(det_vol, cen, device)          # det of the pushed field
    solid = q > SLIVER_Q
    vox = n / 2.0                                        # voxels per normalized unit
    if len(neg_pts):
        dist = cKDTree(neg_pts).query(cen)[0] * vox
    else:
        dist = np.full(len(cen), np.inf)

    n_c, big = cluster_count(faces, flip)
    out = {
        'flip_pct': float(flip.mean() * 100),
        'flip_pct_nonsliver': float(flip[solid].mean() * 100),
        'sliver_pct_of_mesh': float((~solid).mean() * 100),
        'sliver_share_of_flips': float((~solid)[flip].mean() * 100) if flip.any() else 0.0,
        'flip_clusters': n_c,
        'flip_largest_cluster': big,
        # (4) rotation just past 90 degrees = a borderline normal, not a fold.
        'flip_angle_median_deg': float(np.median(angle[flip])) if flip.any() else float('nan'),
        'flip_angle_p10_deg': float(np.percentile(angle[flip], 10)) if flip.any() else float('nan'),
        'flip_under_100deg_pct': float((angle[flip] < 100).mean() * 100) if flip.any() else 0.0,
        # (1) vs (2): is there actually a fold under the flipped triangle?
        'det_at_flip_median': float(np.median(det_c[flip])) if flip.any() else float('nan'),
        'det_at_flip_p5': float(np.percentile(det_c[flip], 5)) if flip.any() else float('nan'),
        'flip_with_positive_det_pct': float((det_c[flip] > 0).mean() * 100) if flip.any() else 0.0,
        'det_at_all_median': float(np.median(det_c)),
        # (1): do the flips sit on the negative-det set?
        'capture_within_2vox_pct': float((dist[flip] <= 2).mean() * 100) if flip.any() else 0.0,
        'capture_within_4vox_pct': float((dist[flip] <= 4).mean() * 100) if flip.any() else 0.0,
        'base_rate_within_2vox_pct': float((dist <= 2).mean() * 100),
        'dist_to_neg_median_vox': float(np.median(dist[flip])) if flip.any() else float('nan'),
        'area_ratio_median': float(np.median(area_a[flip] / np.maximum(area_b[flip], 1e-20)))
                             if flip.any() else float('nan'),
    }
    return out, flip


# =============================================================================
# Damper (measured, not installed)
# =============================================================================

def gauss_blur(x, sigma):
    """Separable Gaussian over the spatial axes of (B,C,D,H,W)."""
    if sigma <= 0:
        return x
    r = max(1, int(round(3 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    C = x.shape[1]
    for ax in (2, 3, 4):
        shape = [1, 1, 1, 1, 1]
        shape[ax] = 2 * r + 1
        w = k.view(shape).expand(C, 1, -1, -1, -1).contiguous()
        pad = [0] * 6
        i = {2: 4, 3: 2, 4: 0}[ax]                     # F.pad order is W,H,D
        pad[i] = pad[i + 1] = r
        x = F.conv3d(F.pad(x, pad, mode='replicate'), w, groups=C)
    return x


def damp_velocity(vel, det, thresh, sigma, floor):
    """Scale the velocity towards `floor` where the composed field is folded or
    nearly folded, 1 where det_norm >= thresh, smoothed over `sigma` voxels.
    Smoothing is not cosmetic: a hard per-voxel scaling has its own large
    gradients and can fold on its own."""
    w = (det.unsqueeze(1) / thresh).clamp(0.0, 1.0)
    s = floor + (1.0 - floor) * gauss_blur(w, sigma)
    return vel * s, s


# =============================================================================
# Per-subject
# =============================================================================

@torch.no_grad()
def recover_velocity(model, template_seg, sample_seg):
    """The model's own front half (affine head + UNet with the mode='nearest'
    affine warp forward() uses) -> (velocity, affine, model flow_fw). The parity
    check against model flow_fw is the guard that this replication has not
    drifted from model.py."""
    affine, tmpl = None, template_seg
    if model.use_affine:
        affine = model.affine_net(template_seg, sample_seg)
        grid = F.affine_grid(affine, template_seg.size(), align_corners=False)
        tmpl = F.grid_sample(template_seg, grid, mode='nearest',
                             padding_mode='zeros', align_corners=False)
    vel, _lam = model.unet(torch.cat([tmpl, sample_seg], dim=1))
    flow_fw_model, _rv, _l, _a = model(template_seg, sample_seg)
    return vel[:, :3], affine, flow_fw_model


@torch.no_grad()
def score_flow(ctx, flow_fw, affine, sample_seg):
    """WM Dice + folding for a candidate forward field, replaying training's warp
    (affine first, then dense flow)."""
    template_seg, stn = ctx['template_seg'], ctx['stn']
    moving = template_seg
    if affine is not None:
        grid = F.affine_grid(affine, template_seg.size(), align_corners=False)
        moving = F.grid_sample(template_seg, grid, mode='bilinear',
                               padding_mode='zeros', align_corners=False)
    dice_pc, fg = compute_dice_score(stn(moving, flow_fw), sample_seg, ctx['num_classes'])
    fold, mind = fold_at(flow_fw, 1)
    return {'dice_wm': dice_pc[WM_LABEL], 'dice_fg': fg, 'fold_pct': fold, 'min_det': mind}


@torch.no_grad()
def probe_subject(ctx, verts_t, verts_np, faces, sample_seg, args, out=None, subject=''):
    model, device, n = ctx['model'], ctx['device'], ctx['target_size'][0]
    vel, affine, flow_fw_model = recover_velocity(model, ctx['template_seg'], sample_seg)

    base_steps = args.steps[0]
    flow_fw = integrate_svf(vel, model.stn, base_steps)
    parity = float((flow_fw - flow_fw_model).abs().max())
    flow_inv = integrate_svf(-vel, model.stn, base_steps)

    rep: dict = {'svf_parity_max': parity}
    rep.update(score_flow(ctx, flow_fw_model, affine, sample_seg))
    # (2) sub-voxel: same field, finer lattice. Both directions — the mesh is
    # pushed by the INVERSE, so its folding is the one that can flip triangles.
    for f in args.refine:
        rep[f'fold_fw@{f}x'], rep[f'min_det_fw@{f}x'] = fold_at(flow_fw, f)
        rep[f'fold_inv@{f}x'], rep[f'min_det_inv@{f}x'] = fold_at(flow_inv, f)

    det_inv = det_norm(flow_inv)
    neg_geom, neg_pts = neg_voxel_geometry(det_inv, n)
    rep.update({f'inv_{k}': v for k, v in neg_geom.items()})

    v_a = undo_affine(verts_t, affine).contiguous()
    v_a_np = v_a.cpu().numpy()
    pushes = {f'field_s{base_steps}': field_push(flow_inv, v_a)}
    # (3) numerics: more squaring steps, and a push that composes nothing.
    for s in args.steps[1:]:
        pushes[f'field_s{s}'] = field_push(integrate_svf(-vel, model.stn, s), v_a)
    v_ode, ode_step = ode_push(vel, v_a, args.ode_steps)
    pushes['ode'] = v_ode
    rep['ode_max_step_vox'] = ode_step * (n / 2.0)      # must stay well under 1
    # Convergence, not agreement: the ODE and the field push are two computations
    # of the SAME map, so if they disagree one of them is unconverged. A 4x-finer
    # ODE that lands in the same place says the ODE is right and the composed
    # field push is the one that is wrong.
    v_ode4, _s4 = ode_push(vel, v_a, args.ode_steps * 4)
    pushes[f'ode_x4'] = v_ode4
    rep['ode_refine_drift_vox'] = float((v_ode4 - v_ode).norm(dim=1).max()) * (n / 2.0)
    rep['ode_vs_field_max_vox'] = float((v_ode - pushes[f'field_s{base_steps}'])
                                        .norm(dim=1).max()) * (n / 2.0)

    # (0) before attributing anything to the model: what does a fold-free warp of
    # the same size score on this mesh, exactly and through the grid?
    amp = args.ctrl_amp or float(flow_inv.pow(2).sum(1).sqrt().mean())
    rep['control'] = control_flips(v_a, v_a_np, faces, n, amp, args.ctrl_cycles,
                                   args.ode_steps, model.stn, args.sq_sweep, base_steps)

    # (6) the same sweep on the MODEL's own field. The control has closed-form
    # truth but is smooth by construction; this one is rough (min det -3.9),
    # which is where resampling error actually bites. The reference here is the
    # ODE, which the control certifies to 0.01 vox and 4x refinement to 0.006.
    sq = {}
    for s in args.sq_sweep:
        vf = field_push(integrate_svf(-vel, model.stn, s), v_a)
        d = (vf - v_ode).norm(dim=1) * (n / 2.0)
        fl, _ = flip_mask(v_a_np, vf.cpu().numpy(), faces)
        sq[str(s)] = {'max_vox': float(d.max()), 'mean_vox': float(d.mean()),
                      'flip_pct': float(fl.mean() * 100)}
    rep['squaring_sweep'] = sq

    # (5) self-intersection only for the two pushes that could actually ship;
    # the extra-steps and 4x-refined variants exist to test convergence, not to
    # be delivered. The un-pushed template is the baseline, since FreeSurfer
    # surfaces are not guaranteed intersection-free and only the increase is the
    # warp's; main() scores it once, because the only thing that differs between
    # subjects is the affine, which is linear and so can neither create nor
    # remove a crossing.
    # Which pushes get the (5) pass. The field push is the transform actually
    # applied, so it is the one that must be reported; ode is a diagnostic and is
    # also the rougher mesh, hence the more expensive one to test.
    want = {x.strip() for x in args.self_int_pushes.split(',')}
    ship = set() if args.no_self_int else (
        ({f'field_s{base_steps}'} if 'field' in want else set())
        | ({'ode'} if 'ode' in want else set()))
    tpl_stats, tpl_hit = ctx.get('template_si', (None, None))
    per_push, self_int = {}, {'template': tpl_stats} if ship else {}
    for name, pv in pushes.items():
        pv_np = pv.cpu().numpy()
        # flip is measured from the AFFINE-ALIGNED verts: the affine is a global
        # linear map and cannot fold, so including it only adds noise here.
        per_push[name], _flip = analyse_flips(v_a_np, pv_np, faces,
                                              det_inv, neg_pts, n, device)
        if name in ship:
            self_int[name], hit = self_intersections(pv_np, faces, name)
            if args.ply:
                write_selfint_ply(out / f'{subject}_{name}_selfint.ply', pv_np, faces,
                                  selfint_colors(faces, hit, tpl_hit, len(pv_np)), ctx['ref'])
    rep['per_push'] = per_push
    rep['self_int'] = self_int
    rep['flip_pct'] = per_push[f'field_s{base_steps}']['flip_pct']

    # ---- damper sweep: does the proposed fix hold Dice while killing flips? ----
    damp = []
    if args.damp:
        det_fw = det_norm(flow_fw)
        for g in args.global_scale:
            d = score_flow(ctx, integrate_svf(vel * g, model.stn, base_steps), affine, sample_seg)
            inv_g = integrate_svf(-vel * g, model.stn, base_steps)
            fl, _ = flip_mask(v_a_np, field_push(inv_g, v_a).cpu().numpy(), faces)
            damp.append({'kind': f'global x{g}', **d, 'flip_pct': float(fl.mean() * 100)})
        for th in args.damp_thresh:
            for sg in args.damp_sigma:
                vel_d, s = damp_velocity(vel, det_fw, th, sg, args.damp_floor)
                d = score_flow(ctx, integrate_svf(vel_d, model.stn, base_steps), affine, sample_seg)
                inv_d = integrate_svf(-vel_d, model.stn, base_steps)
                fl, _ = flip_mask(v_a_np, field_push(inv_d, v_a).cpu().numpy(), faces)
                damp.append({'kind': f'local th={th} sig={sg}', **d,
                             'flip_pct': float(fl.mean() * 100),
                             'touched_pct': float((s < 0.99).float().mean() * 100),
                             'min_scale': float(s.min())})
    rep['damper'] = damp
    return rep


def print_subject(name, r, steps):
    p = r['per_push']
    h = p[f'field_s{steps[0]}']
    c = r['control']
    print(f"\n[{name}]  WM Dice {r['dice_wm']:.4f} | parity {r['svf_parity_max']:.2e}")
    print(f"  (0) fold-free control      identity {c['identity_flip_pct']:.4f}% | analytic "
          f"{c['analytic_flip_pct']:.4f}% | grid {c['grid_flip_pct']:.4f}% | ode "
          f"{c['ode_flip_pct']:.4f}%  "
          f"(amp {c['amp']:.3f}, {c['cycles']:.2f} cycles, jac margin {c['jac_margin']:.2f})")
    print(f"      err vs exact push      ode/analytic {c['err_ode_analytic_vox']:.4f} vox | "
          f"ode/grid {c['err_ode_grid_vox']:.4f} | "
          f"field@s{c['err_field_base_steps']} {c['err_field_vox']:.4f}")
    steps_k = list(c['squaring_sweep'])

    def sweep_row(label, sweep, key, prec, mark=True):
        vals = " ".join(f"{sweep[s][key]:9.{prec}f}" for s in steps_k)
        best = f"  <- min at s={min(steps_k, key=lambda s: sweep[s][key])}" if mark else ''
        print(f"      {label:<22s} {vals}{best}")

    print("  (6) squaring steps         " + " ".join(f"{'s=' + s:>9s}" for s in steps_k))
    sweep_row('control vs exact mean', c['squaring_sweep'], 'mean_vox', 5)
    sweep_row('model vs ode mean', r['squaring_sweep'], 'mean_vox', 4)
    sweep_row('model vs ode max', r['squaring_sweep'], 'max_vox', 4)
    sweep_row('model flip%', r['squaring_sweep'], 'flip_pct', 4, mark=False)
    print("  (2) sub-voxel folding      " + "  ".join(
        f"{k}={r[k]:.4f}%" for k in r if k.startswith('fold_')))
    print(f"      min det inv           " + "  ".join(
        f"{k.split('@')[1]}={r[k]:.3f}" for k in r if k.startswith('min_det_inv@')))
    print(f"  (1) negative-det set       {r['inv_neg_voxels']:,} voxels "
          f"({r['inv_pct']:.4f}%) in {r['inv_neg_components']} blobs, min det {r['inv_min_det']:.3f}")
    print(f"  (3) push agreement         ode-vs-field {r['ode_vs_field_max_vox']:.2f} vox | "
          f"ode 4x-refine drift {r['ode_refine_drift_vox']:.3f} vox | "
          f"ode step {r['ode_max_step_vox']:.3f} vox")
    print(f"      flip by push           " + "  ".join(
        f"{k}={v['flip_pct']:.4f}%" for k, v in p.items()))
    print(f"  (4) metric                 flip {h['flip_pct']:.4f}% | non-sliver "
          f"{h['flip_pct_nonsliver']:.4f}% | slivers are {h['sliver_share_of_flips']:.1f}% of flips "
          f"({h['sliver_pct_of_mesh']:.1f}% of mesh)")
    print(f"      rotation angle         median {h['flip_angle_median_deg']:.1f}deg | "
          f"p10 {h['flip_angle_p10_deg']:.1f}deg | {h['flip_under_100deg_pct']:.1f}% under 100deg")
    print(f"  (1) co-location            {h['capture_within_2vox_pct']:.1f}% of flips within 2 vox "
          f"of a folded voxel (base rate {h['base_rate_within_2vox_pct']:.1f}%), "
          f"median dist {h['dist_to_neg_median_vox']:.1f} vox")
    print(f"      det under flips        median {h['det_at_flip_median']:.3f} | p5 "
          f"{h['det_at_flip_p5']:.3f} | {h['flip_with_positive_det_pct']:.1f}% sit at det>0 "
          f"(mesh-wide median {h['det_at_all_median']:.3f})")
    print(f"      flip geometry          {h['flip_clusters']} clusters, largest "
          f"{h['flip_largest_cluster']} tris | area ratio {h['area_ratio_median']:.3f}")
    for k, s in r['self_int'].items():
        print(f"  (5) self-intersection      {k:<12s} {s['si_faces_pct']:.4f}% of faces "
              f"({s['si_faces']:,} tris, {s['si_pairs']:,} pairs) in {s['si_clusters']} patches, "
              f"largest {s['si_largest']} tris")
    if r['damper']:
        print(f"      {'damper':<22s} {'WM Dice':>8s} {'fold%':>8s} {'min det':>8s} "
              f"{'flip%':>8s} {'touched%':>9s}")
        print(f"      {'(none)':<22s} {r['dice_wm']:8.4f} {r['fold_pct']:8.4f} "
              f"{r['min_det']:8.3f} {h['flip_pct']:8.4f} {'-':>9s}")
        for d in r['damper']:
            print(f"      {d['kind']:<22s} {d['dice_wm']:8.4f} {d['fold_pct']:8.4f} "
                  f"{d['min_det']:8.3f} {d['flip_pct']:8.4f} "
                  f"{d.get('touched_pct', float('nan')):9.3f}")


def main():
    ap = argparse.ArgumentParser(description='Diagnose the mesh-flip / voxel-folding gap')
    ap.add_argument('--model', required=True, help='run dir, best_model.pth, or run name')
    ap.add_argument('--config', default=None)
    ap.add_argument('--output_dir', default=None, help='default <run_dir>/flip_probe')
    ap.add_argument('--val_txt', default=None)
    ap.add_argument('--num_samples', type=int, default=3)
    ap.add_argument('--sample_idxs', default=None, help='comma list of val indices')
    ap.add_argument('--template_surf', nargs=2, metavar=('LH', 'RH'), default=None)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--affine', choices=['auto', 'on', 'off'], default='auto')
    ap.add_argument('--refine', default='1,2',
                    help='sub-voxel folding lattices; 4 needs ~10 GB VRAM')
    ap.add_argument('--steps', default='7,10',
                    help='squaring steps; the first is the model\'s own (7)')
    ap.add_argument('--ode_steps', type=int, default=32)
    ap.add_argument('--sq_sweep', default='3,4,5,6,7,8,10,12',
                    help='squaring counts to sweep for (6); s=7 is the model\'s own')
    ap.add_argument('--ctrl_amp', type=float, default=0.0,
                    help='control-warp amplitude in normalized units (0 = match the '
                         'model\'s own mean displacement)')
    ap.add_argument('--ctrl_cycles', type=float, default=2.0,
                    help='control-warp spatial frequency; capped so the Jacobian stays positive')
    ap.add_argument('--ply', action='store_true',
                    help='write a colour-coded PLY per push (~13 MB each) for a 3D viewer')
    ap.add_argument('--no_self_int', action='store_true',
                    help='skip (5) self-intersection; it is the slow part on a 655k-face mesh')
    ap.add_argument('--self_int_pushes', default='field,ode',
                    help="which pushes get (5): comma list of 'field' and 'ode'. The "
                         "field push is the applied transform and the one to report; "
                         "'field' alone halves this pass")
    ap.add_argument('--damp', action='store_true', help='measure candidate dampers')
    ap.add_argument('--damp_thresh', default='0.05,0.2')
    ap.add_argument('--damp_sigma', default='1,2')
    ap.add_argument('--damp_floor', type=float, default=0.0)
    ap.add_argument('--global_scale', default='0.95')
    args = ap.parse_args()
    args.refine = [int(x) for x in args.refine.split(',')]
    args.steps = [int(x) for x in args.steps.split(',')]
    args.sq_sweep = [int(x) for x in args.sq_sweep.split(',')]
    args.damp_thresh = [float(x) for x in args.damp_thresh.split(',')]
    args.damp_sigma = [float(x) for x in args.damp_sigma.split(',')]
    args.global_scale = [float(x) for x in args.global_scale.split(',')]

    cfg = load_config(args.config)
    ckpt = resolve_checkpoint(args.model, cfg)
    use_affine = detect_affine(ckpt) if args.affine == 'auto' else args.affine == 'on'
    out = Path(args.output_dir).expanduser() if args.output_dir \
        else Path(ckpt).parent.parent / 'flip_probe'
    out.mkdir(parents=True, exist_ok=True)
    write_git_sha(out)
    print(f"[probe] checkpoint = {ckpt}\n[probe] affine = {use_affine} | device = {args.device} "
          f"| out = {out}", flush=True)

    ctx = setup_inference(ckpt, args.config, args.device, use_affine=use_affine, verbose=True)
    ctx['ref'] = nib.load(ref_for(cfg['data']['template_seg_path'])) if args.ply else None
    verts_np, faces, n_lh = load_template_mesh(cfg, args.template_surf)
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=ctx['device'])
    print(f"[probe] mesh {len(verts_np):,} verts | {len(faces):,} faces", flush=True)

    # The template mesh belongs to the TEMPLATE subject, not to whichever subject
    # is being probed, and its self-intersections are the same for all of them —
    # so score it once, and name its PLY after the subject it actually came from.
    tpl_name = Path(cfg['data']['template_seg_path']).parent.name or 'template'
    if not args.no_self_int:
        ctx['template_si'] = self_intersections(verts_np, faces, f'template {tpl_name}')
        if args.ply:
            hit = ctx['template_si'][1]
            write_selfint_ply(out / f'template_{tpl_name}_selfint.ply', verts_np, faces,
                              selfint_colors(faces, hit, hit, len(verts_np)), ctx['ref'])

    d = cfg['data']
    ds = SegDataset(args.val_txt or d['val_txt'], d['template_seg_path'],
                    target_size=ctx['target_size'], seg_filename=d['seg_filename'],
                    preload=False)
    idxs = [int(x) for x in args.sample_idxs.split(',')] if args.sample_idxs else \
        sorted(set(np.linspace(0, len(ds) - 1,
                               min(args.num_samples, len(ds))).astype(int).tolist()))

    rows = []
    for k, i in enumerate(idxs, 1):
        name = Path(ds.subject_dirs[i]).name
        print(f"\n[probe] subject {k}/{len(idxs)}: {name}", flush=True)
        sample_seg = ds[i]['sample_seg'].unsqueeze(0).to(ctx['device'])
        r = probe_subject(ctx, verts_t, verts_np, faces, sample_seg, args, out, name)
        r.update({'subject': name, 'idx': int(i)})
        print_subject(name, r, args.steps)
        rows.append(r)

    (out / 'flip_probe.json').write_text(json.dumps(
        {'checkpoint': ckpt, 'affine': use_affine, 'sliver_q': SLIVER_Q,
         'n_verts': len(verts_np), 'n_faces': len(faces), 'per_sample': rows}, indent=2))
    print(f"\n[probe] n={len(rows)} | json -> {out / 'flip_probe.json'}")
    print("[probe] verdict guide: fold% climbing with the lattice => sub-voxel (2); "
          "flip falling with steps or ode disagreeing => numerics (3); flips on slivers or "
          "near 90deg => metric (4); high capture near folded voxels + det<0 under them "
          "=> few-bad-voxels (1), the only case a local damper fixes. Self-intersection "
          "(5) is the only number that is a topology failure; flip % is not, and the "
          "push whose control error vs exact is smaller is the one to believe.")


if __name__ == '__main__':
    main()
