"""
Surface-distance per-subject refinement: does an objective that measures the
DELIVERABLE (mm distance from the pushed template white-surface vertices to the
subject's white surface) remove the smooth local mismatches the Dice/CE
refinement cannot see?

Same optimization variable as the deployed tool
(bandlimit_opt/instance_opt_bandlimited.py: 50 Adam steps on a coarse tied
delta against the input seg's Dice/CE) — a single tied coarse delta on a d^3
grid, trilinearly upsampled onto the FROZEN net velocity,

    v = lp(v_net) + upsample(delta),  flow_fw = exp(+v),  flow_rv = exp(-v)

with the affine and lambda maps frozen — and one arm per objective:

  dice    the deployed objective, unchanged (control; imports the very
          functions instance_opt_bandlimited.py optimizes).
  dsdf    the deployed Dice/CE data term AND mean |S_in(x_i)| + w_tail * mean
          of the top --tail_frac |S_in(x_i)| over CORTICAL template vertices
          x_i = the pushed mesh (affine undone, then + flow_rv sampled
          trilinear — train._surface_points), where S_in is the signed distance
          to the boundary of the net's OWN INPUT (SynthSeg WM mask).
          Deployment-legal: nothing here sees the GT.
  sym     dsdf + the REVERSE surface term. The one-sided term is blind to
          subject folds the pushed template never reaches, so the subject's
          own WM<->cortex interface (marching cubes on the smoothed SynthSeg
          WM channel, kept where the smoothed cortex channel > CORTEX_MIN) is
          carried subject->template through flow_fw and the affine —
          train._surface_points' s2t — and scored against the TEMPLATE white
          SDF. The template is fixed, so still deployment-legal.
  symM    sym + mesh regularisers on the full pushed template mesh, each an
          EXCESS over the affine-aligned template's own geometry
          (losses.mesh_regularizers): mean (log |e'|/|e|)^2 over edges, mean
          relu(|d'_i| - |d_i|) of the uniform Laplacian displacement, mean
          relu((1 - n'_a.n'_b) - (1 - n_a.n_b)) over adjacent face pairs. The
          template's curvature is free; only added roughness costs.

`symM` at the default knobs IS the arm tuned on the trio sweep (Dice/CE x2,
surface weights annealed over 25 steps, tails over the top 10 %, every
cortex-filtered interface point, mesh weights x1) and is the one to run; the
other three are its ablations.

The surface arms keep the training `bending` and `jacobian` (anti-fold) terms
at their config weights and DROP `displacement` and `lambda_smoothness`: those
penalise moving at all, which is exactly the sliding a mis-matched gyrus needs.
Folding %, self-intersection (utils/push_optimized_mesh.py) and WM Dice are
therefore reported for every arm, so the trade is visible rather than assumed.

The surface weight is CALIBRATED per subject: w_surface is set so that at
iteration 0 w_surface * mean|S| equals the dice arm's data term (w_dice * dice
+ w_ce * ce) at iteration 0, and w_tail = --tail_ratio * w_surface. The reverse
term is calibrated the same way against its own iteration-0 mean; each mesh
term to MESH_FRAC x the data term over its own iteration-0 value (floored at
MESH_EPS), times --mesh_w. The numbers go into metrics.json.

One optimization per arm runs to max(--snapshots) and stores flows + metrics at
each snapshot, so `<arm><iters>` (symM50, symM100, ...) are two reads of one
trajectory rather than two runs.

Two optional knobs, each defaulting to the behaviour above:
  --medial       `harmonic`: a term on the MEDIAL-WALL template vertices (the
                 ones the surface terms exclude) of the pushed mesh, in the
                 surface arms: mean |L u| mm (uniform Laplacian) of the
                 displacement u = pushed - aligned template, so the wall follows
                 its cortical rim. Calibrated like the mesh terms.
  --affine_init  `auto` fits a closed-form similarity of the template class-3
                 mask to the input class-3 mask (centroids + one isotropic
                 scale from the second moments, no rotation) and keeps whichever
                 of it and the net's affine gives the higher Dice3; the UNet,
                 refinement, saved flows and push all use the pick.

Numbers only. Outputs into --output_dir:
    metrics.json                   per-subject x arm x snapshot, arm means,
                                   calibration, weights, wall times
    arms.csv                       one row per subject x arm
    loss_curves/<subj>_<arm>.json  per-term optimization curves
    flows/<subj>_<arm>.npz         raw UNet-scale vel_fw/vel_rv (+ affine), the
                                   layout utils/push_optimized_mesh.py reads
    GIT_SHA.txt                    provenance

Run from the S-RegNET directory (needs torch + scipy + nibabel; GPU cluster):
    python bandlimit_opt/instance_opt_surface.py \\
        --checkpoint <run_dir_or_best_model.pth> --config config_meshsup.yaml \\
        --subjects OASIS_OAS1_0115_MR1,OASIS_OAS1_0146_MR1 \\
        --arms dice@50,symM@50 --coarse_levels 96 --lowpass_init 144 \\
        --output_dir <out> --device cuda:0
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from skimage.measure import marching_cubes

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))          # repo-root modules, whatever the cwd is
sys.path.insert(0, str(_ROOT / 'utils'))

from git_provenance import write_git_sha
from inference import load_config, setup_inference
from losses import (MESH_TERMS, SegRegistrationLoss, compute_dice_score,
                    mesh_geometry, mesh_regularizers, mesh_structure, sample_sdf)
from visualize_run import resolve_checkpoint, detect_affine
from visualize_mesh import (load_template_mesh, mm_per_norm, ref_for,
                            sample_field, undo_affine)
from train import _apply_affine_pts, _warp_template, _warp_sample_inverse
from push_optimized_mesh import load_cortex_mask
from instance_opt_bandlimited import (LOG_EVERY, WM, arm_metrics, build_velocities,
                                      edge_band, integrate, load_onehot,
                                      loss_and_terms, lowpass, net_init,
                                      save_flows)

# arm -> what it adds to the deployed Dice/CE data term. The knobs are CLI
# flags whose defaults are the arm tuned on the trio sweep: Dice/CE x2,
# surface weights annealed over 25 steps, tails over the top 10 %, every
# cortex-filtered interface point, mesh weights x1.
ARMS = {'dice': {},                                         # deployed control
        'dsdf': {'surf': True},                             # + template->subject SDF
        'sym':  {'surf': True, 'rev': True},                # + subject->template SDF
        'symM': {'surf': True, 'rev': True, 'mesh': True}}  # + mesh regularisers
DICE_TERMS = ('dice', 'cross_entropy', 'bending', 'jacobian', 'displacement',
              'lambda_smoothness')
DSDF_TERMS = ('dice', 'cross_entropy', 'surface', 'surface_tail', 'bending',
              'jacobian')
REV_TERMS = ('surface_rev', 'surface_rev_tail')
SURF_TERMS = ('surface', 'surface_tail') + REV_TERMS   # the weights --anneal_steps ramps
TAIL_FRAC = 0.05                # losses.surface_distance_loss's top-5 % tail; the
                                # calibration print's reference, NOT the arms' --tail_frac
CLAMP_MM = 10.0                 # utils/make_white_sdf.py's clamp, matched here
SDF_SIGMA_VOX = 0.75            # Gaussian smoothing of the mask-derived SDF
CORTEX = 1                      # cortex channel in the 5-class one-hot
CORTEX_MIN = 0.2                # smoothed-cortex value marking the WM<->cortex interface
N_METRIC_PTS = 60_000           # interface points the reported rev_mean uses, so the
                                # column stays comparable across --n_rev_pts
MESH_FRAC = 0.2                 # each mesh term's share of the data term at iteration 0
MESH_EPS = 1e-3                 # floor on an iteration-0 mesh term in that calibration
ROW_KEYS = ('seconds', 'wm_dice', 'edge_dice', 'folding_pct', 'worst_det',
            'msd_vox', 'hd95_vox', 'inverse_residual_vox',
            'dgt_mean', 'dgt_p95', 'din_mean', 'din_p95', 'rev_mean', 'lap_mm')


# =============================================================================
# SDF from the input mask
# =============================================================================

def mask_sdf(mask, sigma=SDF_SIGMA_VOX, clamp=CLAMP_MM):
    """Signed distance (mm) to the boundary of a boolean voxel mask on a 1 mm
    grid, negative inside — utils/make_white_sdf.py's sign convention.

    edt(~mask) - edt(mask) puts the zero crossing of the trilinear interpolant
    exactly on the face between an inside and an outside voxel, so no half-voxel
    offset is needed. The EDT of a binary mask is a voxel staircase, though: the
    gradient of |S| is piecewise constant and points along lattice directions,
    which a vertex can slide along without ever reaching the surface. A light
    Gaussian (sigma in voxels) makes the zero level set smooth without moving
    it — the field is odd across the interface, so smoothing is symmetric there.
    """
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    s = gaussian_filter((outside - inside).astype(np.float32), sigma)
    return np.clip(s, -clamp, clamp)


def input_surface_points(onehot, sigma=SDF_SIGMA_VOX):
    """Normalized (x, y, z) coords of the subject's WM<->cortex interface, from
    the net's own input: marching cubes at 0.5 on the Gaussian-smoothed WM
    channel, kept where the smoothed cortex channel exceeds CORTEX_MIN (the
    white surface has no counterpart at the WM<->subcortical-GM or
    WM<->ventricle interfaces). Voxel (i, j, k) -> normalized is
    visualize_mesh.world_to_norm's align_corners=False map.
    Returns (points, n_marching_cubes)."""
    wm = gaussian_filter(onehot[WM].astype(np.float32), sigma)
    cx = gaussian_filter(onehot[CORTEX].astype(np.float32), sigma)
    v = marching_cubes(wm, 0.5)[0]
    n_all = len(v)
    v = v[map_coordinates(cx, v.T, order=1) > CORTEX_MIN]
    norm = (2.0 * v + 1.0) / np.array(wm.shape, dtype=np.float32) - 1.0
    return np.ascontiguousarray(norm[:, ::-1], dtype=np.float32), n_all


def subsample(pts, n, seed=0):
    """<= n random rows of pts (all of them when n is None)."""
    if n is None or len(pts) <= n:
        return pts
    return pts[np.random.default_rng(seed).choice(len(pts), n, replace=False)]


def surface_terms(sdf, pts, frac=TAIL_FRAC):
    """losses.surface_distance_loss — (mean, mean of the top `frac`) unsigned
    mm from pts to the surface of `sdf` — with the tail fraction as a knob."""
    d = sample_sdf(sdf.float(), pts.float()).abs()
    return d.mean(), torch.topk(d, max(1, math.ceil(frac * d.numel()))).values.mean()


# =============================================================================
# Objectives
# =============================================================================

def reverse_distance(flow_fw, s, pts, frac=TAIL_FRAC):
    """(mean, tail) mm from the subject's input WM<->cortex interface points
    `pts`, carried subject->template (x + u_fw(x), then the affine — exactly
    train._surface_points' s2t), to the template white surface."""
    q = pts + sample_field(flow_fw, pts)
    return surface_terms(s['sdf_tpl'], _apply_affine_pts(q, s['affine']), frac)


def laplacian_mm(u, m):
    """|L u|_i in mm: the uniform mesh Laplacian losses.mesh_geometry applies to
    the vertex positions, applied to a per-vertex field u (N, 3) instead, on
    the same precomputed edges and degrees."""
    e0, e1 = m['edges'][:, 0], m['edges'][:, 1]
    nb = torch.zeros_like(u).index_add_(0, e0, u[e1]).index_add_(0, e1, u[e0])
    return (nb / m['deg'][:, None] - u).norm(dim=1) * m['mm']


def medial_term(pushed, s):
    """--medial harmonic: mean |L u| mm over the MEDIAL-WALL template vertices
    (the ones the surface terms exclude) of the pushed mesh's displacement
    u = pushed - verts_all (the displacement varies smoothly across the wall,
    so the wall follows its cortical rim)."""
    u = pushed - s['verts_all']
    return laplacian_mm(u, s['mesh_ops'])[~s['cortex_t']].mean()


def arm_terms(spec, medial=False):
    """The loss keys an arm sums and logs; `medial` adds the --medial wall term
    to the surface arms."""
    if not spec:
        return DICE_TERMS
    return (DSDF_TERMS + (REV_TERMS if spec.get('rev') else ())
            + (MESH_TERMS if spec.get('mesh') else ())
            + (('medial',) if medial else ()))


def dicesdf_objective(vel_fw, vel_rv, s, w, keys):
    """The deployed Dice/CE data term AND the surface terms, on one pair of
    integrations.

    The Dice/CE/bending/jacobian values come out of the same SegRegistrationLoss
    call instance_opt_bandlimited.loss_and_terms makes (its `total` is discarded
    and the terms re-summed with this arm's weights, so `displacement` and
    `lambda_smoothness` drop out). The push is byte-for-byte
    train._surface_points' template->subject direction (the deliverable): the
    affine is undone on the template vertices once (frozen), then the inverse
    dense flow is sampled at them. Normalized coordinates throughout, so the
    SDF's native 256^3 grid and the model's target_size grid need no rescaling.
    `keys` (arm_terms) switches on the reverse surface term and the mesh
    regularisers, which need flow_fw and the full pushed mesh respectively.
    """
    flow_fw = integrate(vel_fw, s['stn'])
    flow_rv = integrate(vel_rv, s['stn'])
    warped_fw = _warp_template(s['template_seg'], flow_fw, s['affine'], s['stn'])
    warped_rv = _warp_sample_inverse(s['fit_seg'], flow_rv, s['affine'], s['stn'])
    _, d = s['loss_fn'](warped_fw, s['fit_seg'], warped_rv, s['template_seg'],
                        flow_fw, flow_rv, s['lambda_map'], affine_matrix=None,
                        return_components=True)
    pushed = s['verts_all'] + sample_field(flow_rv, s['verts_all'])
    mean_d, tail_d = surface_terms(s['sdf_fit'], pushed[s['cortex_t']], s['tail_frac'])
    terms = {'dice': d['dice'], 'cross_entropy': d['cross_entropy'],
             'bending': d['bending'], 'jacobian': d['jacobian'],
             'surface': mean_d, 'surface_tail': tail_d}
    if 'surface_rev' in keys:
        terms['surface_rev'], terms['surface_rev_tail'] = reverse_distance(
            flow_fw, s, s['pts_rev'], s['tail_frac'])
    if 'mesh_edge' in keys:
        terms.update(mesh_regularizers(pushed, s['verts_all'], s['mesh_ops']))
    if 'medial' in keys:
        terms['medial'] = medial_term(pushed, s)
    return sum(w[k] * terms[k] for k in keys), terms


def objective(spec, vel_fw, vel_rv, s, keys):
    """(total, terms) for one arm. `dice` is the deployed objective,
    imported unchanged from instance_opt_bandlimited."""
    if not spec:
        return loss_and_terms(vel_fw, vel_rv, s)
    return dicesdf_objective(vel_fw, vel_rv, s, s['w_sdf'], keys)


# =============================================================================
# Per-arm evaluation
# =============================================================================

@torch.no_grad()
def pushed_stats(vel_fw, vel_rv, s):
    """Push ALL template vertices and measure the pushed mesh against both
    SDFs: mean / p95 |distance| over cortex. Plus the reverse direction (mean
    mm from the pushed input interface points to the template white surface,
    always on the fixed N_METRIC_PTS set) and the pushed mesh's mean
    |Laplacian| in mm."""
    flow_rv = integrate(vel_rv, s['stn'])
    pts = s['verts_all'] + sample_field(flow_rv, s['verts_all'])
    out = {'rev_mean': float(reverse_distance(integrate(vel_fw, s['stn']), s,
                                              s['pts_metric'])[0]),
           'lap_mm': float(mesh_geometry(pts, s['mesh_ops'])[1].mean())}
    for tag in ('gt', 'in'):
        d = sample_sdf(s[f'sdf_{tag}'], pts).abs().cpu().numpy().astype(np.float64)
        c = d[s['cortex']]
        out[f'd{tag}_mean'] = float(c.mean())
        out[f'd{tag}_p95'] = float(np.percentile(c, 95))
    return out


def snapshot(name, arm, vel_fw, vel_rv, s, out, affine, seconds):
    """Full metric set for one (subject, arm) point, flows saved in the layout
    utils/push_optimized_mesh.py reads."""
    met, _ = arm_metrics(vel_fw, vel_rv, s)
    row = {**met, **pushed_stats(vel_fw, vel_rv, s), 'seconds': round(seconds, 1)}
    save_flows(out, name, arm, vel_fw, vel_rv, affine)
    print(f"[surf] {name} {arm:<12s} WM {row['wm_dice']:.4f} | fold "
          f"{row['folding_pct']:.4f}% | d_gt {row['dgt_mean']:.3f} mm (p95 "
          f"{row['dgt_p95']:.3f}) | d_in {row['din_mean']:.3f} mm | rev "
          f"{row['rev_mean']:.3f} mm | lap {row['lap_mm']:.3f} mm", flush=True)
    return row


def optimize(base, s, args, name, out, affine, snaps):
    """Adam on a zero-initialized tied coarse delta; the delta is the only leaf
    with gradients. Snapshots the trajectory at each of `snaps` iterations."""
    spec = ARMS[base]
    keys = arm_terms(spec, args.medial != 'none')
    d = args.coarse_levels
    delta = torch.zeros(1, 3 if s['tied'] else 6, d, d, d, device=s['dev'],
                        requires_grad=True)
    opt = torch.optim.Adam([delta], lr=args.lr)
    anneal, w_full = (args.anneal_steps if spec else 0), dict(s['w_sdf'])
    curve = {'iters': [], 'total': [], 'terms': {k: [] for k in keys}}
    rows, t0 = {}, time.time()

    for it in range(max(snaps)):
        if anneal:      # surface weights ramp 0 -> 1 over the first `anneal` steps
            r = min(1.0, (it + 1) / anneal)
            s['w_sdf'] = {k: v * r if k in SURF_TERMS else v for k, v in w_full.items()}
        opt.zero_grad()
        total, terms = objective(spec, *build_velocities(delta, s), s, keys)
        total.backward()
        opt.step()
        step = it + 1
        if step == 1 or step % LOG_EVERY == 0 or step in snaps:
            curve['iters'].append(step)
            curve['total'].append(float(total.detach()))
            for k in keys:
                curve['terms'][k].append(float(terms[k].detach()))
            print(f'[surf] {name} {base} iter {step}/{max(snaps)} total '
                  f'{float(total.detach()):.5f} | '
                  + ' | '.join(f'{k} {float(terms[k].detach()):.5f}' for k in keys),
                  flush=True)
        if step in snaps:
            vf, vr = build_velocities(delta.detach(), s)
            rows[f'{base}{step}'] = snapshot(name, f'{base}{step}', vf, vr, s, out,
                                             affine, time.time() - t0)
    (out / 'loss_curves' / f'{name}_{base}.json').write_text(json.dumps(curve))
    return rows


# =============================================================================
# Analytic affine (--affine_init auto)
# =============================================================================

def mask_moments(mask):
    """Centroid and covariance of a boolean (D, H, W) mask's voxel centres in
    F.affine_grid's normalized (x, y, z) coordinates (align_corners=False)."""
    idx = mask.nonzero().float()                            # (N, 3) as (i, j, k)
    size = torch.tensor(mask.shape, dtype=torch.float32, device=mask.device)
    p = ((2.0 * idx + 1.0) / size - 1.0).flip(1)            # (k, j, i) -> (x, y, z)
    return p.mean(0), torch.cov(p.T)


def similarity_affine(tpl_mask, in_mask):
    """(1, 3, 4) affine in the model's convention. F.affine_grid's theta is a
    PULL — aligned(x) = template(A·[x;1]) — so A carries an INPUT-space
    coordinate to template space (train._apply_affine_pts, undo_affine). The
    masks' centroids and overall size are matched without rotation: the
    template->input map is M = s I with s = (det C_in / det C_tpl)^(1/6) (a 3-D
    covariance determinant scales as s^6), t = c_in - M c_tpl, and A is its
    inverse."""
    c_t, C_t = mask_moments(tpl_mask)
    c_i, C_i = mask_moments(in_mask)
    M = (torch.det(C_i) / torch.det(C_t)) ** (1 / 6) * torch.eye(3, device=c_t.device)
    A = torch.linalg.inv(M)
    return torch.cat([A, (c_t - A @ c_i)[:, None]], dim=1)[None]


@torch.no_grad()
def affine_dice(template_seg, input_seg, affine):
    """Hard class-WM Dice of the affine-aligned template against the input:
    model.forward's nearest pull, before any dense flow."""
    grid = F.affine_grid(affine, template_seg.size(), align_corners=False)
    aligned = F.grid_sample(template_seg, grid, mode='nearest',
                            padding_mode='zeros', align_corners=False)
    return compute_dice_score(aligned, input_seg, template_seg.shape[1])[0][WM]


@torch.no_grad()
def auto_affine(name, template_seg, input_seg, affine_net, mm):
    """--affine_init auto: fit the similarity and keep whichever of it and the
    net's affine has the higher class-WM Dice of the aligned template against
    the input (printed with the identity's for scale)."""
    affine = similarity_affine(template_seg[0, WM] > 0.5, input_seg[0, WM] > 0.5)
    eye = torch.eye(3, 4, device=input_seg.device)[None]
    desc = lambda a: (f"det {float(torch.det(a[0, :, :3])):.4f} t_mm ("
                      + ' '.join(f'{x * mm:+.2f}' for x in a[0, :, 3].tolist()) + ')')
    dice = lambda a: float(affine_dice(template_seg, input_seg, a))
    d_net, d_fit = dice(affine_net), dice(affine)
    keep_net = d_net >= d_fit
    print(f'[surf] {name} affine: identity Dice3 {dice(eye):.4f} | net {desc(affine_net)} '
          f'Dice3 {d_net:.4f} | similarity {desc(affine)} Dice3 {d_fit:.4f} | auto picks '
          f"{'net' if keep_net else 'similarity'}", flush=True)
    return affine_net if keep_net else affine


# =============================================================================
# Per subject
# =============================================================================

def probe_subject(name, sdir, ctx, mesh, loss_fn, args, out):
    cfg, dev, ts, stn = ctx['cfg'], ctx['device'], ctx['target_size'], ctx['stn']
    d = cfg['data']
    input_seg = load_onehot(sdir / d['input_seg_filename'], ts).to(dev)
    gt_seg = load_onehot(sdir / d['seg_filename'], ts).to(dev)

    affine = None
    if args.affine_init == 'auto':
        with torch.no_grad():
            affine_net = ctx['model'].affine_net(ctx['template_seg'], input_seg)
        affine = auto_affine(name, ctx['template_seg'], input_seg, affine_net,
                             mesh['ops']['mm'])
    vel_fw0, vel_rv0, lambda_map, affine = net_init(ctx['model'], ctx['template_seg'],
                                                    input_seg, affine)
    opt_fw0, opt_rv0 = vel_fw0, vel_rv0
    if args.lowpass_init:
        opt_fw0 = lowpass(vel_fw0, args.lowpass_init, ts)
        opt_rv0 = lowpass(vel_rv0, args.lowpass_init, ts)

    # Both SDFs on their native 256^3 grids: the GT one precomputed against the
    # FreeSurfer white surface, the deployment-legal one built here from the
    # net's own input channel.
    sdf_gt = torch.tensor(np.load(sdir / d['white_sdf_filename']),
                          dtype=torch.float32, device=dev)[None, None]
    in_onehot = np.load(sdir / d['input_seg_filename'])
    sdf_in = torch.tensor(mask_sdf(in_onehot[WM] > 0.5), dtype=torch.float32,
                          device=dev)[None, None]
    # Subject points for the reverse term: the input's own WM<->cortex interface.
    pts_cx, n_mc = input_surface_points(in_onehot)
    print(f'[surf] {name} input interface: {n_mc:,} marching-cubes verts -> '
          f'{len(pts_cx):,} at cortex', flush=True)

    gt_wm = (gt_seg.argmax(1)[0] == WM).cpu().numpy()
    verts_all = undo_affine(mesh['verts_t'], affine)
    s = {'template_seg': ctx['template_seg'], 'gt_seg': gt_seg,
         'fit_seg': input_seg, 'lambda_map': lambda_map, 'affine': affine,
         'stn': stn, 'loss_fn': loss_fn, 'num_classes': ctx['num_classes'],
         'target_size': ts, 'dev': dev, 'tied': True,
         'vel_fw0': opt_fw0, 'vel_rv0': opt_rv0,
         'gt_wm': gt_wm, 'band': edge_band(gt_wm),
         'sdf_gt': sdf_gt, 'sdf_in': sdf_in, 'sdf_fit': sdf_in,
         'sdf_tpl': mesh['sdf_tpl'], 'tail_frac': args.tail_frac,
         'pts_metric': torch.tensor(subsample(pts_cx, N_METRIC_PTS), device=dev),
         'pts_rev': torch.tensor(subsample(pts_cx, args.n_rev_pts or None), device=dev),
         'verts_all': verts_all,
         'cortex': mesh['cortex'], 'cortex_t': mesh['cortex_t'],
         'mesh_ops': mesh['ops'], 'medial': args.medial}

    # Calibration: the dice arm's data term at iteration 0 sets the surface
    # weights (forward and reverse), so every objective starts with a data term
    # of the same magnitude; each mesh term gets MESH_FRAC of it.
    with torch.no_grad():
        vf, vr = build_velocities(torch.zeros(1, 3, 1, 1, 1, device=dev), s)
        _, t0 = loss_and_terms(vf, vr, s)
        data0 = float(loss_fn.weights['dice'] * t0['dice']
                      + loss_fn.weights['cross_entropy'] * t0['cross_entropy'])
        calib = {'data_term_0': data0, 'n_pts_rev': len(s['pts_rev']),
                 'n_pts_metric': len(s['pts_metric']),
                 'tpl_lap_mm': float(mesh_geometry(verts_all, mesh['ops'])[1].mean())}
        pushed = s['verts_all'] + sample_field(integrate(vr, stn), s['verts_all'])
        for tag in ('in', 'gt'):
            m, tl = surface_terms(s[f'sdf_{tag}'], pushed[s['cortex_t']])
            calib[f'{tag}_mean_0_mm'] = float(m)
            calib[f'{tag}_tail_0_mm'] = float(tl)
            calib[f'w_surface_{tag}'] = data0 / float(m)
        m, tl = reverse_distance(integrate(vf, stn), s, s['pts_metric'])
        calib['rev_mean_0_mm'], calib['rev_tail_0_mm'] = float(m), float(tl)
        calib['w_surface_rev'] = data0 / float(m)
        for k, v in mesh_regularizers(pushed, verts_all, mesh['ops']).items():
            calib[f'{k}_0'] = float(v)
            calib[f'w_{k}'] = MESH_FRAC * data0 / max(float(v), MESH_EPS)
        if args.medial != 'none':
            calib['medial_0'] = float(medial_term(pushed, s))
            calib['w_medial'] = MESH_FRAC * data0 / max(calib['medial_0'], MESH_EPS)
    print(f"[surf] {name} calibration: dice data term @0 = {data0:.4f} | "
          + ' | '.join(f"{t}: mean {calib[f'{t}_mean_0_mm']:.3f} mm, tail "
                       f"{calib[f'{t}_tail_0_mm']:.3f} mm -> w_surface "
                       f"{calib[f'w_surface_{t}']:.4f}" for t in ('in', 'gt', 'rev'))
          + f" | template lap {calib['tpl_lap_mm']:.4f} mm | "
          + ' | '.join(f"{k}@0 {calib[f'{k}_0']:.5f} -> w {calib[f'w_{k}']:.4g}"
                       for k in MESH_TERMS)
          + (f" | medial({args.medial})@0 {calib['medial_0']:.5f} mm -> w "
             f"{calib['w_medial']:.4g}" if 'w_medial' in calib else ''), flush=True)

    rows = {'baseline': snapshot(name, 'baseline', vel_fw0, vel_rv0, s, out,
                                 affine, 0.0)}
    for base, snaps in args.arms:
        spec = ARMS[base]
        w = loss_fn.weights
        ws = calib['w_surface_in'] if spec else 0.0
        wr, tr = calib['w_surface_rev'], args.tail_ratio
        s['w_sdf'] = {'surface': ws, 'surface_tail': tr * ws,
                      'surface_rev': wr, 'surface_rev_tail': tr * wr,
                      'bending': w['bending'], 'jacobian': w['jacobian'],
                      'dice': args.dice_mult * w['dice'],
                      'cross_entropy': args.dice_mult * w['cross_entropy'],
                      **{k: args.mesh_w * calib[f'w_{k}'] for k in MESH_TERMS},
                      **({'medial': calib['w_medial']} if 'w_medial' in calib else {})}
        print(f"[surf] {name} {base}: tail_frac {s['tail_frac']} | rev pts "
              f"{len(s['pts_rev']):,} | weights " +
              ' '.join(f'{k}={v:.4g}' for k, v in s['w_sdf'].items()), flush=True)
        rows.update(optimize(base, s, args, name, out, affine, snaps))

    del input_seg, gt_seg, sdf_gt, sdf_in, s
    return {'calibration': calib, 'arms': rows}


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Per-subject refinement with a symmetric SURFACE-DISTANCE '
                    'objective plus mesh regularisers (symM) vs the deployed '
                    'Dice/CE objective, on the same tied coarse delta')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', default='config_meshsup.yaml')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--subjects', default=None,
                    help='comma list of subject dir names from val_txt '
                         '(overrides --num_subjects)')
    ap.add_argument('--num_subjects', type=int, default=3,
                    help='subjects from the head of val_txt, template subject '
                         'and subjects without an input seg skipped — the same '
                         'selection instance_opt_bandlimited.py makes')
    ap.add_argument('--arms', default='dice,symM',
                    help=f'comma list from {sorted(ARMS)}; an arm may carry its '
                         f'own snapshot list as base@i1:i2 (e.g. symM@50:100), '
                         f'otherwise --snapshots applies')
    ap.add_argument('--snapshots', default='50',
                    help='default iterations at which flows + metrics are '
                         'stored; one optimization runs to the largest')
    ap.add_argument('--coarse_levels', type=int, default=96,
                    help='delta grid size d')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lowpass_init', type=int, default=0,
                    help='low-pass the net-init velocity through a d^3 grid '
                         'before the opt arms; the baseline arm stays raw')
    ap.add_argument('--tail_ratio', type=float, default=0.5,
                    help='w_surface_tail / w_surface (forward and reverse)')
    ap.add_argument('--dice_mult', type=float, default=2.0,
                    help='multiplier on the config Dice/CE weights in the surface arms')
    ap.add_argument('--anneal_steps', type=int, default=25,
                    help='ramp the surface weights 0 -> 1 over this many steps (0 = off)')
    ap.add_argument('--tail_frac', type=float, default=0.10,
                    help='top fraction the surface tails average')
    ap.add_argument('--n_rev_pts', type=int, default=0,
                    help='reverse-term point budget (0 = every cortex-filtered '
                         'interface point)')
    ap.add_argument('--mesh_w', type=float, default=1.0,
                    help='multiplier on the calibrated mesh-regulariser weights (symM)')
    ap.add_argument('--medial', choices=('none', 'harmonic'), default='none',
                    help='term on the medial-wall template vertices of the pushed '
                         'mesh (surface arms): mean |Laplacian of the displacement| '
                         'mm, calibrated like the mesh terms')
    ap.add_argument('--affine_init', choices=('net', 'auto'), default='net',
                    help="affine the refinement starts from: the net's own, or "
                         'auto = whichever of it and a closed-form similarity fit of '
                         'the template class-3 mask to the input class-3 mask has the '
                         'higher class-3 Dice')
    args = ap.parse_args()

    t_start = time.time()
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_file() and not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    cfg = load_config(str(cfg_path))
    d = cfg['data']
    ckpt = resolve_checkpoint(args.checkpoint, cfg)
    out = Path(args.output_dir).expanduser()
    if not out.is_absolute():
        out = _ROOT / out
    for sub in ('loss_curves', 'flows'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    write_git_sha(out)

    args.snapshots = sorted({int(x) for x in args.snapshots.split(',') if x})
    # base[@i1:i2] -> (base, snapshots); a bare base takes --snapshots.
    specs = []
    for item in (a for a in args.arms.split(',') if a):
        base, _, sn = item.partition('@')
        if base not in ARMS:
            raise SystemExit(f'[surf] unknown arm {base} (known: {sorted(ARMS)})')
        specs.append((base, sorted({int(x) for x in sn.split(':') if x})
                      if sn else args.snapshots))
    args.arms = specs

    print(f'[surf] checkpoint = {ckpt}\n[surf] config = {cfg_path}\n'
          f'[surf] arms = {args.arms} | delta {args.coarse_levels}^3 | '
          f'lowpass_init = {args.lowpass_init} | lr = {args.lr} | '
          f'tail_ratio = {args.tail_ratio} | medial = {args.medial} | '
          f'affine_init = {args.affine_init} | out = {out}', flush=True)

    ctx = setup_inference(ckpt, str(cfg_path), args.device,
                          use_affine=detect_affine(ckpt), verbose=True)
    if args.affine_init == 'auto' and not ctx['use_affine']:
        raise SystemExit('[surf] --affine_init auto needs an affine-enabled checkpoint')

    # Same loss object (and therefore the same weights) as the deployed tool;
    # lambda_prior is zeroed because the lambda map is frozen here.
    weights = dict(cfg['loss'])
    class_weights = weights.pop('class_weights', None)
    weights['lambda_prior'] = 0.0
    loss_fn = SegRegistrationLoss(weights=weights, class_weights=class_weights)

    verts_np, faces, _ = load_template_mesh(cfg, None)
    cortex = load_cortex_mask(d.get('cortex_labels_dir'))
    if cortex is None or len(cortex) != len(verts_np):
        raise SystemExit('[surf] cortex mask missing or mis-sized — the surface '
                         'objective must not score the medial wall')
    if not d.get('template_white_sdf_path'):
        raise SystemExit('[surf] data.template_white_sdf_path is not set — '
                         'use --config config_meshsup.yaml')
    # The template white SDF (fixed, so deployment-legal) scores the reverse
    # term; the mesh structure (with its mm scale) serves the regularisers.
    mesh = {'verts_t': torch.tensor(verts_np, dtype=torch.float32,
                                    device=ctx['device']),
            'cortex': cortex,
            'cortex_t': torch.tensor(cortex, device=ctx['device']),
            'ops': mesh_structure(faces, len(verts_np),
                                  mm_per_norm(nib.load(ref_for(d['template_seg_path']))),
                                  ctx['device']),
            'sdf_tpl': torch.tensor(np.load(d['template_white_sdf_path']),
                                    dtype=torch.float32,
                                    device=ctx['device'])[None, None]}
    print(f'[surf] template {len(verts_np):,} verts / {len(faces):,} faces | '
          f'cortex {int(cortex.sum()):,} | edges {len(mesh["ops"]["edges"]):,} | '
          f'face pairs {len(mesh["ops"]["pairs"]):,} | '
          f'{mesh["ops"]["mm"]:.1f} mm/norm', flush=True)

    # Subject selection mirrors instance_opt_bandlimited.py: the head of
    # val_txt, template subject excluded, an input seg required.
    scans = Path(d['template_seg_path']).parent.parent
    if args.subjects:
        subjects = [x for x in args.subjects.split(',') if x]
    else:
        template_subject = Path(d['template_seg_path']).parent.name
        subjects = [sd.name for sd in
                    (Path(x).parent for x in
                     Path(d['val_txt']).read_text().splitlines() if x.strip())
                    if sd.name != template_subject
                    and (sd / d['input_seg_filename']).is_file()][:args.num_subjects]
    print(f'[surf] subjects = {subjects}', flush=True)
    results = {}
    for i, name in enumerate(subjects):
        print(f'[surf] === subject {i + 1}/{len(subjects)}: {name} ===', flush=True)
        results[name] = probe_subject(name, scans / name, ctx, mesh, loss_fn,
                                      args, out)
        if 'cuda' in args.device:
            torch.cuda.empty_cache()

    arms = ['baseline'] + [f'{b}{n}' for b, sn in args.arms for n in sn]
    means = {a: {k: float(np.mean([results[s]['arms'][a][k] for s in results]))
                 for k in ROW_KEYS}
             for a in arms if all(a in results[s]['arms'] for s in results)}
    with open(out / 'arms.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['subject', 'arm', *ROW_KEYS])
        w.writeheader()
        for name in list(results) + ['MEAN']:
            src = results[name]['arms'] if name in results else means
            for a in arms:
                if a in src:
                    w.writerow({'subject': name, 'arm': a,
                                **{k: src[a][k] for k in ROW_KEYS}})
    (out / 'metrics.json').write_text(json.dumps(
        {'probe': {**vars(args), 'checkpoint': ckpt, 'config': str(cfg_path),
                   'loss_weights': {k: float(v) for k, v in weights.items()}},
         'arm_means': means, 'subjects': results,
         'wall_seconds': round(time.time() - t_start, 1)}, indent=2))

    for a in arms:
        if a in means:
            m = means[a]
            print(f"[surf] mean {a:<12s} WM {m['wm_dice']:.4f} | fold "
                  f"{m['folding_pct']:.4f}% | d_gt {m['dgt_mean']:.3f} mm | p95 "
                  f"{m['dgt_p95']:.3f} | d_in {m['din_mean']:.3f} mm | rev "
                  f"{m['rev_mean']:.3f} mm", flush=True)
    print(f"[surf] metrics -> {out / 'metrics.json'}\n"
          f"[surf] csv     -> {out / 'arms.csv'}\n"
          f"[surf] wall {time.time() - t_start:.1f} s")


if __name__ == '__main__':
    main()
