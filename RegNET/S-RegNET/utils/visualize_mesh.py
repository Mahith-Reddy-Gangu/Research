"""
Mesh deformation viz: push the genus-0 template WM mesh into each subject's
space through the trained field and overlay it against the subject's own
FreeSurfer white surface.

Ported from the invertible-deform-SRegNET branch (utils/mesh_slice_figure.py,
utils/visualize_mesh_samples.py, overlay_surface.py, wm_template.py) and adapted
to this branch's SVF model. The port's one real subtlety is the direction:

    STN consumes flow_fw as a PULL field — warped(x) = template(x + flow_fw(x)) —
    so flow_fw maps SUBJECT coords to TEMPLATE coords. Applying it to template
    vertices moves them the WRONG WAY; the push needs the inverse.

    flow_rv looks like that inverse but is NOT trustworthy in this checkpoint:
    train.py scores stn(sample_seg, flow_rv) against the RAW template while
    cycle_consistency_loss asks flow_rv to invert flow_fw in AFFINE-ALIGNED
    space. With affine enabled those two demands differ by A^-1, so flow_rv
    converges to a compromise that is neither. See push_mesh.

    Default push is therefore the numerical inverse of flow_fw (a damped fixed
    point) — it inverts the exact field the Dice was trained on. All three
    candidate pushes are always measured against the GT surface so the choice is
    a reported number, not an assumption (--push selects which one is drawn).

Affine (auto-detected from the checkpoint) is undone first: the UNet sees the
affine-warped template, so the chain is

    v_template --(A^-1)--> v_aligned --(+u_rv)--> v_subject

Everything happens in normalized [-1,1] (x,y,z) coords (align_corners=False,
x<->W, y<->H, z<->D); mm distances go through the subject's seg4.nii.gz affine,
so they are resolution-free rather than scaled by a 256/2 constant.

Outputs per subject (into --output_dir):
    <subj>_mesh.png            3 ortho slices (template / GT white / deformed) + distance histogram
    <subj>_deformed.white.surf deformed mesh, FreeSurfer surface-RAS (open in freeview)
    summary.json               per-subject mm distances, flip %, inverse agreement

Run from the S-RegNET directory (needs torch + nibabel):
    python visualize_mesh.py \\
        --model ~/shared_scratch/training_results/mahith_experiment/20260801_083414 \\
        --num_samples 5 --device cuda:4
    # template mesh: config data.template_wm_mesh_path (.npz from wm_template.py) if
    #   present, else auto-derived /meshes/<template subj>/{lh,rh}.white
    # --template_surf lh.white rh.white   explicit override
    # --push numeric | rv | rv_noaffine   which push to draw (all are measured)
    # --subjects 0287,0203                pick val subjects by name substring
    # --suffix _run2                      keep a rerun's outputs beside the old ones
"""

import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from git_provenance import write_git_sha
from get_data import SegDataset
from inference import load_config, setup_inference
from visualize_run import resolve_checkpoint, detect_affine, centroid_slices

WM_LABEL = 3                                        # 0 bg, 1 cortex, 2 subGM, 3 WM, 4 CSF
TEMPLATE_C, GT_C, DEFORM_C = 'gold', 'limegreen', 'deepskyblue'
AXIS_NAMES = ['axial (z)', 'coronal (y)', 'sagittal (x)']
SURF_NAMES = [('lh.white', 'rh.white'), ('lh.white.surf', 'rh.white.surf')]


# =============================================================================
# Coordinate helpers (ported from overlay_surface.py — the only thing that can
# silently go wrong)
# =============================================================================

def world_to_norm(world, ref):
    """World mm -> normalized [-1,1] (x,y,z) of ref's voxel grid. Axis map
    x<->k, y<->j, z<->i, matching one-hot (5,Ni,Nj,Nk)->(5,D,H,W) and
    grid_sample's (x,y,z)->(W,H,D). Uses ref's NATIVE dims: the [-1,1] FOV is
    resolution-independent, so a 128³ field samples at the same coordinate."""
    Ni, Nj, Nk = ref.shape[:3]
    ijk = apply_affine(np.linalg.inv(ref.affine), world)
    return np.stack([(2 * ijk[:, 2] + 1) / Nk - 1,
                     (2 * ijk[:, 1] + 1) / Nj - 1,
                     (2 * ijk[:, 0] + 1) / Ni - 1], axis=1)


def norm_to_world(norm, ref):
    """Inverse of world_to_norm."""
    Ni, Nj, Nk = ref.shape[:3]
    ijk = np.stack([((norm[:, 2] + 1) * Ni - 1) / 2,
                    ((norm[:, 1] + 1) * Nj - 1) / 2,
                    ((norm[:, 0] + 1) * Nk - 1) / 2], axis=1)
    return apply_affine(ref.affine, ijk)


def mm_per_norm(ref):
    """mm per unit of normalized coordinate for ref's grid — the [-1,1] FOV spans
    the whole volume, so 1 norm unit = (voxel size * dim) / 2 per axis. Averaged
    over axes; used for scalar residuals only (real distances go through the
    full affine)."""
    zooms = np.abs(np.linalg.norm(ref.affine[:3, :3], axis=0))
    return float(np.mean(zooms * np.array(ref.shape[:3]) / 2.0))


def norm_to_vox(verts, n):
    """Normalized (x,y,z) -> voxel (i,j,k) of the n³ model grid."""
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    return np.stack([((z + 1) * n - 1) / 2,
                     ((y + 1) * n - 1) / 2,
                     ((x + 1) * n - 1) / 2], axis=1)


def slice_segments(vox, faces, axis, s):
    """True mesh-plane intersection at {axis == s}: every triangle straddling the
    plane contributes the segment where it crosses, so the contour is continuous
    instead of a rasterized band of vertices. Returns (K,2,2) in-plane coords."""
    p, q = [ax for ax in range(3) if ax != axis]
    d = vox[:, axis] - s
    xy = vox[:, [p, q]]
    df, pf = d[faces], xy[faces]

    pts, masks = [], []
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        di, dj = df[:, i], df[:, j]
        masks.append((di > 0) != (dj > 0))
        denom = di - dj
        safe = np.abs(denom) > 1e-12
        t = np.where(safe, di / np.where(safe, denom, 1.0), 0.5)
        pts.append(pf[:, i, :] + t[:, None] * (pf[:, j, :] - pf[:, i, :]))

    P, Mk = np.stack(pts, axis=1), np.stack(masks, axis=1)
    sel = Mk.sum(1) == 2
    return P[sel][Mk[sel]].reshape(-1, 2, 2)


def triangle_flip_fraction(verts_before, verts_after, faces):
    """Fraction of faces whose orientation reversed — the surface analogue of a
    negative Jacobian, which voxel folding alone does NOT capture (a run with
    0.013% voxel folding had 10.49% cortical triangle flips)."""
    def normals(v):
        v0, v1, v2 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
        return np.cross(v1 - v0, v2 - v0)
    nb, na = normals(np.asarray(verts_before)), normals(np.asarray(verts_after))
    return float(((nb * na).sum(1) < 0).mean()) * 100.0


# =============================================================================
# Mesh push
# =============================================================================

def sample_field(field, pts):
    """Sample (1,3,D,H,W) field at pts (N,3 normalized x,y,z) -> (N,3).
    align_corners=False matches the SpatialTransformer; padding_mode='border'
    keeps vertices near the FOV edge from being handed a zero displacement."""
    g = pts.view(1, -1, 1, 1, 3)
    s = F.grid_sample(field, g, mode='bilinear', padding_mode='border',
                      align_corners=False)
    return s.squeeze(0).squeeze(-1).squeeze(-1).permute(1, 0)


def undo_affine(verts, affine_matrix):
    """Template coords -> affine-aligned coords. affine_grid is itself a PULL:
    aligned(x) = template(A·[x;1]), so a template point p sits at A^-1·p in the
    aligned space the dense field is defined on."""
    if affine_matrix is None:
        return verts
    A = torch.eye(4, dtype=verts.dtype, device=verts.device)
    A[:3, :] = affine_matrix[0]
    inv = torch.linalg.inv(A)
    h = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=1)
    return (h @ inv.T)[:, :3]


def integrate_svf(vel, stn, steps=7):
    """Scaling-and-squaring, byte-for-byte the loop in model.SegRegistrationNet.forward:
    phi = exp(vel) as a displacement field."""
    f = vel / (2 ** steps)
    for _ in range(steps):
        f = f + stn(f, f)
    return f


@torch.no_grad()
def svf_inverse_flow(model, template_seg, sample_seg, steps=7):
    """The EXACT inverse for an SVF model: phi = exp(v), so phi^-1 = exp(-v).
    No fixed point, no residual, no stalling — the thing the numerical inverse is
    only approximating.

    Recovers the velocity by re-running the model's own front half (affine head +
    UNet, with the same mode='nearest' affine warp forward() uses), then
    integrates -vel with the same loop. Also re-integrates +vel and returns the
    max deviation from the model's own flow_fw: if that parity number is not
    ~1e-6, this replication has drifted from model.py and the inverse must not be
    trusted. Returns (flow_inv, affine_matrix, parity_max_abs)."""
    affine_matrix, tmpl = None, template_seg
    if model.use_affine:
        affine_matrix = model.affine_net(template_seg, sample_seg)
        grid = F.affine_grid(affine_matrix, template_seg.size(), align_corners=False)
        tmpl = F.grid_sample(template_seg, grid, mode='nearest',
                             padding_mode='zeros', align_corners=False)
    vel, _lam = model.unet(torch.cat([tmpl, sample_seg], dim=1))
    vel_fw = vel[:, :3]

    flow_inv = integrate_svf(-vel_fw, model.stn, steps)
    flow_fw_model, _rv, _l, _a = model(template_seg, sample_seg)
    parity = float((integrate_svf(vel_fw, model.stn, steps) - flow_fw_model).abs().max())
    return flow_inv, affine_matrix, parity


def invert_numerically(flow_fw, v, n_iter=500, alpha=0.5):
    """Damped fixed point for o with o + u_fw(o) = v.

    NOTE the residual is not just a convergence stat — it is per-vertex NOISE on
    a mesh whose triangles are ~1 mm across, so a residual of a few tenths of a
    mm can reverse triangles on its own. Any flip % read off this push must be
    checked against a tighter solve (raise n_iter) before being blamed on the
    model. Returns (o, per-vertex residual in normalized units)."""
    o = v.clone()
    for _ in range(n_iter):
        o = o - alpha * (o + sample_field(flow_fw, o) - v)
    res = (o + sample_field(flow_fw, o) - v).norm(dim=1)
    return o, res


@torch.no_grad()
def push_mesh(ctx, verts_t, sample_seg, n_iter=500, alpha=0.5):
    """Push template mesh vertices into subject space THREE ways, because what
    flow_rv actually learned is ambiguous in this checkpoint:

      numeric      A^-1 then the numerical inverse of flow_fw. Principled: it
                   inverts the exact field the Dice was trained on, so it is
                   correct whatever flow_rv did. Trust this one.
      rv           A^-1 then + u_rv. Correct only if flow_rv is flow_fw's
                   inverse in AFFINE-ALIGNED space (what cycle_consistency_loss
                   asks for).
      rv_noaffine  + u_rv applied directly to raw template verts. Correct if
                   flow_rv absorbed the affine — which is what the reverse Dice
                   term actually supervises: train.py warps sample_seg with
                   flow_rv and scores it against the RAW (un-affined) template,
                   so u_rv is pushed to include A^-1.

    Those two flow_rv objectives contradict each other whenever the affine is
    not identity, so 'rv' and 'rv_noaffine' bracket the compromise the network
    settled on. Reporting all three turns the ambiguity into a measurement.
    Returns numpy vertex sets plus diagnostics."""
    model, template_seg = ctx['model'], ctx['template_seg']
    flow_fw, flow_rv, _lambda, affine_matrix = model(template_seg, sample_seg)
    flow_inv, _aff, parity = svf_inverse_flow(model, template_seg, sample_seg)
    return push_from_flows(verts_t, flow_fw, flow_rv, affine_matrix, n_iter, alpha,
                           flow_inv=flow_inv, parity=parity)


@torch.no_grad()
def push_from_flows(verts_t, flow_fw, flow_rv, affine_matrix, n_iter=500, alpha=0.5,
                    flow_inv=None, parity=None):
    """The push itself, given flows already computed — so a caller that has
    already run the model (e.g. evaluate_all.py) does not pay for a second
    forward pass. See push_mesh for what the modes mean; 'svf_inv' is added when
    flow_inv (from svf_inverse_flow) is supplied and is the one to prefer."""
    v_a = undo_affine(verts_t, affine_matrix)                 # affine-aligned space
    o, res = invert_numerically(flow_fw, v_a, n_iter, alpha)
    pushes = {
        'numeric': o,
        'rv': v_a + sample_field(flow_rv, v_a),
        'rv_noaffine': verts_t + sample_field(flow_rv, verts_t),
    }
    if flow_inv is not None:
        pushes['svf_inv'] = v_a + sample_field(flow_inv, v_a)

    det = None
    if affine_matrix is not None:
        det = float(torch.linalg.det(affine_matrix[0][:, :3]).item())
    return {
        'pushes': {k: v.cpu().numpy() for k, v in pushes.items()},
        'v_aligned': v_a.cpu().numpy(),
        'inv_residual_norm': res.cpu().numpy(),
        'affine_det': det,
        'svf_parity': parity,
    }


# =============================================================================
# Mesh loading
# =============================================================================

def _read_hemi(path, ref):
    """FreeSurfer surface -> (verts in normalized coords of ref, faces).
    surface-RAS + cras = scanner RAS, then world_to_norm."""
    coords, faces, meta = nib.freesurfer.read_geometry(path, read_metadata=True)
    world = coords + meta.get('cras', np.zeros(3))
    return world_to_norm(world, ref).astype(np.float32), np.asarray(faces, dtype=np.int64)


def _surf_pair(mesh_dir):
    """First existing (lh, rh) naming pair under mesh_dir, or None."""
    for lh_name, rh_name in SURF_NAMES:
        lh, rh = Path(mesh_dir) / lh_name, Path(mesh_dir) / rh_name
        if lh.is_file() and rh.is_file():
            return str(lh), str(rh)
    return None


def _join_hemis(lh, rh, ref):
    lv, lf = _read_hemi(lh, ref)
    rv, rf = _read_hemi(rh, ref)
    return (np.concatenate([lv, rv]), np.concatenate([lf, rf + len(lv)]), len(lv))


def ref_for(seg_npy_path):
    """The .nii.gz sibling of a seg one-hot .npy — the geometry reference."""
    return str(seg_npy_path).replace('_onehot.npy', '.nii.gz')


def load_template_mesh(cfg, template_surf):
    """Template mesh in normalized coords. Priority: --template_surf, then
    config data.template_wm_mesh_path (.npz from wm_template.py), then the
    /scans/->/meshes/ sibling of the template subject."""
    tpl_seg = cfg['data']['template_seg_path']
    if template_surf:
        ref = nib.load(ref_for(tpl_seg))
        return _join_hemis(template_surf[0], template_surf[1], ref)

    npz = cfg['data'].get('template_wm_mesh_path')
    if npz and Path(npz).is_file():
        d = np.load(npz)
        print(f"[mesh] template mesh from {npz}")
        return (d['verts_norm'].astype(np.float32), d['faces'].astype(np.int64),
                int(d['n_lh']))

    mesh_dir = str(Path(tpl_seg).parent).replace('/scans/', '/meshes/')
    pair = _surf_pair(mesh_dir)
    if pair is None:
        raise FileNotFoundError(
            f"No template mesh. Tried config data.template_wm_mesh_path ({npz}) and "
            f"{mesh_dir}/{{lh,rh}}.white[.surf]. Pass --template_surf <lh> <rh>, or build "
            f"the .npz with wm_template.py on the invertible-deform-SRegNET branch.")
    print(f"[mesh] template mesh from {pair[0]} + {pair[1]}")
    return _join_hemis(pair[0], pair[1], nib.load(ref_for(tpl_seg)))


def load_subject_white(subject_dir, ref):
    """The subject's OWN lh+rh.white — the genus-0 gold standard, a like-for-like
    white surface rather than a marching-cubes isosurface of the WM label.
    Returns (verts_norm, faces, n_lh) or (None, None, None). n_lh matters for
    distance metrics: lh and rh are two separate surfaces whose medial walls sit
    ~1-3 mm apart, so a nearest neighbour must be looked up within a hemisphere."""
    pair = _surf_pair(str(subject_dir).replace('/scans/', '/meshes/'))
    if pair is None:
        return None, None, None
    return _join_hemis(pair[0], pair[1], ref)


def save_deformed_surf(verts_norm, faces, ref, path):
    """Write the deformed mesh as a FreeSurfer surface in surface-RAS
    (= world - cras, cras being the world coord of the volume centre), so it
    opens against the subject's own volume in freeview."""
    world = norm_to_world(verts_norm, ref)
    cras = apply_affine(ref.affine, np.array(ref.shape[:3], dtype=float) / 2.0)
    nib.freesurfer.write_geometry(str(path), world - cras, faces)


# =============================================================================
# Figure
# =============================================================================

def render_sample(name, n, gt_wm, curves, slices, dist, flip, res_mm, push_name,
                  output, dpi=140, tmpl_label='template (undeformed)'):
    """3 ortho slices over the subject's GT WM mask + a distance histogram.
    Draw order matters: template and GT first (context), deformed last (on top)."""
    has_dist = dist is not None
    ncol = 4 if has_dist else 3
    fig, axes = plt.subplots(1, ncol, figsize=(5.8 * ncol, 6.5))

    for a in range(3):
        ax = axes[a]
        s = int(slices[a])
        ax.imshow(np.take(gt_wm, s, axis=a).T, cmap='gray', origin='lower', aspect='equal')
        for (verts, faces), col, lw in curves:
            seg = slice_segments(norm_to_vox(verts, n), faces, a, s)
            if len(seg):
                ax.add_collection(LineCollection(list(seg), colors=col,
                                                 linewidths=lw, alpha=0.9))
        ax.set_xlim(0, n - 1)
        ax.set_ylim(0, n - 1)
        ax.set_title(f'{AXIS_NAMES[a]} @ {s} / {n - 1}', fontsize=11)
        ax.axis('off')

    axes[0].legend(handles=[
        Line2D([0], [0], color=TEMPLATE_C, lw=2, label=tmpl_label),
        Line2D([0], [0], color=GT_C, lw=2, label="subject GT white surface"),
        Line2D([0], [0], color=DEFORM_C, lw=2, label='deformed template mesh'),
    ], loc='lower right', fontsize=7, framealpha=0.6)

    if has_dist:
        axh = axes[3]
        axh.hist(dist['both'], bins=60, color='steelblue')
        for v, c, lab in [(dist['both'].mean(), 'red', 'mean'),
                          (float(np.median(dist['both'])), 'orange', 'median'),
                          (float(np.percentile(dist['both'], 95)), 'purple', 'hd95')]:
            axh.axvline(v, color=c, ls='--', lw=1.2, label=f'{lab} {v:.2f} mm')
        axh.axvline(dist['init_mean'], color='gray', ls=':', lw=1.2,
                    label=f"undeformed {dist['init_mean']:.2f} mm")
        axh.set_xlabel('symmetric deformed-mesh <-> GT white distance (mm)')
        axh.set_ylabel('vertices')
        axh.set_title('distance distribution')
        axh.legend(fontsize=8)

    head = f'{name} — genus-0 template mesh pushed to subject (push={push_name})'
    if has_dist:
        head += (f"   (sym mean {dist['both'].mean():.2f} / hd95 "
                 f"{np.percentile(dist['both'], 95):.2f} mm; undeformed "
                 f"{dist['init_mean']:.2f} mm)")
    head += f'   [triangle flip {flip:.4f}%; inverse residual {res_mm:.3f} mm]'
    fig.suptitle(head, fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description='Push the genus-0 template mesh through a trained field')
    ap.add_argument('--model', required=True,
                    help='run dir, a best_model.pth path, or a run name under output.base_dir')
    ap.add_argument('--output_dir', default=None, help='default <run_dir>/viz_mesh')
    ap.add_argument('--config', default=None, help='config.yaml (default: ./config.yaml)')
    ap.add_argument('--val_txt', default=None, help='override config data.val_txt')
    ap.add_argument('--num_samples', type=int, default=5,
                    help='subjects, evenly spaced across the val list')
    ap.add_argument('--sample_idxs', default=None, help='comma list of val indices')
    ap.add_argument('--subjects', default=None,
                    help='comma list of subject-name substrings, e.g. 0287,0203 '
                         '(takes precedence over --sample_idxs / --num_samples)')
    ap.add_argument('--suffix', default='',
                    help='appended to every output basename, so a rerun does not '
                         "overwrite an earlier one (e.g. --suffix _affine_tmpl)")
    ap.add_argument('--template_surf', nargs=2, metavar=('LH', 'RH'), default=None,
                    help='explicit template lh/rh white surfaces')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--affine', choices=['auto', 'on', 'off'], default='auto')
    ap.add_argument('--push', choices=['svf_inv', 'numeric', 'rv', 'rv_noaffine'],
                    default='svf_inv',
                    help='which push to draw and save (all are always measured); '
                         'svf_inv = exp(-v), exact for an SVF model')
    ap.add_argument('--no_save_mesh', action='store_true', help='skip writing the deformed .surf')
    ap.add_argument('--dpi', type=int, default=140)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = resolve_checkpoint(args.model, cfg)
    use_affine = detect_affine(ckpt) if args.affine == 'auto' else args.affine == 'on'
    out = Path(args.output_dir).expanduser() if args.output_dir \
        else Path(ckpt).parent.parent / 'viz_mesh'
    out.mkdir(parents=True, exist_ok=True)
    write_git_sha(out)

    print(f"[mesh] checkpoint = {ckpt}")
    print(f"[mesh] device = {args.device} | affine = {use_affine} | output_dir = {out}", flush=True)

    ctx = setup_inference(ckpt, args.config, args.device, use_affine=use_affine, verbose=True)
    device, n = ctx['device'], ctx['target_size'][0]

    verts_np, faces, n_lh = load_template_mesh(cfg, args.template_surf)
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device)
    inside = ((verts_np >= -1) & (verts_np <= 1)).all(1).mean() * 100
    print(f"[mesh] {len(verts_np):,} verts ({n_lh:,} lh) | {len(faces):,} faces | "
          f"{inside:.1f}% inside the [-1,1] FOV (low => wrong ref or axis order)", flush=True)

    d = cfg['data']
    ds = SegDataset(args.val_txt or d['val_txt'], d['template_seg_path'],
                    target_size=ctx['target_size'], seg_filename=d['seg_filename'],
                    preload=False)
    if args.subjects:
        names = [Path(sd).name for sd in ds.subject_dirs]
        idxs = []
        for pat in args.subjects.split(','):
            hits = [i for i, nm in enumerate(names) if pat.strip() in nm]
            if not hits:
                raise SystemExit(f"[mesh] no val subject matches '{pat.strip()}' "
                                 f"(val list looks like {names[:3]})")
            idxs += hits
        idxs = sorted(set(idxs))
    elif args.sample_idxs:
        idxs = [int(x) for x in args.sample_idxs.split(',')]
    else:
        idxs = sorted(set(np.linspace(0, len(ds) - 1,
                                      min(args.num_samples, len(ds))).astype(int).tolist()))
    for i in idxs:
        if not (0 <= i < len(ds)):
            raise IndexError(f"sample index {i} out of range [0, {len(ds)})")
    print(f"[mesh] {len(ds)} val subjects | pushing mesh for {idxs}", flush=True)

    results = []
    for i in idxs:
        sample = ds[i]
        subject_dir = ds.subject_dirs[i]
        name = Path(subject_dir).name
        sample_seg = sample['sample_seg'].unsqueeze(0).to(device)

        p = push_mesh(ctx, verts_t, sample_seg)
        v_s = p['pushes'][args.push]
        flip = triangle_flip_fraction(verts_np, v_s, faces)

        ref_path = ref_for(Path(subject_dir) / d['seg_filename'])
        ref = nib.load(ref_path) if Path(ref_path).is_file() else None
        dist, gt, per_mode = None, None, {}
        res_mm, nonconv = float('nan'), float('nan')

        if ref is not None:
            # Fixed-point residual: did the numerical inverse actually converge?
            # Large here and 'numeric' is untrustworthy too.
            mm = mm_per_norm(ref)
            res_mm = float(p['inv_residual_norm'].mean() * mm)
            nonconv = float((p['inv_residual_norm'] * mm > 1.0).mean() * 100)

            gt_v, gt_f, gt_n_lh = load_subject_white(subject_dir, ref)
            if gt_v is not None:
                gt = (gt_v, gt_f)
                gt_w = norm_to_world(gt_v, ref)
                from evaluate_all import sym_dist, hemi_scores

                init_per_hemi = sym_dist(norm_to_world(verts_np, ref), gt_w, n_lh, gt_n_lh)
                init_hemi_scores = hemi_scores(init_per_hemi, 'init_')
                # Score every candidate push against the same GT surface.
                for mode, v in p['pushes'].items():
                    w = norm_to_world(v, ref)
                    per_hemi = sym_dist(w, gt_w, n_lh, gt_n_lh)
                    scores = hemi_scores(per_hemi, 'sym_')
                    per_mode[mode] = {
                        'sym_mean_mm': scores['sym_mean_mm'],
                        'sym_hd95_mm': scores['sym_hd95_mm'],
                        'flip_pct': triangle_flip_fraction(verts_np, v, faces),
                    }
                    if mode == args.push:
                        # Collect all directed distances into 'both' for visualization
                        both = np.concatenate([d for h in per_hemi for d in h])
                        dist = {'both': both, 'init_mean': init_hemi_scores['init_mean_mm']}

        gt_wm = (sample['sample_seg'].argmax(0).numpy() == WM_LABEL).astype(np.float32)
        # Yellow = the template as the UNet sees it (A^-1 applied), so the gap to
        # blue is the dense flow alone. Identical to the raw mesh when affine off.
        curves = [((p['v_aligned'], faces), TEMPLATE_C, 1.2)]
        if gt is not None:
            curves.append((gt, GT_C, 1.2))
        curves.append(((v_s, faces), DEFORM_C, 1.7))
        tmpl_label = 'template (affine-aligned)' if use_affine else 'template (undeformed)'
        render_sample(name, n, gt_wm, curves, centroid_slices(gt_wm > 0),
                      dist, flip, res_mm, args.push, out / f'{name}{args.suffix}_mesh.png',
                      args.dpi, tmpl_label=tmpl_label)

        if not args.no_save_mesh and ref is not None:
            save_deformed_surf(v_s, faces, ref,
                               out / f'{name}{args.suffix}_deformed.white.surf')

        r = {'subject': name, 'idx': int(i), 'push_drawn': args.push,
             'flip_pct': flip,
             'flip_pct_dense_only': triangle_flip_fraction(p['v_aligned'], v_s, faces),
             'affine_det': p['affine_det'],
             'inverse_residual_mm': res_mm,
             'inverse_nonconverged_pct': nonconv,
             'per_push_mode': per_mode}
        if dist is not None:
            r.update({'sym_mean_mm': float(dist['both'].mean()),
                      'sym_hd95_mm': float(np.percentile(dist['both'], 95)),
                      'undeformed_mean_mm': dist['init_mean']})
            table = " | ".join(f"{m} {v['sym_mean_mm']:6.2f}mm flip {v['flip_pct']:6.3f}%"
                               for m, v in per_mode.items())
            best = min(per_mode, key=lambda m: per_mode[m]['sym_mean_mm'])
            print(f"[mesh] {name:>20s} undeformed {dist['init_mean']:6.2f}mm || {table} "
                  f"|| best={best} drawn={args.push} | inv residual {res_mm:.3f} mm "
                  f"({nonconv:.2f}% verts >1mm)", flush=True)
        else:
            print(f"[mesh] {name:>20s} | no GT white surface under "
                  f"{str(subject_dir).replace('/scans/', '/meshes/')} — figure only | "
                  f"flip {flip:.4f}% | inv residual {res_mm:.3f} mm", flush=True)
        results.append(r)

    means = {}
    for k in results[0]:
        vals = [r[k] for r in results
                if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        if vals:
            means[k] = float(np.mean(vals))
    mode_means = {m: {kk: float(np.mean([r['per_push_mode'][m][kk] for r in results]))
                      for kk in ('sym_mean_mm', 'sym_hd95_mm', 'flip_pct')}
                  for m in results[0]['per_push_mode']}
    summary_path = out / f'summary{args.suffix}.json'
    summary_path.write_text(json.dumps(
        {'checkpoint': ckpt, 'affine': use_affine, 'n_verts': len(verts_np),
         'n_faces': len(faces), 'sample_idxs': idxs, 'push_drawn': args.push,
         'per_sample': results, 'mean': means, 'mean_per_push_mode': mode_means}, indent=2))

    print(f"\n[mesh] N={len(results)}")
    if mode_means:
        print(f"[mesh] undeformed baseline {means['undeformed_mean_mm']:.3f} mm — "
              f"any push must beat this")
        for m, v in sorted(mode_means.items(), key=lambda kv: kv[1]['sym_mean_mm']):
            print(f"[mesh]   push={m:<12s} sym {v['sym_mean_mm']:6.3f} mm | "
                  f"hd95 {v['sym_hd95_mm']:6.3f} mm | flip {v['flip_pct']:.4f}%")
        print(f"[mesh] 'numeric' inverts flow_fw (the field the Dice was trained on) — "
              f"prefer it unless its inverse residual is large")
    print(f"[mesh] mean triangle flip {means['flip_pct']:.4f}% for push={args.push}  "
          f"(topology gate — voxel folding does not certify this)")
    print(f"[mesh] mean inverse residual {means['inverse_residual_mm']:.3f} mm, "
          f"{means['inverse_nonconverged_pct']:.2f}% of verts >1mm  "
          f"(large => the fixed point did not converge, 'numeric' is unreliable too)")
    print(f"[mesh] summary -> {summary_path}")


if __name__ == '__main__':
    main()
