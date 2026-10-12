"""
Render the deformed template WM mesh ITSELF — the FreeSurfer surfaces written
by utils/push_optimized_mesh.py::save_deformed_surf — as 3D surface figures.
NOT a marching-cubes isosurface of any voxel volume; the triangles drawn are
the pushed template mesh's own faces.

Per subject, one figure: one row per --rows entry (each mesh colored per-vertex
by its mm distance to the subject's GT white surface, nearest GT vertex via
cKDTree) plus a neutral GT-white reference row, times 3 views (left lateral,
superior, right lateral). The same row set is repeated zoomed on the dorsal
region (top third of the shared z-range) so gyral/sulcal tracking is visible.
Distances are always computed on the FULL vertex set; --decimate thins FACES
for display only (stated in the figure footer).

A second figure per subject (<subj>_mesh_overlay.png, --no_overlay to skip):
2D slice overlays in the style of utils/visualize_mesh.py's render_sample —
rows = the --rows arms, columns = 3 orthogonal slices at the centroid of the
GT WM mask, background = the native-grid WM mask in grayscale, curves = the
exact mesh-plane intersections (triangles cut by the slice plane, NOT voxel
contours) of the undeformed template (yellow, drawn beneath), the GT white
(green) and the deformed mesh (blue).

Coordinate conventions (mirrors utils/visualize_mesh.py, torch-free copies):
  * GT white:  read_geometry(read_metadata=True); world = coords + metadata
    cras (visualize_mesh._read_hemi).
  * deformed surfs: save_deformed_surf wrote world - cras with
    cras = apply_affine(ref.affine, ref.shape/2) and NO volume_info, so the
    file has no cras metadata — world is recovered by re-adding that computed
    cras from the subject's seg4.nii.gz (ref_for sibling of seg_filename).
  * undeformed template: data.template_wm_mesh_path .npz stores verts_norm in
    the shared normalized [-1,1] frame (exactly what push_optimized_mesh feeds
    the deformation) — no cras involved; its position on the subject grid is
    norm_to_world(verts_norm, subject ref), the same placement that tool's
    'undeformed' distances and gold curve use.
  Both meshes therefore live in the same scanner-RAS mm frame; distances and
  renders happen there directly (no normalized-coordinate round trip).
  * overlays: world -> NATIVE voxel (i,j,k) via inv(ref.affine); the on-disk
    one-hot is (5, Ni, Nj, Nk) straight from the same nii (convert_one_hot.py),
    so mask slices and mesh cross-sections share the frame with no resize.

Pure CPU — numpy, scipy, nibabel, matplotlib (+ yaml for the config); no torch,
safe to run beside a GPU job. Numbers and figures only.

Run from the S-RegNET directory (CPU, GPU cluster ok):
    python bandlimit_opt/render_deformed_mesh.py \\
        --rows baseline=<mesh_eval_dir>:baseline,refined=<mesh_eval_dir>:d64 \\
        --config config.yaml --output_dir <renders_dir>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import nibabel as nib
from nibabel.affines import apply_affine
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3d projection

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))          # repo-root modules, whatever the cwd is

from git_provenance import write_git_sha

SURF_NAMES = [('lh.white', 'rh.white'), ('lh.white.surf', 'rh.white.surf')]
VIEWS = [('left lateral', dict(elev=0, azim=180)),
         ('superior', dict(elev=90, azim=-90)),
         ('right lateral', dict(elev=0, azim=0))]
CMAP = 'viridis'
GT_COLOR = '#b9b9c4'
BG = '#f5f5f0'
TXT = '#222222'
WM_LABEL = 3                                   # 0 bg, 1 cortex, 2 subGM, 3 WM, 4 CSF
GT_CURVE, DEFORM_CURVE = '#32CD32', '#00BFFF'  # slice-overlay curve colors
TEMPLATE_CURVE = '#FFD700'                     # undeformed template curve
# Anatomical plane you see when slicing ALONG an axis with this direction code.
_PLANE = {'L': 'sagittal', 'R': 'sagittal', 'A': 'coronal', 'P': 'coronal',
          'S': 'axial', 'I': 'axial'}


def axis_names(ref):
    """Per-voxel-axis panel names from the nii's actual orientation — the
    native FreeSurfer-conformed grid is NOT z/y/x ordered, so hardcoded
    axial/coronal/sagittal labels would be wrong."""
    codes = nib.orientations.aff2axcodes(ref.affine)
    return [f'{_PLANE[c]} ({c})' for c in codes]


# =============================================================================
# Coordinate + mesh loading (torch-free copies of utils/visualize_mesh.py
# helpers — that module imports torch at module level, so it cannot be
# imported here; only the metadata-cras / computed-cras handling is kept)
# =============================================================================

def ref_for(seg_npy_path):
    """The .nii.gz sibling of a seg one-hot .npy — the geometry reference."""
    return str(seg_npy_path).replace('_onehot.npy', '.nii.gz')


def volume_cras(ref):
    """World coord of the volume centre — the cras save_deformed_surf subtracted."""
    return apply_affine(ref.affine, np.array(ref.shape[:3], dtype=float) / 2.0)


def _surf_pair(mesh_dir):
    """First existing (lh, rh) naming pair under mesh_dir, or None."""
    for lh_name, rh_name in SURF_NAMES:
        lh, rh = Path(mesh_dir) / lh_name, Path(mesh_dir) / rh_name
        if lh.is_file() and rh.is_file():
            return str(lh), str(rh)
    return None


def _read_hemi_world(path):
    """FreeSurfer surface -> verts in scanner-world mm: surface-RAS + metadata
    cras (real FreeSurfer files carry it; visualize_mesh._read_hemi convention)."""
    coords, faces, meta = nib.freesurfer.read_geometry(path, read_metadata=True)
    world = coords + meta.get('cras', np.zeros(3))
    return np.asarray(world, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def load_subject_white_world(subject_dir):
    """The subject's own lh+rh.white joined, in world mm. (None, None) if absent."""
    pair = _surf_pair(str(subject_dir).replace('/scans/', '/meshes/'))
    if pair is None:
        return None, None
    lv, lf = _read_hemi_world(pair[0])
    rv, rf = _read_hemi_world(pair[1])
    return np.concatenate([lv, rv]), np.concatenate([lf, rf + len(lv)])


def load_deformed_world(surf_path, ref):
    """A <subj>_<arm>_deformed.white.surf back to world mm. save_deformed_surf
    wrote world - cras WITHOUT volume_info, so there is no cras metadata to
    read — re-add the same computed cras (exact inverse of the writer)."""
    coords, faces = nib.freesurfer.read_geometry(str(surf_path))
    return (np.asarray(coords, dtype=np.float64) + volume_cras(ref),
            np.asarray(faces, dtype=np.int64))


# =============================================================================
# Slice overlays (torch-free copies: centroid_slices from utils/visualize_run.py,
# slice_segments + the imshow transpose/origin convention from
# utils/visualize_mesh.py render_sample — here on the NATIVE voxel grid)
# =============================================================================

def world_to_vox(world, ref):
    """World mm -> ref's native voxel (i,j,k). The on-disk one-hot is
    (5, Ni, Nj, Nk) straight from the same nii, so voxel axis a of the mesh
    coords slices the mask's spatial axis a directly."""
    return apply_affine(np.linalg.inv(ref.affine), world)


def norm_to_world(norm, ref):
    """Normalized [-1,1] (x,y,z) -> world mm on ref's grid (torch-free copy of
    visualize_mesh.norm_to_world). The template .npz stores verts_norm; with
    the SUBJECT's ref this is the placement push_optimized_mesh scores as
    'undeformed' — the true starting position of the mesh that gets pushed."""
    Ni, Nj, Nk = ref.shape[:3]
    ijk = np.stack([((norm[:, 2] + 1) * Ni - 1) / 2,
                    ((norm[:, 1] + 1) * Nj - 1) / 2,
                    ((norm[:, 0] + 1) * Nk - 1) / 2], axis=1)
    return apply_affine(ref.affine, ijk)


def load_wm_mask(subject_dir, seg_filename):
    """Native-grid WM mask from the pre-resize one-hot: argmax over the 5
    channels == WM_LABEL. None if the .npy is absent."""
    npy = Path(subject_dir) / seg_filename
    if not npy.is_file():
        return None
    return np.load(str(npy)).argmax(axis=0) == WM_LABEL


def centroid_slices(mask):
    """Per-axis centroid slice index of a boolean volume (its densest cross
    section) — a more informative cut than the mid-slice."""
    out = []
    for a in range(3):
        others = tuple(o for o in range(3) if o != a)
        prof = mask.sum(axis=others)
        tot = prof.sum()
        out.append(int(round((np.arange(len(prof)) * prof).sum() / tot)) if tot
                   else len(prof) // 2)
    return out


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


def render_overlay(subj, mesh_rows, gt, tpl, wm_mask, ref, out_path, args):
    """2D companion figure: rows = the --rows arms, columns = 3 orthogonal
    slices at the WM-mask centroid. Background = native-grid GT WM mask in
    grayscale; curves = the meshes themselves cut by the slice plane
    (undeformed template yellow drawn beneath, GT white green, deformed blue),
    all in native voxel coords so they share the frame.
    imshow(slice.T, origin='lower') as in visualize_mesh.render_sample: the
    panel x axis is the first remaining voxel axis, y the second."""
    gt_vox = world_to_vox(gt[0], ref)
    tpl_vox = (world_to_vox(norm_to_world(tpl[0], ref), ref)
               if tpl is not None else None)
    slices = centroid_slices(wm_mask)
    dims = wm_mask.shape
    names = axis_names(ref)

    nrows = len(mesh_rows)
    fig, axes = plt.subplots(nrows, 3, figsize=(13.0, 4.3 * nrows + 0.7),
                             squeeze=False, facecolor=BG)
    for r, (label, verts_w, faces, _dist) in enumerate(mesh_rows):
        def_vox = world_to_vox(verts_w, ref)
        curves = [(gt_vox, gt[1], GT_CURVE, 1.2),
                  (def_vox, faces, DEFORM_CURVE, 1.7)]
        if tpl_vox is not None:                # yellow first: blue/green on top
            curves.insert(0, (tpl_vox, tpl[1], TEMPLATE_CURVE, 1.2))
        for a in range(3):
            ax = axes[r][a]
            s = int(slices[a])
            ax.imshow(np.take(wm_mask, s, axis=a).astype(np.float32).T,
                      cmap='gray', origin='lower', aspect='equal')
            for vox, fc, col, lw in curves:
                seg = slice_segments(vox, fc, a, s)
                if len(seg):
                    ax.add_collection(LineCollection(list(seg), colors=col,
                                                     linewidths=lw, alpha=0.9))
            p, q = [x for x in range(3) if x != a]
            ax.set_xlim(0, dims[p] - 1)
            ax.set_ylim(0, dims[q] - 1)
            ax.set_title(f'{names[a]} @ {s} / {dims[a] - 1}', fontsize=10,
                         color=TXT)
            ax.axis('off')
        axes[r][0].text(-0.04, 0.5, label, transform=axes[r][0].transAxes,
                        ha='right', va='center', fontsize=9, color=TXT)

    handles = [
        Line2D([0], [0], color=GT_CURVE, lw=2, label='GT white (mesh)'),
        Line2D([0], [0], color=DEFORM_CURVE, lw=2, label='deformed template (mesh)'),
    ]
    if tpl_vox is not None:
        handles.insert(0, Line2D([0], [0], color=TEMPLATE_CURVE, lw=2,
                                 label='undeformed template (mesh)'))
    axes[0][0].legend(handles=handles, loc='lower right', fontsize=7,
                      framealpha=0.6)
    fig.suptitle(f'{subj} — mesh cross-sections over GT WM mask '
                 f'(native grid, slices at WM centroid)', fontsize=12,
                 fontweight='bold', color=TXT)
    fig.tight_layout(rect=(0.07, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


# =============================================================================
# Rendering
# =============================================================================

def draw_panel(ax, verts, faces_show, dist, vmax, lims, view, zoom):
    """One trisurf panel. dist=None renders neutral shaded GT; otherwise flat
    per-face colors = mean of the (full-resolution) per-vertex distances."""
    ax.set_axis_off()
    if len(faces_show) == 0:
        ax.text2D(0.5, 0.5, 'empty region', transform=ax.transAxes,
                  ha='center', va='center', fontsize=9, color=TXT)
        return
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    if dist is None:
        ax.plot_trisurf(x, y, z, triangles=faces_show, color=GT_COLOR,
                        linewidth=0, antialiased=False, shade=True)
    else:
        p = ax.plot_trisurf(x, y, z, triangles=faces_show, cmap=CMAP,
                            linewidth=0, antialiased=False, shade=False)
        p.set_array(dist[faces_show].mean(axis=1))
        p.set_clim(0.0, vmax)
    (xlo, ylo, zlo), (xhi, yhi, zhi) = lims
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_zlim(zlo, zhi)
    aspect = (xhi - xlo, yhi - ylo, zhi - zlo)
    try:
        ax.set_box_aspect(aspect, zoom=zoom)
    except TypeError:                      # older matplotlib: no zoom kwarg
        ax.set_box_aspect(aspect)
    try:
        ax.set_proj_type('ortho')
    except Exception:
        pass
    ax.view_init(**view)


def _bbox(verts_list, pad=2.0):
    allv = np.concatenate(verts_list)
    return allv.min(axis=0) - pad, allv.max(axis=0) + pad


def render_subject(subj, mesh_rows, gt, out_path, args):
    """mesh_rows: [(label, verts_world, faces, dist_mm)]; gt: (verts, faces).
    Full-view rows (arms + GT) then dorsal close-up rows of the same meshes."""
    k = max(1, args.decimate)
    gt_v, gt_f = gt
    all_meshes = [(lb, v, f[::k], d) for lb, v, f, d in mesh_rows]
    all_meshes.append(('GT white\n(reference)', gt_v, gt_f[::k], None))

    lims_full = _bbox([v for _, v, _, _ in all_meshes])
    zmin = min(v[:, 2].min() for _, v, _, _ in all_meshes)
    zmax = max(v[:, 2].max() for _, v, _, _ in all_meshes)
    z_lo = zmin + (zmax - zmin) * (2.0 / 3.0)      # dorsal = top third of z-range

    rows = [(lb, v, fd, d, lims_full, 1.25) for lb, v, fd, d in all_meshes]
    zoom_verts = []
    zoom_rows = []
    for lb, v, fd, d in all_meshes:
        keep = fd[(v[fd][:, :, 2] >= z_lo).all(axis=1)]
        if len(keep):
            zoom_verts.append(v[np.unique(keep)])
        zoom_rows.append((lb.split('\n')[0] + '\n(dorsal close-up)', v, keep, d))
    lims_zoom = _bbox(zoom_verts) if zoom_verts else lims_full
    rows += [(lb, v, fd, d, lims_zoom, 1.45) for lb, v, fd, d in zoom_rows]

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 3, figsize=(13.0, 3.6 * nrows + 0.9),
                             subplot_kw={'projection': '3d'}, squeeze=False,
                             facecolor=BG)
    for r, (label, verts, faces_show, dist, lims, zoom) in enumerate(rows):
        for c, (view_name, view) in enumerate(VIEWS):
            draw_panel(axes[r][c], verts, faces_show, dist, args.max_dist_mm,
                       lims, view, zoom)
            if r == 0:
                axes[r][c].set_title(view_name, fontsize=11, color=TXT)
        axes[r][0].text2D(-0.04, 0.5, label, transform=axes[r][0].transAxes,
                          rotation=0, ha='right', va='center', fontsize=9,
                          color=TXT)

    sm = cm.ScalarMappable(norm=Normalize(0.0, args.max_dist_mm), cmap=CMAP)
    sm.set_array([])                       # older matplotlib needs this for colorbar
    fig.subplots_adjust(left=0.15, right=0.90, top=0.965, bottom=0.030,
                        wspace=0.0, hspace=0.03)
    cax = fig.add_axes([0.925, 0.42, 0.012, 0.18])
    fig.colorbar(sm, cax=cax, extend='max',
                 label='nearest-vertex distance to GT white (mm)')

    n_faces = len(mesh_rows[0][2])
    n_verts = len(mesh_rows[0][1])
    fig.suptitle(f'{subj} — deformed template WM mesh (surface render), '
                 f'color = mm distance to GT white', fontsize=12,
                 fontweight='bold', color=TXT)
    fig.text(0.01, 0.004,
             f'display decimation: every {k}th face shown '
             f'({len(mesh_rows[0][2][::k]):,} of {n_faces:,} faces) — NOT the '
             f'real mesh density; distances/colors computed on the full '
             f'{n_verts:,}-vertex mesh; close-up rows: z >= {z_lo:.1f} mm '
             f'(top third of z-range); colors capped at {args.max_dist_mm:g} mm',
             fontsize=8, color='#555555')
    fig.savefig(out_path, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def parse_rows(spec):
    """'label=mesh_eval_dir:arm,...' -> [(label, dir Path, arm)]."""
    rows = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        label, eq, rest = item.partition('=')
        mesh_dir, col, arm = rest.rpartition(':')
        if not (label and eq and mesh_dir and col and arm):
            raise SystemExit(f"[render] bad --rows item '{item}' "
                             f"(want label=mesh_eval_dir:arm)")
        rows.append((label, Path(mesh_dir).expanduser(), arm))
    if not rows:
        raise SystemExit('[render] --rows is empty')
    return rows


def main():
    ap = argparse.ArgumentParser(
        description='Render deformed template WM meshes (the .surf files '
                    'themselves — no marching cubes, no voxel isosurfaces) '
                    'colored by mm distance to the subject GT white')
    ap.add_argument('--rows', required=True,
                    help='comma list of label=mesh_eval_dir:arm triples; each '
                         'contributes one figure row reading '
                         '<dir>/<subj>_<arm>_deformed.white.surf')
    ap.add_argument('--distilled_dir', default=None,
                    help="extra 'distilled' row: mesh_eval dir of a distilled "
                         "run, read as <dir>/<subj>_baseline_deformed.white.surf")
    ap.add_argument('--config', default='config.yaml',
                    help='config.yaml (a bare name resolves against the S-RegNET dir)')
    ap.add_argument('--subjects', default=None,
                    help='comma list of subject dir names (default: first 3 of '
                         'val_txt — head of the list, template subject excluded, '
                         'input seg present — same rule as '
                         'instance_opt_bandlimited.py)')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--dpi', type=int, default=150)
    ap.add_argument('--max_dist_mm', type=float, default=3.0,
                    help='colorbar cap (mm); larger distances clip to the top color')
    ap.add_argument('--decimate', type=int, default=4,
                    help='keep every k-th FACE for display only; distances always '
                         'use the full vertex set')
    ov = ap.add_mutually_exclusive_group()
    ov.add_argument('--overlay', dest='overlay', action='store_true', default=True,
                    help='also write <subj>_mesh_overlay.png: 2D mesh-plane '
                         'cross-sections over the GT WM mask (default on)')
    ov.add_argument('--no_overlay', dest='overlay', action='store_false',
                    help='skip the 2D slice-overlay figures')
    args = ap.parse_args()

    row_specs = parse_rows(args.rows)
    if args.distilled_dir:
        row_specs.append(('distilled', Path(args.distilled_dir).expanduser(),
                          'baseline'))
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_file() and not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    d = cfg['data']

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    write_git_sha(out)

    # Same subject-dir lookup as push_optimized_mesh: val_txt lines -> parents.
    subject_dirs = [Path(p).parent for p in
                    Path(d['val_txt']).read_text().splitlines() if p.strip()]
    by_name = {sd.name: sd for sd in subject_dirs}
    if args.subjects:
        wanted = [s.strip() for s in args.subjects.split(',') if s.strip()]
        missing = [s for s in wanted if s not in by_name]
        if missing:
            raise SystemExit(f"[render] subjects not in {d['val_txt']}: {missing}")
        chosen = [by_name[s] for s in wanted]
    else:
        # Same selection as instance_opt_bandlimited.py: head of val_txt,
        # template subject excluded, input seg required.
        input_name = d.get('input_seg_filename') or d['seg_filename']
        template_subject = Path(d['template_seg_path']).parent.name
        chosen = [sd for sd in subject_dirs
                  if sd.name != template_subject
                  and (sd / input_name).is_file()][:3]
    if not chosen:
        raise SystemExit('[render] no subjects to render')

    # Undeformed template mesh for the overlay's yellow curve — the exact .npz
    # push_optimized_mesh feeds the deformation (verts_norm, normalized coords).
    tpl = None
    if args.overlay:
        tpl_path = d.get('template_wm_mesh_path')
        if tpl_path and Path(tpl_path).is_file():
            z = np.load(tpl_path)
            tpl = (np.asarray(z['verts_norm'], dtype=np.float64),
                   np.asarray(z['faces'], dtype=np.int64))
            print(f'[render] undeformed template mesh from {tpl_path}')
        else:
            print(f'[render] no template mesh at {tpl_path} — '
                  f'overlay yellow curve skipped', flush=True)

    print(f"[render] rows = {[(lb, str(md), arm) for lb, md, arm in row_specs]}")
    print(f"[render] subjects = {[sd.name for sd in chosen]} | "
          f"decimate = {args.decimate} | max_dist = {args.max_dist_mm} mm | "
          f"output_dir = {out}", flush=True)

    for subject_dir in chosen:
        subj = subject_dir.name
        ref_path = ref_for(subject_dir / d['seg_filename'])
        if not Path(ref_path).is_file():
            print(f'[render] skip {subj}: no reference volume {ref_path}', flush=True)
            continue
        ref = nib.load(ref_path)

        gt_v, gt_f = load_subject_white_world(subject_dir)
        if gt_v is None:
            print(f"[render] skip {subj}: no GT white surface under "
                  f"{str(subject_dir).replace('/scans/', '/meshes/')}", flush=True)
            continue
        tree = cKDTree(gt_v)

        mesh_rows = []
        for label, mesh_dir, arm in row_specs:
            surf = mesh_dir / f'{subj}_{arm}_deformed.white.surf'
            if not surf.is_file():
                print(f'[render] {subj}: missing {surf} — row skipped', flush=True)
                continue
            verts_w, faces = load_deformed_world(surf, ref)
            dist = tree.query(verts_w)[0]
            mesh_rows.append((f'{label}\nmean {dist.mean():.2f} / '
                              f'max {dist.max():.2f} mm', verts_w, faces, dist))
        if not mesh_rows:
            print(f'[render] skip {subj}: no deformed meshes found', flush=True)
            continue

        fig_path = out / f'{subj}_mesh_render.png'
        render_subject(subj, mesh_rows, (gt_v, gt_f), fig_path, args)
        stats = '; '.join(f"{lb.splitlines()[0]} {lb.splitlines()[1]}"
                          for lb, _, _, _ in mesh_rows)
        print(f'[render] {subj}: {stats} -> {fig_path}', flush=True)

        if args.overlay:
            wm_mask = load_wm_mask(subject_dir, d['seg_filename'])
            if wm_mask is None:
                print(f"[render] {subj}: no {d['seg_filename']} — overlay skipped",
                      flush=True)
            elif wm_mask.shape != ref.shape[:3]:
                print(f'[render] {subj}: one-hot grid {wm_mask.shape} != ref grid '
                      f'{ref.shape[:3]} — overlay skipped', flush=True)
            else:
                ov_path = out / f'{subj}_mesh_overlay.png'
                render_overlay(subj, mesh_rows, (gt_v, gt_f), tpl, wm_mask, ref,
                               ov_path, args)
                print(f'[render] {subj}: overlay -> {ov_path}', flush=True)


if __name__ == '__main__':
    main()
