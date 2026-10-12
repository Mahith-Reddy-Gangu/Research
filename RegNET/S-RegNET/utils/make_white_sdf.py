"""
Signed distance volumes to the FreeSurfer white surface, one per subject plus
the template — the field the surface loss samples at the pushed mesh vertices.

The loss needs a mm distance at arbitrary (non-lattice) vertex positions, so it
reads a precomputed volume with trilinear grid_sample instead of running a
nearest-triangle query inside the training loop. This script builds that volume
on each subject's OWN seg4.nii.gz grid (256^3, 1 mm):

    distance   two passes. A cKDTree over a dense point sampling of the mesh
               (vertices + edge midpoints + face centroids, so no surface point
               is further than ~0.3 mm from a sample) covers the whole 10 mm
               band; every voxel within EXACT_BAND_MM is then refined with the
               exact point-to-triangle distance to its K_TRI nearest triangles.
               Both passes are upper bounds on the true distance, so the minimum
               of the two is taken and the near field — the only part the loss
               gradient uses — is exact to float.
    sign       ray parity (solid voxelization along i), not a normal test: a
               voxel centre is inside iff an odd number of triangle crossings
               lie below it on its ray. Exact by construction, so thin WM blades
               and the two hemispheres' nested crossings cannot flip sign.

Outputs (data files go NEXT TO EACH SUBJECT'S SEGS, not into --output_dir):
    <scan_dir>/white_sdf_v1.npy      float16 (256,256,256) in (i,j,k) layout,
                                     mm, negative inside the white surface,
                                     clamped to +-10
    <scan_dir>/white_surf_v1.npz     verts_norm float32 (N,3) normalized coords
                                     of that grid (lh then rh), n_lh int
    <template mesh dir>/template_white_sdf_v1.npy   same format, template grid
    <output_dir>/sdf_qc.csv          one row per subject (appended)
    <output_dir>/GIT_SHA.txt         provenance

Run from the S-RegNET directory (CPU only; needs torch + nibabel + scipy):
    python utils/make_white_sdf.py --lists train,val --template --workers 32 \\
        --output_dir ~/shared_scratch/surface_v1/sdf_qc
    # --subjects OASIS_OAS1_0115_MR1,OASIS_OAS1_0146_MR1   just these
    # --template --lists ''                                just the template
    # --force                                              recompute existing
"""

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))          # repo-root modules, whatever the cwd is
sys.path.insert(0, str(_ROOT / 'utils'))

from git_provenance import write_git_sha
from inference import load_config
from visualize_mesh import (_join_hemis, _surf_pair, load_template_mesh,
                            norm_to_world, ref_for, world_to_norm)

SDF_NAME = 'white_sdf_v1.npy'
SURF_NAME = 'white_surf_v1.npz'
TEMPLATE_SDF_NAME = 'template_white_sdf_v1.npy'
CLAMP_MM = 10.0             # the loss never looks further out than this
EXACT_BAND_MM = 2.0         # exact point-to-triangle refinement inside this
K_TRI = 16                  # candidate triangles per refined voxel (k=48 agrees
                            # to 6e-7 mm mean / 0.065 mm max, and k=32 exactly)
CHUNK = 100_000             # refinement chunk (voxels), keeps the temporaries
                            # inside cache — a 2x bigger chunk ran 6x slower
QC_FIELDS = ('subject', 'surf_abs_mean_mm', 'surf_abs_p99_mm', 'sign_ok_frac',
             'inside_ml', 'n_verts', 'seconds')
PASS = {'surf_abs_mean_mm': 0.15, 'surf_abs_p99_mm': 0.6}   # sign_ok_frac > 0.99


# =============================================================================
# Geometry
# =============================================================================

def surface_samples(verts, faces):
    """Dense point sampling of the mesh: vertices, edge midpoints, centroids."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return np.concatenate([verts, (a + b) / 2, (b + c) / 2, (c + a) / 2,
                           (a + b + c) / 3])


def point_tri_dist(p, tri):
    """Exact point-to-triangle distance (Ericson, Real-Time Collision Detection
    5.1.5): the closest point is the barycentric solution clamped to the
    triangle, i.e. one of the 7 Voronoi regions. Vectorized over (M,3) points
    and their (M,3,3) triangles; the region tests are mutually exclusive, so
    they can overwrite the face solution in any order."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    def ratio(num, den):
        return num / np.where(den == 0, 1.0, den)

    w1, w2 = ratio(vb, va + vb + vc), ratio(vc, va + vb + vc)
    q = a + w1[:, None] * ab + w2[:, None] * ac                     # face
    q = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, q)            # vertices
    q = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, q)
    q = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, q)
    t = ratio(d1, d1 - d3)                                          # edges
    q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[:, None], a + t[:, None] * ab, q)
    t = ratio(d2, d2 - d6)
    q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[:, None], a + t[:, None] * ac, q)
    t = ratio(d4 - d3, (d4 - d3) + (d5 - d6))
    q = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[:, None],
                 b + t[:, None] * (c - b), q)
    return np.linalg.norm(p - q, axis=1)


def unsigned_distance(ref, verts_w, faces):
    """Unsigned mm distance from every voxel centre of ref to the mesh, clamped
    to CLAMP_MM. Voxels outside the band get exactly CLAMP_MM (their true
    distance is larger, and the loss is flat there)."""
    shape = ref.shape[:3]
    zooms = np.linalg.norm(ref.affine[:3, :3], axis=0)
    inv = np.linalg.inv(ref.affine)
    samples = surface_samples(verts_w, faces)

    # Band: EDT from the voxels the surface samples land in. Those centres are
    # within half a voxel diagonal of the surface, so a 2 mm margin over the
    # clamp cannot cut a voxel whose true distance is below CLAMP_MM.
    seed = np.zeros(shape, dtype=bool)
    sij = np.rint(apply_affine(inv, samples)).astype(np.int64)
    keep = ((sij >= 0) & (sij < np.array(shape))).all(1)
    seed[tuple(sij[keep].T)] = True
    band = distance_transform_edt(~seed, sampling=zooms) <= CLAMP_MM + 2.0

    idx = np.argwhere(band)
    pts = apply_affine(ref.affine, idx.astype(np.float64))
    # compact_nodes=False is a 3x faster query here for identical answers.
    d, _ = cKDTree(samples, balanced_tree=False,
                   compact_nodes=False).query(pts, workers=1)

    near = np.flatnonzero(d <= EXACT_BAND_MM)
    tri = verts_w[faces]
    ctree = cKDTree(tri.mean(1))
    for s in range(0, len(near), CHUNK):
        sel = near[s:s + CHUNK]
        _, cand = ctree.query(pts[sel], k=K_TRI, workers=1)
        e = point_tri_dist(np.repeat(pts[sel], K_TRI, axis=0),
                           tri[cand.ravel()]).reshape(cand.shape).min(1)
        d[sel] = np.minimum(d[sel], e)

    out = np.full(shape, CLAMP_MM, dtype=np.float32)
    out[band] = np.minimum(d, CLAMP_MM)
    return out


def inside_mask(vox, faces, shape):
    """Solid voxelization by ray parity: the voxel centre (i,j,k) is inside the
    closed surface iff an odd number of triangle crossings sit below it on the
    ray along i. Exact — no normals, no medial-axis heuristics — so a 1.5 mm WM
    blade and the lh/rh pair (two disjoint closed surfaces, parity adds up) are
    both signed correctly."""
    Ni, Nj, Nk = shape
    tri = vox[faces]                                        # (F,3,3) voxel coords
    x, y, z = tri[..., 0], tri[..., 1], tri[..., 2]         # rays: i along x, lattice (j,k)
    j0 = np.maximum(np.ceil(y.min(1)), 0).astype(np.int64)
    j1 = np.minimum(np.floor(y.max(1)), Nj - 1).astype(np.int64)
    k0 = np.maximum(np.ceil(z.min(1)), 0).astype(np.int64)
    k1 = np.minimum(np.floor(z.max(1)), Nk - 1).astype(np.int64)
    nj, nk = np.maximum(j1 - j0 + 1, 0), np.maximum(k1 - k0 + 1, 0)
    n = nj * nk
    f = np.repeat(np.arange(len(faces)), n)                 # face x candidate ray
    loc = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    jj, kk = j0[f] + loc // nk[f], k0[f] + loc % nk[f]

    # Barycentric coords of the lattice point in the (j,k)-projected triangle.
    y0, y1, y2 = y[f, 0], y[f, 1], y[f, 2]
    z0, z1, z2 = z[f, 0], z[f, 1], z[f, 2]
    det = (y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0)
    py, pz = jj - y0, kk - z0
    ok = det != 0
    den = np.where(ok, det, 1.0)
    u = np.where(ok, (py * (z2 - z0) - pz * (y2 - y0)) / den, -1.0)
    v = np.where(ok, (pz * (y1 - y0) - py * (z1 - z0)) / den, -1.0)
    hit = (u >= 0) & (v >= 0) & (u + v <= 1)

    i_cross = (x[f, 0] + u * (x[f, 1] - x[f, 0]) + v * (x[f, 2] - x[f, 0]))[hit]
    # A crossing at i_cross flips the parity of every voxel with i >= ceil(i_cross).
    start = np.clip(np.ceil(i_cross), 0, Ni).astype(np.int64)
    flat = (start * Nj + jj[hit]) * Nk + kk[hit]
    acc = np.bincount(flat, minlength=(Ni + 1) * Nj * Nk).reshape(Ni + 1, Nj, Nk)
    par = (acc[:Ni] & 1).astype(np.uint8)
    # uint8 cumsum wraps mod 256, which is even, so the parity bit survives.
    return (np.cumsum(par, axis=0, dtype=np.uint8) & 1).astype(bool)


def outward_vertex_normals(verts_w, faces):
    """Unit vertex normals in world mm, oriented outward. The area-weighted sum
    of face normals is flipped globally when the mesh's signed volume is
    negative (lh and rh share FreeSurfer's face orientation, so one sign does)."""
    a, b, c = verts_w[faces[:, 0]], verts_w[faces[:, 1]], verts_w[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    n = np.zeros_like(verts_w)
    for col in range(3):
        np.add.at(n, faces[:, col], fn)
    n *= np.sign((a * np.cross(b, c)).sum() / 6.0)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


# =============================================================================
# Build + QC
# =============================================================================

def sample_sdf(vol, pts):
    """Trilinear sample of an (D,H,W) volume at normalized (x,y,z) pts (N,3) —
    the exact op the training loss uses (align_corners=False, border padding)."""
    g = torch.from_numpy(np.ascontiguousarray(pts)).float().view(1, -1, 1, 1, 3)
    s = F.grid_sample(vol, g, mode='bilinear', padding_mode='border',
                      align_corners=False)
    return s.view(-1).numpy()


def qc(sdf, verts_norm, verts_w, faces, ref, inside):
    """|SDF| at the mesh's own vertices (should be ~0) and the sign at vertices
    stepped +-0.5 mm along the outward normal."""
    vol = torch.from_numpy(sdf.astype(np.float32))[None, None]
    at_surf = np.abs(sample_sdf(vol, verts_norm))
    nrm = outward_vertex_normals(verts_w, faces)
    s_out = sample_sdf(vol, world_to_norm(verts_w + 0.5 * nrm, ref))
    s_in = sample_sdf(vol, world_to_norm(verts_w - 0.5 * nrm, ref))
    vox_ml = abs(np.linalg.det(ref.affine[:3, :3])) / 1000.0
    return {'surf_abs_mean_mm': float(at_surf.mean()),
            'surf_abs_p99_mm': float(np.percentile(at_surf, 99)),
            'sign_ok_frac': float(((s_out > 0).sum() + (s_in < 0).sum())
                                  / (2 * len(verts_norm))),
            'inside_ml': float(inside.sum() * vox_ml),
            'n_verts': len(verts_norm)}


def load_mesh(job, ref):
    """The mesh this job signs: the template npz, or the subject's lh+rh white."""
    if job['kind'] == 'template':
        return load_template_mesh(job['cfg'], None)
    pair = _surf_pair(job['mesh_dir'])
    if pair is None:
        raise FileNotFoundError(f"no {{lh,rh}}.white[.surf] under {job['mesh_dir']}")
    return _join_hemis(pair[0], pair[1], ref)


def build_one(job):
    """One subject (or the template): write the SDF (+ vertex npz) and QC it."""
    torch.set_num_threads(1)            # one core per worker, not one per op
    t0 = time.time()
    try:
        ref = nib.load(job['ref_path'])
        verts_norm, faces, n_lh = load_mesh(job, ref)
        verts_w = norm_to_world(verts_norm, ref)
        d = unsigned_distance(ref, verts_w, faces)
        inside = inside_mask(apply_affine(np.linalg.inv(ref.affine), verts_w),
                             faces, ref.shape[:3])
        sdf = np.where(inside, -d, d).astype(np.float16)
        np.save(job['sdf_path'], sdf)
        if job['surf_path']:
            np.savez(job['surf_path'], verts_norm=verts_norm.astype(np.float32),
                     n_lh=n_lh)
        row = qc(sdf, verts_norm, verts_w, faces, ref, inside)
    except Exception as e:                      # one bad subject must not kill 412
        return {'subject': job['name'], 'error': f'{type(e).__name__}: {e}',
                'seconds': time.time() - t0}
    row.update(subject=job['name'], seconds=time.time() - t0)
    return row


# =============================================================================
# Main
# =============================================================================

def collect_jobs(cfg, args):
    d = cfg['data']
    sdf_name = d.get('white_sdf_filename', SDF_NAME)
    surf_name = d.get('white_surf_filename', SURF_NAME)
    jobs, seen = [], set()

    for which in [w.strip() for w in args.lists.split(',') if w.strip()]:
        key = f'{which}_txt'
        if key not in d:
            raise SystemExit(f'[sdf] unknown list {which!r} (config has no data.{key})')
        for line in Path(d[key]).read_text().splitlines():
            if not line.strip():
                continue
            scan_dir = Path(line.strip()).parent
            if scan_dir.name in seen:
                continue
            seen.add(scan_dir.name)
            jobs.append({'kind': 'subject', 'name': scan_dir.name,
                         'ref_path': ref_for(scan_dir / d['seg_filename']),
                         'mesh_dir': str(scan_dir).replace('/scans/', '/meshes/'),
                         'sdf_path': str(scan_dir / sdf_name),
                         'surf_path': str(scan_dir / surf_name)})

    if args.subjects:
        want = [s.strip() for s in args.subjects.split(',') if s.strip()]
        missing = [s for s in want if s not in seen]
        if missing:
            raise SystemExit(f'[sdf] not in --lists {args.lists!r}: {missing}')
        jobs = [j for j in jobs if j['name'] in want]

    if args.template:
        npz = Path(d['template_wm_mesh_path'])
        tpl = d.get('template_white_sdf_path') or str(npz.parent / TEMPLATE_SDF_NAME)
        jobs.append({'kind': 'template', 'name': 'TEMPLATE', 'cfg': cfg,
                     'ref_path': ref_for(d['template_seg_path']),
                     'sdf_path': tpl, 'surf_path': None})
    return jobs


def main():
    ap = argparse.ArgumentParser(
        description='Signed distance volumes to the FreeSurfer white surface, '
                    'per subject + template (input to the surface loss)')
    ap.add_argument('--config', default='config.yaml',
                    help='config.yaml (a bare name resolves against the S-RegNET dir)')
    ap.add_argument('--lists', default='train,val',
                    help="which of data.train_txt / data.val_txt to walk "
                         "(empty string: none, e.g. --template --lists '')")
    ap.add_argument('--subjects', default=None,
                    help='comma list of subject dir names (default: all in --lists)')
    ap.add_argument('--template', action='store_true',
                    help='also build the template SDF')
    ap.add_argument('--workers', type=int, default=32,
                    help='processes over subjects (each one is single-threaded). '
                         'Peak ~0.5 GB of RAM per worker: the int64 parity '
                         'bincount over 257x256x256 is 135 MB on its own, plus '
                         'the face x ray temporaries, the EDT and the KD-tree '
                         'over ~3M surface samples — 32 workers want ~16 GB')
    ap.add_argument('--force', action='store_true',
                    help='recompute subjects whose output files already exist')
    ap.add_argument('--output_dir', required=True,
                    help='QC csv + GIT_SHA only — the volumes go next to each seg')
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_file() and not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    cfg = load_config(str(cfg_path))

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    write_git_sha(out)

    jobs = collect_jobs(cfg, args)
    if not args.force:
        todo = [j for j in jobs
                if not (Path(j['sdf_path']).is_file()
                        and (j['surf_path'] is None
                             or Path(j['surf_path']).is_file()))]
        print(f'[sdf] {len(jobs) - len(todo)} of {len(jobs)} already built — skipping '
              f'(--force to recompute)', flush=True)
        jobs = todo
    if not jobs:
        raise SystemExit('[sdf] nothing to do')

    csv_path = out / 'sdf_qc.csv'
    new = not csv_path.is_file()
    print(f'[sdf] {len(jobs)} to build | workers {args.workers} | '
          f'clamp +-{CLAMP_MM:g} mm, exact band {EXACT_BAND_MM:g} mm | '
          f'qc -> {csv_path}', flush=True)

    rows, bad = [], []
    t0 = time.time()
    with open(csv_path, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=(*QC_FIELDS, 'error'))
        if new:
            writer.writeheader()
        with Pool(max(1, args.workers)) as pool:
            for i, row in enumerate(pool.imap_unordered(build_one, jobs), 1):
                writer.writerow(row)
                fh.flush()                      # partial results survive a kill
                if 'error' in row:
                    bad.append(row)
                    print(f"[sdf] {i:>3}/{len(jobs)} {row['subject']:>20s} "
                          f"FAILED {row['error']}", flush=True)
                    continue
                rows.append(row)
                flag = ''
                if any(row[k] > v for k, v in PASS.items()) or row['sign_ok_frac'] <= 0.99:
                    flag = '  <-- outside QC thresholds'
                print(f"[sdf] {i:>3}/{len(jobs)} {row['subject']:>20s} "
                      f"|sdf| at verts mean {row['surf_abs_mean_mm']:.4f} / p99 "
                      f"{row['surf_abs_p99_mm']:.4f} mm | sign_ok "
                      f"{row['sign_ok_frac']:.5f} | inside {row['inside_ml']:.0f} mL | "
                      f"{row['seconds']:.1f} s{flag}", flush=True)

    print(f'\n[sdf] built {len(rows)} in {(time.time() - t0) / 60:.1f} min '
          f'({len(bad)} failed)')
    if rows:
        for k in ('surf_abs_mean_mm', 'surf_abs_p99_mm', 'sign_ok_frac', 'seconds'):
            v = np.array([r[k] for r in rows])
            print(f'[sdf]   {k:<17s} mean {v.mean():8.4f} | min {v.min():8.4f} | '
                  f'max {v.max():8.4f}')
        n_out = sum(1 for r in rows
                    if any(r[k] > v for k, v in PASS.items()) or r['sign_ok_frac'] <= 0.99)
        print(f'[sdf] {n_out}/{len(rows)} outside the QC thresholds '
              f'(mean < 0.15 mm, p99 < 0.6 mm, sign_ok > 0.99)')
    for r in bad:
        print(f"[sdf] FAILED {r['subject']}: {r['error']}")


if __name__ == '__main__':
    main()
