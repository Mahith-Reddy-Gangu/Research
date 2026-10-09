"""
Topology baseline of the INPUT segmentations — the numbers the pushed genus-0
mesh is compared against.

A voxel seg cannot self-intersect, and a marching-cubes surface of one is a
level set, embedded by construction — its self-intersection count is ~0 by
definition, not by cleanliness. SynthSeg's topological dirt lives elsewhere:

    components   spurious islands (26-connectivity voxel count, plus the
                 surface's own connected-component count)
    handles      holes through the surface: Euler characteristic chi = V-E+F,
                 total handles = (2C - chi)/2 over C closed components

Those are exactly the defects the deformed genus-0 template cannot have, which
is the deliverable's argument. Three baselines per subject, all at NATIVE
resolution (no 128 resize — honest input topology):

    synthseg   WM channel of synthseg_onehot_v1.npy -> marching cubes
    gt         WM channel of seg4_onehot.npy        -> marching cubes
               (multi-component BY construction: channel 3 folds in cerebellar
                WM + brainstem, and ventricles/deep GM perforate it)
    white      the subject's own FreeSurfer ?h.white — the meaningful
               SELF-INTERSECTION baseline (raw template 0406 was 0.1007%),
               context for the pushed mesh's number

Self-int on the MC surfaces is measured (not asserted) so the ~0 is on the
record; skip it with --no_mc_self_int once the point is made — it is the slow
half on a ~1-3M-face MC mesh.

CPU-only (no GPU, no model). Run from the S-RegNET directory:
    python utils/synthseg_topology_baseline.py \\
        --config config.yaml --num_samples 10 \\
        --output_dir <scratch>/synthseg_topology_baseline
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
import scipy.sparse as sp
from scipy.ndimage import label as ndi_label
from skimage.measure import marching_cubes

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'utils'))

from git_provenance import write_git_sha
from inference import load_config
from visualize_mesh import WM_LABEL, load_subject_white, norm_to_world, ref_for
from mesh_flip_probe import self_intersections


def mesh_stats(verts, faces):
    """V, E, F, Euler characteristic, connected components, handles.
    handles = (2C - chi)/2 holds for closed orientable surfaces — the padded
    marching-cubes output is closed; skimage's Lewiner MC resolves the cube
    ambiguities that would break manifoldness."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                faces[:, [2, 0]]]), axis=1)
    e = np.unique(e, axis=0)
    chi = len(verts) - len(e) + len(faces)
    adj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                        shape=(len(verts), len(verts)))
    n_c = int(sp.csgraph.connected_components(adj, directed=False)[0])
    return {'verts': len(verts), 'faces': len(faces), 'euler_chi': int(chi),
            'surf_components': n_c, 'handles': int((2 * n_c - chi) // 2)}


def surface_of(mask, ref):
    """Closed WM surface of a binary mask in world mm (pad so the border seals)."""
    padded = np.pad(mask, 1).astype(np.float32)
    verts, faces, _n, _v = marching_cubes(padded, 0.5)
    return apply_affine(ref.affine, verts - 1.0), faces.astype(np.int64)


def voxel_components(mask):
    """26-connectivity island count + how dominant the largest island is."""
    lab, n = ndi_label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return {'vox_components': 0, 'vox_largest_pct': 0.0}
    sizes = np.bincount(lab.ravel())[1:]
    return {'vox_components': int(n),
            'vox_largest_pct': float(sizes.max() / sizes.sum() * 100)}


def seg_row(tag, npy_path, ref, name, mc_self_int):
    """All topology numbers for one seg's WM channel at native resolution."""
    mask = np.load(npy_path)[WM_LABEL] > 0.5
    row = {f'{tag}_{k}': v for k, v in voxel_components(mask).items()}
    w, faces = surface_of(mask, ref)
    row.update({f'{tag}_{k}': v for k, v in mesh_stats(w, faces).items()})
    if mc_self_int:
        si, _ = self_intersections(w, faces, f'{tag} {name}')
        row[f'{tag}_si_faces'] = si['si_faces']
        row[f'{tag}_si_faces_pct'] = si['si_faces_pct']
    return row


def main():
    ap = argparse.ArgumentParser(
        description='Topology baseline of SynthSeg/GT WM segs + FreeSurfer white self-int')
    ap.add_argument('--config', default=None, help='config.yaml (default: ./config.yaml)')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--val_txt', default=None, help='override config data.val_txt')
    ap.add_argument('--num_samples', type=int, default=10,
                    help='subjects, evenly spaced across the val list (match the eval)')
    ap.add_argument('--sample_idxs', default=None, help='comma list of val indices')
    ap.add_argument('--synthseg_filename', default='synthseg_onehot_v1.npy')
    ap.add_argument('--no_mc_self_int', action='store_true',
                    help='skip self-int on the MC surfaces (provably ~0; the slow half)')
    ap.add_argument('--no_white', action='store_true',
                    help='skip the FreeSurfer ?h.white self-int baseline')
    args = ap.parse_args()

    cfg_path = Path(args.config or 'config.yaml').expanduser()
    if not cfg_path.is_file() and not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    cfg = load_config(str(cfg_path))
    d = cfg['data']
    template_subject = Path(d['template_seg_path']).parent.name

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    write_git_sha(out)

    subject_dirs = [Path(p).parent for p in
                    Path(args.val_txt or d['val_txt']).read_text().splitlines() if p.strip()]
    if args.sample_idxs:
        idxs = [int(x) for x in args.sample_idxs.split(',')]
    else:
        idxs = sorted(set(np.linspace(0, len(subject_dirs) - 1,
                                      min(args.num_samples, len(subject_dirs)))
                          .astype(int).tolist()))
    print(f"[topo] {len(subject_dirs)} val subjects | evaluating {idxs} | "
          f"mc_self_int = {not args.no_mc_self_int} | output_dir = {out}", flush=True)

    csv_path = out / 'topology_per_subject.csv'
    rows, writer, fh = [], None, None
    try:
        for i in idxs:
            sd = subject_dirs[i]
            name = sd.name
            if name == template_subject:
                print(f"[topo] skip {name} (template subject)", flush=True)
                continue
            synth_path = sd / args.synthseg_filename
            if not synth_path.is_file():
                print(f"[topo] skip {name}: no {synth_path}", flush=True)
                continue
            ref = nib.load(ref_for(sd / d['seg_filename']))

            row = {'subject': name, 'idx': int(i)}
            row.update(seg_row('synthseg', synth_path, ref, name, not args.no_mc_self_int))
            row.update(seg_row('gt', sd / d['seg_filename'], ref, name,
                               not args.no_mc_self_int))

            if not args.no_white:
                gt_v, gt_f, _n_lh = load_subject_white(sd, ref)
                if gt_v is not None:
                    # 2 closed hemispheres: components must be 2, handles 0 —
                    # FreeSurfer guarantees the genus; self-int is its one flaw.
                    si, _ = self_intersections(norm_to_world(gt_v, ref), gt_f,
                                               f'white {name}')
                    row.update({'white_si_faces': si['si_faces'],
                                'white_si_faces_pct': si['si_faces_pct'],
                                **{f'white_{k}': v for k, v in
                                   mesh_stats(gt_v, gt_f).items()}})

            if fh is None:
                fh = open(csv_path, 'w', newline='')
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
            assert writer is not None
            writer.writerow({k: row.get(k) for k in writer.fieldnames})
            fh.flush()
            rows.append(row)

            msg = (f"[topo] {name:>20s} synthseg: {row['synthseg_vox_components']:4d} islands "
                   f"| {row['synthseg_handles']:4d} handles || gt: "
                   f"{row['gt_vox_components']:4d} islands | {row['gt_handles']:4d} handles")
            if 'synthseg_si_faces_pct' in row:
                msg += f" || MC self-int {row['synthseg_si_faces_pct']:.4f}%"
            if 'white_si_faces_pct' in row:
                msg += f" || white self-int {row['white_si_faces_pct']:.4f}%"
            print(msg, flush=True)
    finally:
        if fh is not None:
            fh.close()

    if not rows:
        raise SystemExit("no subjects evaluated")

    stats = {}
    for c in [k for k in rows[0] if k not in ('subject', 'idx')]:
        vals = np.array([r[c] for r in rows if c in r], dtype=float)
        if len(vals):
            stats[c] = {'mean': float(vals.mean()), 'min': float(vals.min()),
                        'max': float(vals.max()), 'n': int(len(vals))}
    print(f"\n=== input topology baseline (n={len(rows)}) ===")
    print(f"{'metric':<28s} {'mean':>10s} {'min':>10s} {'max':>10s}")
    for c, s in stats.items():
        print(f"{c:<28s} {s['mean']:10.2f} {s['min']:10.2f} {s['max']:10.2f}")

    (out / 'topology_summary.json').write_text(json.dumps(
        {'synthseg_filename': args.synthseg_filename, 'wm_label': WM_LABEL,
         'n_subjects': len(rows), 'stats': stats, 'per_subject': rows}, indent=2))
    print(f"\n[topo] the deliverable's counterpart numbers: pushed template mesh = "
          f"2 components, 0 handles (structural), self-int from the eval run")
    print(f"[topo] per-subject CSV -> {csv_path}")
    print(f"[topo] summary        -> {out / 'topology_summary.json'}")


if __name__ == '__main__':
    main()
