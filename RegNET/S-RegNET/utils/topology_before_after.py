"""
Before/after topology of the SynthSeg -> pushed-template pipeline, in one table.

The claim is "we correct the topology of an incorrect segmentation map". This
puts the numbers on it: per subject, the WM surface you would get straight from
the SynthSeg seg (marching cubes) vs the pushed genus-0 template mesh the method
actually outputs. Nothing is regenerated — inputs and pushed meshes are read.

Two facts this makes concrete:

  handles / components   SynthSeg's WM surface carries hundreds of spurious
                         handles and dozens of components. Marching cubes
                         OVER-counts (staircase adds tiny handles), so read the
                         input number as an upper bound on the real defect. The
                         pushed template mesh has 0 handles / 2 components BY
                         CONSTRUCTION: the warp moves vertices, never faces, so
                         V-E+F and the genus are invariant. The tool verifies
                         the faces are byte-identical to the template's — that
                         invariance IS the topology guarantee, not a measurement
                         that could have come out otherwise.

  self-intersection      A marching-cubes surface is a level set, embedded by
                         definition, so its self-int is ~0 — meaningless as an
                         "after" number. The pushed MESH is not a level set and
                         CAN self-cross; its self-int % is the ONE per-subject
                         after-number that can actually fail, so it is the honest
                         realness check on the genus-0 claim (an abstractly
                         genus-0 surface that self-intersects is not cleanly
                         embedded). Input self-int is off by default (slow, ~0).

Run from the S-RegNET dir (venv with nibabel/skimage/scipy; CPU is fine):
    python utils/topology_before_after.py \\
        --mesh_eval_dir <push_optimized_mesh output_dir> \\
        --config config.yaml --arm d64 \\
        --output_dir <out_dir>
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.freesurfer.io import read_geometry

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'utils'))

from git_provenance import write_git_sha
from inference import load_config
from visualize_mesh import WM_LABEL, load_template_mesh, ref_for
from synthseg_topology_baseline import mesh_stats, surface_of, voxel_components
from mesh_flip_probe import self_intersections

SURF_SUFFIX = '_deformed.white.surf'


def main():
    ap = argparse.ArgumentParser(
        description='Before/after WM topology: SynthSeg input surface vs pushed template mesh')
    ap.add_argument('--mesh_eval_dir', required=True,
                    help='dir of pushed *_<arm>_deformed.white.surf (push_optimized_mesh.py output)')
    ap.add_argument('--config', default='config.yaml',
                    help='config.yaml (a bare name resolves against the S-RegNET dir)')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--arm', default='baseline', help='which pushed arm to score')
    ap.add_argument('--synthseg_filename', default='synthseg_onehot_v1.npy')
    ap.add_argument('--subjects', default=None, help='comma list of subject dir names')
    ap.add_argument('--in_self_int', action='store_true',
                    help='also measure self-int on the SynthSeg MC surface (~0 by construction, slow)')
    args = ap.parse_args()

    mesh_eval = Path(args.mesh_eval_dir).expanduser()
    if not mesh_eval.is_dir():
        raise SystemExit(f'[topo] no such dir: {mesh_eval}')
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_file() and not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    cfg = load_config(str(cfg_path))
    d = cfg['data']

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    write_git_sha(out)

    # subjects = the pushed surfaces present for this arm
    suffix = f'_{args.arm}{SURF_SUFFIX}'
    surfs = {p.name[:-len(suffix)]: p for p in sorted(mesh_eval.glob(f'*{suffix}'))}
    if args.subjects:
        wanted = args.subjects.split(',')
        surfs = {s: surfs[s] for s in wanted if s in surfs}
    if not surfs:
        raise SystemExit(f'[topo] no *{suffix} files in {mesh_eval}')

    # val_txt line -> subject dir, by dir name (same lookup as push_optimized_mesh)
    by_name = {Path(p).parent.name: Path(p).parent
               for p in Path(d['val_txt']).read_text().splitlines() if p.strip()}

    # template reference (the topology every output inherits)
    verts_t, faces_t, _n_lh = load_template_mesh(cfg, None)
    tpl = mesh_stats(verts_t, faces_t)
    print(f"[topo] template mesh: {tpl['surf_components']} components, "
          f"{tpl['handles']} handles, chi={tpl['euler_chi']} | arm={args.arm} | "
          f"{len(surfs)} subjects\n", flush=True)

    fieldnames = ['subject',
                  'in_vox_islands', 'in_components', 'in_handles', 'in_euler_chi',
                  'in_self_int_pct',
                  'out_components', 'out_handles', 'out_euler_chi',
                  'out_faces_match_template', 'out_self_int_pct']
    rows = []
    with open(out / 'topology_before_after.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for subj, surf_path in surfs.items():
            row = {'subject': subj}

            # ---- OUTPUT: the pushed template mesh ----
            vo, fo = read_geometry(str(surf_path))
            faces_ok = bool(np.array_equal(fo, faces_t))
            om = mesh_stats(vo, fo)
            osi, _ = self_intersections(vo, fo, f'{subj} out')
            row.update({'out_components': om['surf_components'],
                        'out_handles': om['handles'],
                        'out_euler_chi': om['euler_chi'],
                        'out_faces_match_template': faces_ok,
                        'out_self_int_pct': round(osi['si_faces_pct'], 4)})

            # ---- INPUT: SynthSeg WM surface ----
            sd = by_name.get(subj)
            if sd is None:
                print(f'[topo] {subj}: not in val_txt — input columns blank', flush=True)
            else:
                ref = nib.load(ref_for(sd / d['seg_filename']))
                mask = np.load(sd / args.synthseg_filename)[WM_LABEL] > 0.5
                vc = voxel_components(mask)
                w, faces = surface_of(mask, ref)
                im = mesh_stats(w, faces)
                row.update({'in_vox_islands': vc['vox_components'],
                            'in_components': im['surf_components'],
                            'in_handles': im['handles'],
                            'in_euler_chi': im['euler_chi']})
                if args.in_self_int:
                    isi, _ = self_intersections(w, faces, f'{subj} in')
                    row['in_self_int_pct'] = round(isi['si_faces_pct'], 4)

            writer.writerow({k: row.get(k) for k in fieldnames})
            fh.flush()
            rows.append(row)
            print(f"[topo] {subj:>20s} | IN {row.get('in_handles','?'):>4} handles "
                  f"{row.get('in_components','?'):>3} comp  ->  OUT "
                  f"{row['out_handles']} handles {row['out_components']} comp "
                  f"(faces match={row['out_faces_match_template']}, "
                  f"self-int {row['out_self_int_pct']:.4f}%)", flush=True)

    # aggregate
    def col(k):
        return np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)

    print(f"\n=== before -> after (n={len(rows)}, arm={args.arm}) ===")
    ih, ic = col('in_handles'), col('in_components')
    osi = col('out_self_int_pct')
    if len(ih):
        print(f"input handles      mean {ih.mean():8.1f}  [{ih.min():.0f}, {ih.max():.0f}]  (MC upper bound)")
        print(f"input components   mean {ic.mean():8.1f}  [{ic.min():.0f}, {ic.max():.0f}]")
    print(f"output handles     0 for all {len(rows)} (structural; genus invariant under the warp)")
    print(f"output components  {tpl['surf_components']} for all (= template)")
    print(f"output self-int %  mean {osi.mean():.4f}  [{osi.min():.4f}, {osi.max():.4f}]  "
          f"(the only after-number that can fail)")
    faces_all_ok = all(r['out_faces_match_template'] for r in rows)
    print(f"faces == template for every subject: {faces_all_ok}")

    (out / 'topology_before_after.json').write_text(json.dumps(
        {'arm': args.arm, 'synthseg_filename': args.synthseg_filename,
         'wm_label': WM_LABEL, 'template': tpl, 'n_subjects': len(rows),
         'faces_all_match_template': faces_all_ok, 'per_subject': rows}, indent=2))
    print(f"\n[topo] CSV  -> {out / 'topology_before_after.csv'}")
    print(f"[topo] JSON -> {out / 'topology_before_after.json'}")


if __name__ == '__main__':
    main()
