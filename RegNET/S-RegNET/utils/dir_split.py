"""Direction split of the symmetric mesh distance: mesh->GT vs GT->mesh, per arm.

Numbers only. Reads <subj>_<arm>_deformed.white.surf written by
utils/push_optimized_mesh.py and the subject's own lh+rh.white through the
repo's loaders, so both live in the same world frame.

usage: python utils/dir_split.py <mesh_dir> --config config_meshsup.yaml [--arms a,b]
"""
import argparse
import re
import glob
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
from scipy.spatial import cKDTree

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))          # repo-root modules, whatever the cwd is
sys.path.insert(0, str(_ROOT / 'utils'))

from inference import load_config
from visualize_mesh import load_subject_white, norm_to_world, ref_for  # noqa
from push_optimized_mesh import load_cortex_mask  # noqa

N_LH_TPL = 163842


def agg(rows, key):
    vals = [np.concatenate([r[f'{key}_lh'], r[f'{key}_rh']]) for r in rows]
    return (np.mean([x.mean() for x in vals]),
            np.mean([np.percentile(x, 95) for x in vals]),
            np.mean([np.percentile(x, 99) for x in vals]),
            np.mean([(x > 2.0).mean() for x in vals]),
            np.mean([x[x > 2.0].sum() / x.sum() for x in vals]))


def main():
    ap = argparse.ArgumentParser(
        description='Direction split of the symmetric mesh distance: mesh->GT '
                    'vs GT->mesh, per arm')
    ap.add_argument('mesh_dir')
    ap.add_argument('--config', default='config.yaml',
                    help='config.yaml (a bare name resolves against the S-RegNET dir)')
    ap.add_argument('--arms', default=None, help='comma list of arms to keep')
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_file() and not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    cfg = load_config(str(cfg_path))
    d = cfg['data']
    scans = Path(d['template_seg_path']).parent.parent
    cortex = load_cortex_mask(d.get('cortex_labels_dir'))
    if cortex is None:
        raise SystemExit('[split] cortex mask missing — cortex columns would be meaningless')

    mesh_dir = Path(args.mesh_dir).expanduser()
    arms = args.arms.split(',') if args.arms else None

    rows = []
    for f in sorted(glob.glob(str(mesh_dir / '*_deformed.white.surf'))):
        m = re.match(r'(OASIS_OAS1_\d+_MR1)_(.+)_deformed\.white\.surf$', Path(f).name)
        subj, arm = m.group(1), m.group(2)
        if arms and arm not in arms:
            continue
        ref = nib.load(ref_for(scans / subj / d['seg_filename']))
        gt_n, gt_f, gt_nlh = load_subject_white(scans / subj, ref)
        gt_w = norm_to_world(gt_n, ref)
        v, _ = nib.freesurfer.read_geometry(f)
        cras = apply_affine(ref.affine, np.array(ref.shape[:3], dtype=float) / 2.0)
        mesh_w = v + cras
        same_index = (len(gt_w) == len(mesh_w)) and (gt_nlh == N_LH_TPL)
        out = {'subj': subj, 'arm': arm}
        for tag, (ms, gs) in (('lh', (slice(0, N_LH_TPL), slice(0, gt_nlh))),
                              ('rh', (slice(N_LH_TPL, None), slice(gt_nlh, None)))):
            mw, gw = mesh_w[ms], gt_w[gs]
            m2g = cKDTree(gw).query(mw)[0]
            g2m = cKDTree(mw).query(gw)[0]
            out[f'm2g_{tag}'] = m2g; out[f'g2m_{tag}'] = g2m
            cx = cortex[ms]
            out[f'm2g_cx_{tag}'] = m2g[cx]
            out[f'g2m_cx_{tag}'] = g2m[cortex[gs]] if same_index else g2m
        rows.append(out)

    print(f'{"arm":12s} n  | {"m2g mean":>9s} {"p95":>6s} {"p99":>6s} {">2mm%":>6s} {"share":>6s} | {"g2m mean":>9s} {"p95":>6s} {"p99":>6s} {">2mm%":>6s} {"share":>6s} '
          f'| {"sym mean":>8s} | cortex: {"m2g":>6s} {"g2m":>6s}')
    for arm in sorted({r['arm'] for r in rows}):
        rs = [r for r in rows if r['arm'] == arm]
        a, b = agg(rs, 'm2g'), agg(rs, 'g2m')
        ca, cb = agg(rs, 'm2g_cx'), agg(rs, 'g2m_cx')
        sym = np.mean([np.concatenate([r['m2g_lh'], r['m2g_rh'], r['g2m_lh'], r['g2m_rh']]).mean() for r in rs])
        print(f'{arm:12s} {len(rs):2d} | {a[0]:9.3f} {a[1]:6.2f} {a[2]:6.2f} {100*a[3]:6.2f} {100*a[4]:6.1f} | {b[0]:9.3f} {b[1]:6.2f} {b[2]:6.2f} {100*b[3]:6.2f} {100*b[4]:6.1f} '
              f'| {sym:8.3f} | {ca[0]:6.3f} {cb[0]:6.3f}')
    print()
    print('per subject (mean mm): subj arm m2g g2m')
    for r in rows:
        m2g = np.concatenate([r['m2g_lh'], r['m2g_rh']]).mean()
        g2m = np.concatenate([r['g2m_lh'], r['g2m_rh']]).mean()
        print(f'{r["subj"]} {r["arm"]:10s} {m2g:.3f} {g2m:.3f}')


if __name__ == '__main__':
    main()
