"""
Push the repaired template WM mesh through instance-optimized velocity fields
and score the mesh deliverable: self-intersection, triangle flips, and mm
distance to the subject's own FreeSurfer white surface.

bandlimit_opt/instance_opt_bandlimited.py saves per subject x arm the raw UNet-scale
velocities (flows/<subject>_<arm>.npz: vel_fw, vel_rv, and affine_matrix when
the checkpoint has an affine head). This script replays the mesh half of
utils/evaluate_synthseg_input.py from those files, model-free: the forward
flow is exp(+vel_fw) via the model's own scaling-and-squaring loop, the exact
inverse is exp(-vel_fw), and the default push undoes the affine first, then
applies the inverse flow.

Distances come in two flavours per row: sym_* / symcx_* are vertex-to-vertex
(evaluate_all.sym_dist), p2s_* / p2s_cx_* are vertex-to-triangle
(evaluate_all.p2s_dist); *cx* drops the medial wall from the query side.
--repair runs utils/repair_template_mesh.repair (vertices only, faces untouched)
on the pushed mesh before scoring: si_faces_pct_pre keeps the pre-repair
self-intersection, every other column and the saved surf describe the repaired
mesh. t_*_s columns split the wall time per stage (push, self-int counts,
distances, repair).

Numbers and files only. Outputs into --output_dir:
    mesh_metrics.csv                  one row per subject x arm
    summary.json                      per-arm stats + full args
    <subj>_<arm>_deformed.white.surf  open in freeview against the subject volume
    <subj>_<arm>_deformed_prerepair.white.surf   with --repair --keep_prerepair
    <subj>_<arm>_mesh.png             3 ortho slices + distance histogram
    GIT_SHA.txt                       provenance

Run from the S-RegNET directory (needs torch + nibabel + scipy; GPU cluster):
    python utils/push_optimized_mesh.py \\
        --probe_dir <instance_opt_bandlimited output_dir> \\
        --config config.yaml \\
        --output_dir <that output_dir>/mesh_eval \\
        --device cuda:0 [--arms baseline,symM50] [--repair]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))          # repo-root modules, whatever the cwd is
sys.path.insert(0, str(_ROOT / 'utils'))

from git_provenance import write_git_sha
from inference import load_config
from model import SpatialTransformer
from visualize_run import centroid_slices
from visualize_mesh import (WM_LABEL, integrate_svf, load_template_mesh,
                            load_subject_white, mm_per_norm, norm_to_world,
                            push_from_flows, ref_for, render_sample,
                            save_deformed_surf, triangle_flip_fraction, world_to_norm)
from evaluate_all import sym_dist, p2s_dist, cortex_only, hemi_scores, summarize
from mesh_flip_probe import self_intersections
from repair_template_mesh import repair

SI_KEYS = ('si_faces', 'si_faces_pct', 'si_pairs', 'si_clusters', 'si_largest')
METRIC_KEYS = ('flip_pct', 'inverse_residual_mm', *SI_KEYS,
               'sym_mean_mm', 'sym_hd95_mm', 'sym_max_mm',
               'sym_mean_lh_mm', 'sym_hd95_lh_mm', 'sym_max_lh_mm',
               'sym_mean_rh_mm', 'sym_hd95_rh_mm', 'sym_max_rh_mm',
               'symcx_mean_mm', 'symcx_hd95_mm', 'symcx_max_mm',
               'symcx_mean_lh_mm', 'symcx_hd95_lh_mm', 'symcx_max_lh_mm',
               'symcx_mean_rh_mm', 'symcx_hd95_rh_mm', 'symcx_max_rh_mm',
               'p2s_mean_mm', 'p2s_hd95_mm', 'p2s_max_mm',
               'p2s_cx_mean_mm', 'p2s_cx_hd95_mm', 'p2s_cx_max_mm',
               'undeformed_mean_mm',
               'si_faces_pct_pre', 'repair_rounds', 'repair_max_move_mm', 'repair_seconds',
               't_push_s', 't_si_s', 't_dist_s', 't_repair_s')

# fsaverage-164k medial-wall masks (neuromaps copies of the fsaverage cortex
# definition; per-vertex 1 = cortex, 0 = medial wall).
CORTEX_LABEL_FILES = (
    'tpl-fsaverage_den-164k_hemi-L_desc-nomedialwall_dparc.label.gii',
    'tpl-fsaverage_den-164k_hemi-R_desc-nomedialwall_dparc.label.gii')


def load_cortex_mask(labels_dir):
    """Joined lh+rh cortex mask (True = cortical vertex). One fsaverage mask
    serves every mesh scored here: the subject surfaces are TopoFit outputs
    built with mri_surf2surf --trgsubject fsaverage (oasis_data/meshes/
    README.md), and the template npz is the 0406 pair repaired vertex-only —
    all share fsaverage ico7 indexing. Returns None (symcx_* -> NaN) when the
    label files are absent."""
    if labels_dir is None:
        return None
    masks = []
    for fname in CORTEX_LABEL_FILES:
        path = Path(labels_dir).expanduser() / fname
        if not path.is_file():
            print(f'[push] no cortex label {path} — symcx_* metrics NaN', flush=True)
            return None
        masks.append(nib.load(str(path)).darrays[0].data > 0)
    return np.concatenate(masks)


def sym_dist_cortex(mesh_w, gt_w, n_lh, cortex):
    """sym_dist with the medial wall dropped from each DIRECTED query: the
    template->GT direction keeps only cortex-labelled template vertices, the
    GT->template direction only cortex-labelled GT vertices; the target trees
    stay full surfaces. Same [(lh), (rh)] convention as sym_dist; the caller
    guarantees both meshes share the mask's indexing (gt_n_lh == n_lh)."""
    out = []
    for h in (slice(0, n_lh), slice(n_lh, None)):
        cx = cortex[h]
        out.append((cKDTree(gt_w[h]).query(mesh_w[h][cx])[0],
                    cKDTree(mesh_w[h]).query(gt_w[h][cx])[0]))
    return out


def load_onehot(path, target_size):
    """One-hot .npy -> (1,5,D,H,W) float tensor, nearest-resized like SegDataset."""
    seg = torch.tensor(np.load(path), dtype=torch.float32)
    return F.interpolate(seg.unsqueeze(0), size=target_size, mode='nearest')


@torch.no_grad()
def eval_pair(npz_path, verts_t, verts_np, faces, n_lh, gt, stn, dev, ref, args,
              cortex=None):
    """Integrate the stored velocities, push the template mesh, repair it when
    asked, score it. Returns (metric row, scored verts in normalized coords,
    dist for the figure, pre-repair verts or None)."""
    z = np.load(npz_path)
    vel_fw = torch.from_numpy(z['vel_fw']).float().unsqueeze(0).to(dev)
    vel_rv = torch.from_numpy(z['vel_rv']).float().unsqueeze(0).to(dev)
    affine = torch.from_numpy(z['affine_matrix']).float().unsqueeze(0).to(dev) \
        if 'affine_matrix' in z else None

    t0 = time.perf_counter()                    # after the npz read: compute only
    flow_fw = integrate_svf(vel_fw, stn)
    flow_rv = integrate_svf(vel_rv, stn)
    flow_inv = integrate_svf(-vel_fw, stn)      # exact SVF inverse: exp(-v)
    # svf_inverse_flow's parity guards its velocity RE-computation against
    # model.forward; here forward and inverse integrate the same stored tensor
    # through the same loop, so there is nothing independent to diff against.
    p = push_from_flows(verts_t, flow_fw, flow_rv, affine,
                        args.inv_iter, args.inv_alpha, flow_inv=flow_inv, parity=0.0)
    v = p['pushes'][args.push]
    w = norm_to_world(v, ref)
    row = {'inverse_residual_mm': float((p['inv_residual_norm'] * mm_per_norm(ref)).mean()),
           't_push_s': time.perf_counter() - t0}

    si, t_si, t_rep, v_pre = None, 0.0, 0.0, None
    if not args.no_self_int:
        t = time.perf_counter()
        si, _ = self_intersections(w, faces, npz_path.stem)
        t_si += time.perf_counter() - t
    if args.repair:
        t = time.perf_counter()
        w_rep, hist = repair(w, faces)
        t_rep = time.perf_counter() - t
        row.update({'si_faces_pct_pre': si['si_faces_pct'] if si else float('nan'),
                    'repair_rounds': sum(h['si_pairs'] > 0 for h in hist),
                    'repair_max_move_mm': float(np.linalg.norm(w_rep - w, axis=1).max()),
                    'repair_seconds': t_rep})
        v_pre, w = v, w_rep
        v = world_to_norm(w, ref)
        if not args.no_self_int:            # the deliverable is the repaired mesh
            t = time.perf_counter()
            si, _ = self_intersections(w, faces, f'{npz_path.stem} repaired')
            t_si += time.perf_counter() - t
    if si is not None:
        row.update({k: si[k] for k in SI_KEYS})
    row['flip_pct'] = triangle_flip_fraction(verts_np, v, faces)

    dist, t_v2v, t_p2s = None, 0.0, 0.0
    gt_v, gt_f, gt_n_lh = gt
    if gt_v is not None:
        gt_w = norm_to_world(gt_v, ref)
        shared_index = cortex is not None and gt_n_lh == n_lh and len(gt_w) == len(cortex)
        t = time.perf_counter()
        per_hemi = sym_dist(w, gt_w, n_lh, gt_n_lh)
        row.update(hemi_scores(per_hemi, 'sym_'))
        if shared_index:
            row.update(hemi_scores(sym_dist_cortex(w, gt_w, n_lh, cortex), 'symcx_'))
        und = sym_dist(norm_to_world(verts_np, ref), gt_w, n_lh, gt_n_lh)
        row['undeformed_mean_mm'] = hemi_scores(und, 'undeformed_')['undeformed_mean_mm']
        t_v2v = time.perf_counter() - t
        t = time.perf_counter()
        p2s = p2s_dist(w, faces, gt_w, gt_f, n_lh, gt_n_lh)
        row.update(hemi_scores(p2s, 'p2s_'))
        if shared_index:
            row.update(hemi_scores(cortex_only(p2s, cortex, n_lh), 'p2s_cx_'))
        t_p2s = time.perf_counter() - t
        dist = {'both': np.concatenate([d for h in per_hemi for d in h]),
                'init_mean': row['undeformed_mean_mm']}
    row.update({'t_si_s': t_si, 't_dist_s': t_v2v + t_p2s, 't_repair_s': t_rep})
    print(f"      [time] push {row['t_push_s']:.1f} s | self-int {t_si:.1f} s | "
          f"v2v {t_v2v:.1f} s | p2s {t_p2s:.1f} s | repair {t_rep:.1f} s", flush=True)
    return row, v, dist, v_pre


def main():
    ap = argparse.ArgumentParser(
        description='Push the template WM mesh through instance-optimized flows '
                    '(flows/*.npz from instance_opt_bandlimited.py) and score it')
    ap.add_argument('--probe_dir', required=True,
                    help='instance_opt_bandlimited output dir containing flows/*.npz '
                         '(a bare name resolves against the S-RegNET dir)')
    ap.add_argument('--config', default='config.yaml',
                    help='config.yaml (a bare name resolves against the S-RegNET dir)')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--arms', default=None,
                    help='comma list of arms to process, as named in flows/<subject>_<arm>.npz '
                         '(default: every arm found there)')
    ap.add_argument('--subjects', default=None,
                    help='comma list of subject dir names (default: all in flows/)')
    ap.add_argument('--push', choices=['svf_inv', 'numeric', 'rv', 'rv_noaffine'],
                    default='svf_inv',
                    help='svf_inv = exp(-v), the exact SVF inverse')
    ap.add_argument('--inv_iter', type=int, default=500)
    ap.add_argument('--inv_alpha', type=float, default=0.5)
    ap.add_argument('--cortex_labels_dir',
                    default=None,
                    help='dir with the fsaverage-164k aparc label GIFTIs for the '
                         'cortex-only (medial-wall-masked) symmetric distance; '
                         'omitted -> symcx_* NaN')
    ap.add_argument('--repair', action='store_true',
                    help='relax the self-intersecting patches of the pushed mesh '
                         '(repair_template_mesh.repair, vertices only) before scoring; '
                         'si_faces_pct_pre keeps the pre-repair count')
    ap.add_argument('--keep_prerepair', action='store_true',
                    help='with --repair, also save <subj>_<arm>_deformed_prerepair.white.surf')
    ap.add_argument('--no_self_int', action='store_true')
    ap.add_argument('--no_save_mesh', action='store_true')
    ap.add_argument('--dpi', type=int, default=140)
    args = ap.parse_args()

    # Bare names resolve against the S-RegNET dir, not the cwd.
    probe = Path(args.probe_dir).expanduser()
    if not probe.is_dir() and not probe.is_absolute():
        probe = _ROOT / probe
    flows_dir = probe / 'flows'
    if not flows_dir.is_dir():
        raise SystemExit(f'[push] no flows/ under {probe}')
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_file() and not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    args.config = str(cfg_path)

    cfg = load_config(args.config)
    d = cfg['data']
    dev = torch.device(args.device)
    target_size = tuple(cfg['model']['target_size'])
    stn = SpatialTransformer(size=target_size, device=dev).to(dev)

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    write_git_sha(out)

    # <subject>_<arm>.npz — subject names contain underscores, arms do not.
    avail = {}
    for f in sorted(flows_dir.glob('*.npz')):
        subj, _, arm = f.stem.rpartition('_')
        if subj and arm:
            avail.setdefault(subj, set()).add(arm)
    known = set().union(*avail.values()) if avail else set()
    arms = [a for a in args.arms.split(',') if a] if args.arms else sorted(known)
    unknown = [a for a in arms if a not in known]
    if unknown:
        raise SystemExit(f'[push] unknown arms {unknown} '
                         f'(found in flows/: {sorted(known)})')
    if args.subjects:
        wanted = args.subjects.split(',')
        missing = [s for s in wanted if s not in avail]
        if missing:
            raise SystemExit(f'[push] no flows for subjects: {missing}')
        subjects = wanted
    else:
        subjects = sorted(avail)
    if not subjects:
        raise SystemExit(f'[push] no <subject>_<arm>.npz files in {flows_dir}')

    print(f'[push] probe_dir = {probe}')
    print(f'[push] device = {args.device} | push = {args.push} | arms = {arms} | '
          f'output_dir = {out}', flush=True)

    verts_np, faces, n_lh = load_template_mesh(cfg, None)
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=dev)
    print(f'[push] template mesh {len(verts_np):,} verts ({n_lh:,} lh) | '
          f'{len(faces):,} faces | {len(subjects)} subjects', flush=True)

    cortex = load_cortex_mask(args.cortex_labels_dir)
    if cortex is not None and len(cortex) != len(verts_np):
        print(f'[push] cortex mask {len(cortex):,} verts != template '
              f'{len(verts_np):,} — symcx_* metrics NaN', flush=True)
        cortex = None
    elif cortex is not None:
        print(f'[push] cortex mask: {int(cortex.sum()):,}/{len(cortex):,} '
              f'cortical verts', flush=True)

    template_si = None
    if not args.no_self_int:
        # The floor the warp adds to — 0 for the repaired template.
        tpl_ref = nib.load(ref_for(d['template_seg_path']))
        template_si = self_intersections(norm_to_world(verts_np, tpl_ref), faces,
                                         'template')[0]
        print(f"[push] template self-int {template_si['si_faces']} faces "
              f"({template_si['si_faces_pct']:.4f}%)", flush=True)

    # Subject-dir lookup: val_txt lines -> parent dirs.
    by_name = {Path(p).parent.name: Path(p).parent
               for p in Path(d['val_txt']).read_text().splitlines() if p.strip()}

    fieldnames = ['subject', 'arm', *METRIC_KEYS]
    rows = []
    with open(out / 'mesh_metrics.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for subj in subjects:
            subject_dir = by_name.get(subj)
            if subject_dir is None:
                print(f"[push] skip {subj}: not in {d['val_txt']}", flush=True)
                continue
            ref_path = ref_for(subject_dir / d['seg_filename'])
            if not Path(ref_path).is_file():
                print(f'[push] skip {subj}: no {ref_path}', flush=True)
                continue
            ref = nib.load(ref_path)
            gt = load_subject_white(subject_dir, ref)   # (None,)*3 tolerated
            if gt[0] is None:
                print(f'[push] {subj}: no GT white surface — distances NaN', flush=True)
            gt_wm = (load_onehot(subject_dir / d['seg_filename'], target_size)
                     .argmax(1)[0].numpy() == WM_LABEL)
            slices = centroid_slices(gt_wm)

            for arm in (a for a in arms if a in avail[subj]):
                npz_path = flows_dir / f'{subj}_{arm}.npz'
                met, v, dist, v_pre = eval_pair(npz_path, verts_t, verts_np, faces, n_lh,
                                                gt, stn, dev, ref, args, cortex)

                curves = [((verts_np, faces), 'gold', 1.2)]
                if gt[0] is not None:
                    curves.append(((gt[0], gt[1]), 'limegreen', 1.2))
                curves.append(((v, faces), 'deepskyblue', 1.7))
                render_sample(f'{subj} {arm}', target_size[0], gt_wm.astype(np.float32),
                              curves, slices, dist, met['flip_pct'],
                              met['inverse_residual_mm'], args.push,
                              out / f'{subj}_{arm}_mesh.png', args.dpi)
                if not args.no_save_mesh:
                    save_deformed_surf(v, faces, ref,
                                       out / f'{subj}_{arm}_deformed.white.surf')
                    if v_pre is not None and args.keep_prerepair:
                        save_deformed_surf(v_pre, faces, ref,
                                           out / f'{subj}_{arm}_deformed_prerepair.white.surf')

                row = {'subject': subj, 'arm': arm,
                       **{k: met.get(k, float('nan')) for k in METRIC_KEYS}}
                writer.writerow(row)
                fh.flush()                          # partial results survive a kill
                rows.append(row)

                msg = (f"[push] {subj:>20s} {arm:<9s} flip {row['flip_pct']:.4f}% | "
                       f"inv res {row['inverse_residual_mm']:.3f} mm")
                if not np.isnan(row['sym_mean_mm']):
                    msg += (f" | sym {row['sym_mean_mm']:5.2f} mm | hd95 "
                            f"{row['sym_hd95_mm']:5.2f} | undef "
                            f"{row['undeformed_mean_mm']:5.2f}")
                if not np.isnan(row['symcx_mean_mm']):
                    msg += f" | symcx {row['symcx_mean_mm']:5.2f} mm"
                if not np.isnan(row['p2s_mean_mm']):
                    msg += f" | p2s {row['p2s_mean_mm']:5.2f} mm / hd95 {row['p2s_hd95_mm']:5.2f}"
                if not args.no_self_int:
                    msg += f" | self-int {row['si_faces_pct']:.4f}%"
                if args.repair:
                    msg += (f" (pre {row['si_faces_pct_pre']:.4f}%) | repair "
                            f"{row['repair_rounds']} rounds, max move "
                            f"{row['repair_max_move_mm']:.2f} mm")
                print(msg, flush=True)
            if 'cuda' in args.device:
                torch.cuda.empty_cache()

    if not rows:
        raise SystemExit('[push] no subject x arm pairs evaluated')

    summary = {'args': vars(args), 'probe_dir': str(probe),
               'template_mesh': d.get('template_wm_mesh_path'),
               'template_self_int': template_si, 'n_rows': len(rows),
               'per_arm': {arm: summarize([r for r in rows if r['arm'] == arm],
                                          f'arm={arm}')
                           for arm in arms},
               'per_subject_arm': rows}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f"\n[push] per-pair CSV -> {out / 'mesh_metrics.csv'}")
    print(f"[push] summary      -> {out / 'summary.json'}")


if __name__ == '__main__':
    main()
