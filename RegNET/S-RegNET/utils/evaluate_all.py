"""
Numbers-only evaluation over EVERY subject: per-subject voxel + mesh metrics to
CSV, then means/std/min/max. No figures.

The template subject is excluded automatically (it is the moving image — warping
it onto itself is not a data point). Train and val are reported as separate
blocks as well as pooled, because train-subject numbers are not generalization
numbers and pooling them silently would flatter the model.

Per subject:
  voxel  dice_c0..c4, dice_fg_mean, dice_wm, folding_pct, min_det, cycle,
         flow_mag_mean/max, lambda_mean            (the numbers training optimizes)
  mesh   sym_{mean,hd95,max}_mm and their _lh_/_rh_ parts, mesh_to_gt_mm,
         gt_to_mesh_mm, undeformed_{mean,hd95,max}_mm, flip_pct,
         flip_pct_dense_only, svf_parity_max, inverse_residual_{mm,max_mm},
         inverse_over_0p2mm_pct
         (genus-0 template mesh pushed to the subject vs its own FreeSurfer
          white surface; skipped per subject when the surfaces are missing)
         lh is matched against lh and rh against rh, and the headline
         sym_{mean,hd95,max}_mm scores the POOLED lh+rh distances (a percentile
         of the union, not the average of two percentiles — the average is not a
         percentile of anything, and averaging two maxima is not a maximum). A
         single KD-tree over the joined mesh would let a medial vertex match the
         opposite hemisphere (their walls sit ~1-3 mm apart) and flatter the tail;
         the *_hemi_blind columns are that old joined match, kept to size the bias.
         hd95 is the default name; --hd_percentile P renames the column to hdP.
         undeformed_* is symmetric like the deformed metric, so before/after are
         now like-for-like — it used to be one-directional and thus not comparable.
  --self_int adds si_faces{,_pct}, si_pairs, si_clusters, si_largest — the
         topology deliverable. The un-pushed template is scored once into
         metrics_summary.json:template_self_int, because the warp only ADDS to
         whatever the template already had (raw 0406 ?h.white: 0.1007%; the
         repaired template from utils/repair_template_mesh.py: 0).

The default push is exp(-v), the exact inverse of an SVF (--push svf_inv). The
inverse_* columns describe the OLD fixed-point inverse and are kept as a
reference: it stalls (residual does not shrink with --inv_iter) wherever the
field is not contractive, and its per-vertex noise creates triangle flips of its
own on a mesh with ~1 mm triangles. svf_parity_max must stay ~1e-6 — it is the
guard that our velocity replication still matches model.py.

One model forward per subject drives both halves. Mesh metrics dominate the
runtime (a 500-iteration fixed-point inverse plus two KD-tree queries over all
vertices) — pass --no_mesh for a fast voxel-only sweep. Rows are appended to the
CSV as they are computed, so a long run that dies partway keeps its results.

Run from the S-RegNET directory:
    python evaluate_all.py \\
        --model ~/shared_scratch/training_results/mahith_experiment/20260801_083414 \\
        --device cuda:4
    # --split val | train | both   (default both)
    # --limit 5                    smoke-test on the first few subjects
    # --push numeric | rv | rv_noaffine    which mesh push to score (default numeric)
    # --no_mesh                    voxel metrics only (fast)
    # --self_int                   self-intersecting faces of the pushed mesh
    # --template_surf <lh> <rh>    override the template mesh (e.g. the repaired one)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from scipy.spatial import cKDTree

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))          # repo-root modules, whatever the cwd is
sys.path.insert(0, str(_ROOT / 'utils'))

from git_provenance import write_git_sha
from get_data import SegDataset
from inference import load_config, setup_inference
from losses import compute_dice_score, cycle_consistency_loss
from visualize_run import resolve_checkpoint, detect_affine, warp_template, folding_stats
from visualize_mesh import (WM_LABEL, load_template_mesh, load_subject_white, mm_per_norm,
                            norm_to_world, push_from_flows, ref_for, svf_inverse_flow,
                            triangle_flip_fraction)
from mesh_flip_probe import self_intersections


def svf_inverse(ctx, sample_seg):
    """Exact SVF inverse field exp(-v) for this subject, plus the parity check
    that our velocity replication still matches model.py."""
    return svf_inverse_flow(ctx['model'], ctx['template_seg'], sample_seg)


def voxel_metrics(ctx, sample_seg):
    """Replay training's warp and score it. Returns (row dict, flows, affine)."""
    warped, flow_fw, flow_rv, lambda_map, affine = warp_template(ctx, sample_seg)
    dice_pc, fg = compute_dice_score(warped, sample_seg, ctx['num_classes'])
    fold, min_det = folding_stats(flow_fw)
    mag = flow_fw.pow(2).sum(1).sqrt()
    row = {f'dice_c{c}': dice_pc[c] for c in range(len(dice_pc))}
    row.update({
        'dice_fg_mean': fg,
        'dice_wm': dice_pc[WM_LABEL],
        'folding_pct': fold,
        'min_det': min_det,
        'cycle': cycle_consistency_loss(flow_fw, flow_rv, ctx['stn']).item(),
        'flow_mag_mean': mag.mean().item(),
        'flow_mag_max': mag.max().item(),
        'lambda_mean': lambda_map.mean().item(),
    })
    return row, (flow_fw, flow_rv), affine


def sym_dist(mesh_w, gt_w, n_lh, gt_n_lh):
    """Per-hemisphere symmetric nearest-neighbour distances in mm: returns
    [(m2g_lh, g2m_lh), (m2g_rh, g2m_rh)].

    lh and rh are two separate closed surfaces whose medial walls face each other
    across the interhemispheric fissure, ~1-3 mm apart. One KD-tree over the joined
    point cloud therefore lets a medial lh vertex match an rh triangle and report a
    small distance for a vertex that landed on the wrong side of the brain — an
    optimistic bias concentrated exactly in the tail the HD95 reads. lh is matched
    against lh and rh against rh; hemi_scores pools the four arrays for the
    headline number and keeps the per-hemisphere ones alongside."""
    out = []
    for m, g in ((slice(0, n_lh), slice(0, gt_n_lh)),
                 (slice(n_lh, None), slice(gt_n_lh, None))):
        out.append((cKDTree(gt_w[g]).query(mesh_w[m])[0],
                    cKDTree(mesh_w[m]).query(gt_w[g])[0]))
    return out


def faces_of_vertex(faces, n_v):
    """(n_v, max_degree) incident-face ids per vertex, short rows padded by
    repeating the vertex's first face (a duplicate candidate costs nothing)."""
    vi, fi = faces.ravel(), np.repeat(np.arange(len(faces)), 3)
    order = np.argsort(vi, kind='stable')
    vi, fi = vi[order], fi[order]
    cnt = np.bincount(vi, minlength=n_v)
    ptr = np.concatenate([[0], np.cumsum(cnt)])
    out = np.repeat(fi[np.minimum(ptr[:-1], len(fi) - 1)][:, None], int(cnt.max()), 1)
    rank = np.arange(len(vi)) - np.repeat(ptr[:-1], cnt)
    out[vi, rank] = fi
    return out


def closest_on_triangles(p, tri):
    """Distance from p[i] to triangle tri[i] (Ericson, Real-Time Collision
    Detection 5.1.5): classify p against the six Voronoi regions of the
    triangle's vertices and edges, else project into the face."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, bc = b - a, c - a, c - b
    d1, d2 = (ab * (p - a)).sum(1), (ac * (p - a)).sum(1)
    d3, d4 = (ab * (p - b)).sum(1), (ac * (p - b)).sum(1)
    d5, d6 = (ab * (p - c)).sum(1), (ac * (p - c)).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide='ignore', invalid='ignore'):
        v, w = vb / (va + vb + vc), vc / (va + vb + vc)
        inside = a + ab * v[:, None] + ac * w[:, None]
        t1 = np.nan_to_num(np.clip(d1 / (d1 - d3), 0, 1))
        t2 = np.nan_to_num(np.clip(d2 / (d2 - d6), 0, 1))
        t3 = np.nan_to_num(np.clip((d4 - d3) / ((d4 - d3) + (d5 - d6)), 0, 1))
    q = np.where(np.isfinite(inside).all(1)[:, None], inside, a)
    for m, pt in ((((vc <= 0) & (d1 >= 0) & (d3 <= 0)), a + ab * t1[:, None]),
                  (((vb <= 0) & (d2 >= 0) & (d6 <= 0)), a + ac * t2[:, None]),
                  (((va <= 0) & (d4 >= d3) & (d5 >= d6)), b + bc * t3[:, None]),
                  (((d1 <= 0) & (d2 <= 0)), a),
                  (((d3 >= 0) & (d4 <= d3)), b),
                  (((d6 >= 0) & (d5 <= d6)), c)):
        q = np.where(m[:, None], pt, q)
    return np.linalg.norm(p - q, axis=1)


def point_to_surface(q, tv, tf, k=4, chunk=40_000):
    """Exact point-to-triangle distance from each q to the surface (tv, tf),
    searched over the incident faces of the k nearest target vertices. Every
    triangle within reach of a query point has a vertex among its nearest few
    at this vertex density, so the 1-ring union is the whole candidate set."""
    cand = faces_of_vertex(tf, len(tv))[cKDTree(tv).query(q, k=k)[1]]
    cand = cand.reshape(len(q), -1)
    d = np.empty(len(q))
    for s in range(0, len(q), chunk):
        c = cand[s:s + chunk]
        d[s:s + chunk] = closest_on_triangles(np.repeat(q[s:s + chunk], c.shape[1], 0),
                                              tv[tf[c.ravel()]]).reshape(c.shape).min(1)
    return d


def hemi_faces(faces, lo, hi):
    """The faces of one hemisphere block [lo, hi), re-indexed to that block —
    no face spans the two hemispheres."""
    keep = (faces[:, 0] >= lo) & (faces[:, 0] < hi)
    return faces[keep] - lo


def p2s_dist(mesh_w, mesh_f, gt_w, gt_f, n_lh, gt_n_lh):
    """sym_dist with point-to-TRIANGLE distances: a vertex that lands on a face
    between the other surface's vertices scores its true gap to the surface, not
    the gap to the nearest vertex (v2v overstates the mean by ~0.15 mm at this
    vertex spacing, the HD95 by ~0.02). Same [(m2g_lh, g2m_lh), (m2g_rh, g2m_rh)]
    layout, lh vs lh and rh vs rh."""
    out = []
    for (m_lo, m_hi), (g_lo, g_hi) in (((0, n_lh), (0, gt_n_lh)),
                                       ((n_lh, len(mesh_w)), (gt_n_lh, len(gt_w)))):
        out.append((point_to_surface(mesh_w[m_lo:m_hi], gt_w[g_lo:g_hi],
                                     hemi_faces(gt_f, g_lo, g_hi)),
                    point_to_surface(gt_w[g_lo:g_hi], mesh_w[m_lo:m_hi],
                                     hemi_faces(mesh_f, m_lo, m_hi))))
    return out


def cortex_only(per_hemi, cortex, n_lh):
    """Drop the medial wall from per-hemisphere directed distances: keep only
    cortex-labelled QUERY vertices in each direction (both meshes share the
    mask's indexing), the same selection push_optimized_mesh.sym_dist_cortex
    makes — but as a mask on distances already computed against the whole
    target surface, so it costs nothing."""
    return [(m2g[cortex[h]], g2m[cortex[h]])
            for h, (m2g, g2m) in zip((slice(0, n_lh), slice(n_lh, None)), per_hemi)]


def hemi_scores(per_hemi, prefix='', percentile=95):
    """lh and rh scored separately, then pooled for the combined score.
    percentile controls the tail metric: 95 -> HD95, 99 -> HD99, etc."""
    p = int(percentile)
    row = {}
    all_both = []
    for tag, (m2g, g2m) in zip(('lh', 'rh'), per_hemi):
        both = np.concatenate([m2g, g2m])
        all_both.append(both)
        row[f'{prefix}mean_{tag}_mm'] = float(both.mean())
        row[f'{prefix}hd{p}_{tag}_mm'] = float(np.percentile(both, p))
        row[f'{prefix}max_{tag}_mm'] = float(both.max())

    combined = np.concatenate(all_both)
    row[f'{prefix}mean_mm'] = float(combined.mean())
    row[f'{prefix}hd{p}_mm'] = float(np.percentile(combined, p))
    row[f'{prefix}max_mm'] = float(combined.max())
    return row


def mesh_metrics(verts_t, verts_np, faces, n_lh, flows, affine, subject_dir, seg_filename,
                 push, n_iter=500, alpha=0.5, flow_inv=None, parity=None, all_pushes=False,
                 self_int=False, percentile=95):
    """Push the template mesh with the already-computed flows and score it against
    the subject's own white surface. Returns {} when the subject has no surfaces."""
    ref_path = ref_for(Path(subject_dir) / seg_filename)
    if not Path(ref_path).is_file():
        return {}
    ref = nib.load(ref_path)
    gt_v, _, gt_n_lh = load_subject_white(subject_dir, ref)
    if gt_v is None:
        return {}

    p = push_from_flows(verts_t, flows[0], flows[1], affine, n_iter, alpha,
                        flow_inv=flow_inv, parity=parity)
    v = p['pushes'][push]
    mm = mm_per_norm(ref)
    res_mm = p['inv_residual_norm'] * mm

    gt_w, w = norm_to_world(gt_v, ref), norm_to_world(v, ref)
    per_hemi = sym_dist(w, gt_w, n_lh, gt_n_lh)
    m2g = np.concatenate([h[0] for h in per_hemi])
    g2m = np.concatenate([h[1] for h in per_hemi])
    # The old hemisphere-blind match, kept as a column so the size of the bias it
    # introduced is on the record rather than argued about.
    tree = cKDTree(gt_w)
    joined = np.concatenate([tree.query(w)[0], cKDTree(w).query(gt_w)[0]])

    extra = {'svf_parity_max': p['svf_parity']}
    if self_int:
        # The topology deliverable: does the pushed surface pass through itself?
        # Scored in world mm so the broadphase cell size is the same scale as the
        # template baseline printed once in main().
        si, _ = self_intersections(w, faces, 'pushed')
        extra.update({k: si[k] for k in ('si_faces', 'si_faces_pct', 'si_pairs',
                                         'si_clusters', 'si_largest')})
    if all_pushes:                                # diagnostic: score every mode
        for mode, mv in p['pushes'].items():
            if mode == push:
                continue
            mw = norm_to_world(mv, ref)
            mb = np.concatenate([tree.query(mw)[0], cKDTree(mw).query(gt_w)[0]])
            extra[f'{mode}_sym_mean_mm'] = float(mb.mean())
            extra[f'{mode}_flip_pct'] = triangle_flip_fraction(verts_np, mv, faces)
    und = sym_dist(norm_to_world(verts_np, ref), gt_w, n_lh, gt_n_lh)
    pct = int(percentile)
    return {**extra,
        # sym_{mean,hdP,max}_mm = pooled lh+rh scores; per-hemisphere numbers kept alongside.
        **hemi_scores(per_hemi, 'sym_', percentile=pct),
        **hemi_scores(und, 'undeformed_', percentile=pct),
        'sym_mean_mm_hemi_blind': float(joined.mean()),
        f'sym_hd{pct}_mm_hemi_blind': float(np.percentile(joined, pct)),
        'mesh_to_gt_mm': float(m2g.mean()),
        'gt_to_mesh_mm': float(g2m.mean()),
        # flip vs the raw template includes the affine; vs the aligned verts
        # isolates the dense inverse. A near-identity affine makes them equal —
        # if they differ, the affine itself is reversing triangles.
        'flip_pct': triangle_flip_fraction(verts_np, v, faces),
        'flip_pct_dense_only': triangle_flip_fraction(p['v_aligned'], v, faces),
        # Solver noise, not just a convergence stat: the mesh has ~1 mm triangles,
        # so residual of this size can create flips by itself.
        'inverse_residual_mm': float(res_mm.mean()),
        'inverse_residual_max_mm': float(res_mm.max()),
        'inverse_over_0p2mm_pct': float((res_mm > 0.2).mean() * 100),
    }


def summarize(rows, label):
    """mean / std / min / max for every numeric column present in rows."""
    if not rows:
        return {}
    cols = [k for k in rows[0] if k != 'idx' and isinstance(rows[0][k], (int, float))
            and not isinstance(rows[0][k], bool)]
    out = {}
    for c in cols:
        vals = np.array([r[c] for r in rows if c in r and r[c] is not None], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            out[c] = {'mean': float(vals.mean()), 'std': float(vals.std()),
                      'min': float(vals.min()), 'max': float(vals.max()), 'n': int(len(vals))}
    print(f"\n=== {label} (n={len(rows)}) ===")
    print(f"{'metric':<22s} {'mean':>10s} {'std':>9s} {'min':>10s} {'max':>10s} {'n':>5s}")
    for c, s in out.items():
        print(f"{c:<22s} {s['mean']:10.4f} {s['std']:9.4f} {s['min']:10.4f} "
              f"{s['max']:10.4f} {s['n']:5d}")
    return out


def main():
    ap = argparse.ArgumentParser(description='Per-subject metrics over all samples, no figures')
    ap.add_argument('--model', required=True,
                    help='run dir, a best_model.pth path, or a run name under output.base_dir')
    ap.add_argument('--config', default=None, help='config.yaml (default: ./config.yaml)')
    ap.add_argument('--output_dir', default=None, help='default <run_dir>/metrics')
    ap.add_argument('--split', choices=['val', 'train', 'both'], default='val',
                    help='default val — the generalization numbers; train/both on request')
    ap.add_argument('--limit', type=int, default=None, help='first N subjects per split (smoke test)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--affine', choices=['auto', 'on', 'off'], default='auto')
    ap.add_argument('--push', choices=['svf_inv', 'numeric', 'rv', 'rv_noaffine'],
                    default='svf_inv',
                    help='svf_inv = exp(-v), the exact SVF inverse (no fixed point)')
    ap.add_argument('--all_pushes', action='store_true',
                    help='also score every other push mode per subject (diagnostic; '
                         'adds a KD-tree pair per mode, use with --limit)')
    ap.add_argument('--inv_iter', type=int, default=500,
                    help='fixed-point iterations for the numeric inverse. Raise it and '
                         'compare flip_pct: if flip falls with the residual, the flips '
                         'were solver noise, not the model.')
    ap.add_argument('--inv_alpha', type=float, default=0.5, help='fixed-point damping')
    ap.add_argument('--no_mesh', action='store_true', help='voxel metrics only (much faster)')
    ap.add_argument('--self_int', action='store_true',
                    help='also count self-intersecting faces of the pushed mesh (the '
                         'topology deliverable). ~10-20 s/subject on a 655k-face mesh; '
                         'the un-pushed template baseline is scored once for subtraction')
    ap.add_argument('--hd_percentile', type=int, default=95, choices=range(90, 100),
                    metavar='P', help='percentile for the Hausdorff-distance metric '
                         '(90-99 inclusive; default 95 -> HD95)')
    ap.add_argument('--template_surf', nargs=2, metavar=('LH', 'RH'), default=None)
    args = ap.parse_args()

    # A config name is relative to the S-RegNET dir, not to wherever this was
    # invoked from, so `python utils/evaluate_all.py` and `cd utils && python
    # evaluate_all.py` resolve the same file.
    cfg_path = Path(args.config or 'config.yaml').expanduser()
    if not cfg_path.is_file() and not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    args.config = str(cfg_path)

    cfg = load_config(args.config)
    ckpt = resolve_checkpoint(args.model, cfg)
    use_affine = detect_affine(ckpt) if args.affine == 'auto' else args.affine == 'on'
    out = Path(args.output_dir).expanduser() if args.output_dir \
        else Path(ckpt).parent.parent / 'metrics'
    try:
        out.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # The default lands next to the checkpoint, which is often someone else's
        # run dir on a shared box. Say so instead of dying on a bare errno.
        raise SystemExit(f"[eval] cannot write {out} — the run dir belongs to another "
                         f"user. Pass --output_dir <a dir you own>.")
    write_git_sha(out)

    print(f"[eval] checkpoint = {ckpt}")
    print(f"[eval] device = {args.device} | affine = {use_affine} | mesh = {not args.no_mesh} | "
          f"push = {args.push} | output_dir = {out}", flush=True)

    ctx = setup_inference(ckpt, args.config, args.device, use_affine=use_affine, verbose=True)
    d = cfg['data']
    template_subject = Path(d['template_seg_path']).parent.name

    verts_t = verts_np = faces = n_lh = None
    template_si = None
    if not args.no_mesh:
        verts_np, faces, n_lh = load_template_mesh(cfg, args.template_surf)
        verts_t = torch.tensor(verts_np, dtype=torch.float32, device=ctx['device'])
        print(f"[eval] template mesh {len(verts_np):,} verts ({n_lh:,} lh) | "
              f"{len(faces):,} faces", flush=True)
        if args.self_int:
            # Same for every subject, so score it once — it is the floor the warp
            # adds to, and 0 here is what the repaired template is supposed to give.
            tpl_ref = nib.load(ref_for(d['template_seg_path']))
            template_si = self_intersections(norm_to_world(verts_np, tpl_ref), faces,
                                            f'template {template_subject}')[0]
            print(f"[eval] template self-int {template_si['si_faces']} faces "
                  f"({template_si['si_faces_pct']:.4f}%), {template_si['si_clusters']} patches",
                  flush=True)

    splits = {'val': d['val_txt'], 'train': d['train_txt']}
    if args.split != 'both':
        splits = {args.split: splits[args.split]}

    csv_path = out / 'metrics_per_subject.csv'
    rows, writer, fh, seen, skipped = [], None, None, set(), 0
    try:
        for split, txt in splits.items():
            ds = SegDataset(txt, d['template_seg_path'], target_size=ctx['target_size'],
                            seg_filename=d['seg_filename'], preload=False)
            n = len(ds) if args.limit is None else min(args.limit, len(ds))
            print(f"\n[eval] split={split}: {len(ds)} subjects, evaluating {n}", flush=True)

            for i in range(n):
                subject_dir = ds.subject_dirs[i]
                name = Path(subject_dir).name
                if name == template_subject or name in seen:
                    skipped += 1
                    print(f"[eval] skip {name} "
                          f"({'template subject' if name == template_subject else 'duplicate'})",
                          flush=True)
                    continue
                seen.add(name)

                sample_seg = ds[i]['sample_seg'].unsqueeze(0).to(ctx['device'])
                row = {'subject': name, 'split': split, 'idx': i}
                vox, flows, affine = voxel_metrics(ctx, sample_seg)
                row.update(vox)
                if not args.no_mesh:
                    flow_inv, _aff, parity = svf_inverse(ctx, sample_seg)
                    row.update(mesh_metrics(verts_t, verts_np, faces, n_lh, flows, affine,
                                            subject_dir, d['seg_filename'], args.push,
                                            args.inv_iter, args.inv_alpha,
                                            flow_inv=flow_inv, parity=parity,
                                            all_pushes=args.all_pushes,
                                            self_int=args.self_int,
                                            percentile=args.hd_percentile))

                if fh is None:                          # header from the first row
                    fh = open(csv_path, 'w', newline='')
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                assert writer is not None
                dropped = [k for k in row if k not in writer.fieldnames]
                if dropped:      # first row set the header; don't lose columns silently
                    print(f"[eval] WARNING {name}: columns absent from the CSV header "
                          f"(first subject lacked them): {dropped}", flush=True)
                writer.writerow({k: row.get(k) for k in writer.fieldnames})
                fh.flush()                              # partial results survive a crash
                rows.append(row)

                p = args.hd_percentile
                msg = (f"[eval] {len(rows):4d} {name:>20s} [{split}] WM {row['dice_wm']:.4f} | "
                       f"fg {row['dice_fg_mean']:.4f} | fold {row['folding_pct']:.4f}%")
                if 'sym_mean_mm' in row:
                    msg += (f" | mesh {row['sym_mean_mm']:5.2f} mm | hd{p} "
                            f"{row[f'sym_hd{p}_mm']:5.2f} (lh {row[f'sym_hd{p}_lh_mm']:5.2f} / "
                            f"rh {row[f'sym_hd{p}_rh_mm']:5.2f}) | undef "
                            f"{row['undeformed_mean_mm']:5.2f}")
                    if 'si_faces_pct' in row:
                        msg += f" | self-int {row['si_faces_pct']:.4f}%"
                elif not args.no_mesh:
                    msg += " | mesh: no GT surface"
                print(msg, flush=True)
    finally:
        if fh is not None:
            fh.close()

    if not rows:
        raise SystemExit("no subjects evaluated")

    summary = {'checkpoint': ckpt, 'affine': use_affine, 'push': args.push,
               'mesh': not args.no_mesh, 'template_subject_excluded': template_subject,
               'template_mesh': args.template_surf or cfg['data'].get('template_wm_mesh_path'),
               'template_self_int': template_si,
               'n_subjects': len(rows), 'n_skipped': skipped}
    for split in splits:
        sub = [r for r in rows if r['split'] == split]
        if sub:
            summary[split] = summarize(sub, f'split={split}')
    if len(splits) > 1:
        summary['pooled'] = summarize(rows, 'pooled train+val')

    n_mesh = sum('sym_mean_mm' in r for r in rows)
    if not args.no_mesh:
        print(f"\n[eval] mesh metrics available for {n_mesh}/{len(rows)} subjects "
              f"(rest had no FreeSurfer white surface)")
    (out / 'metrics_summary.json').write_text(json.dumps(summary, indent=2))
    print(f"[eval] per-subject CSV -> {csv_path}")
    print(f"[eval] summary        -> {out / 'metrics_summary.json'}")


if __name__ == '__main__':
    main()
