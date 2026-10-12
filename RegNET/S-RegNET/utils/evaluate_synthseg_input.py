"""
SynthSeg-INPUT evaluation — the deployment condition.

Every eval so far conditions the flow on the subject's GT seg. At deployment
the only seg available is SynthSeg's, so this script swaps the input and
measures the gap:

    flow   = model(GT template seg, SynthSeg sample seg)     <- input swap
    warped = affine-then-flow(GT template seg)               <- same replay as training
    Dice(warped, GT sample seg)                              <- scored against GT

The mesh half pushes the GT template WM mesh through the same flow (exp(-v)
by default) and scores self-intersection (the topology deliverable), triangle
flips, and mm distance to the subject's own FreeSurfer white surface.

Two reference Dice numbers put the headline in context per subject:
    dice_wm_synthseg    warped vs the SynthSeg input — the objective the flow
                        actually optimized; the headline can't beat this by much
    synthseg_wm_vs_gt   SynthSeg input vs GT — the input's own noise floor

Outputs (into --output_dir):
    <subj>_seg.png       3 ortho slices: GT WM + SynthSeg / GT / warped contours
    <subj>_mesh.png      pushed template mesh vs subject white surface + histogram
    <subj>_deformed.white.surf   open in freeview against the subject volume
    metrics_per_subject.csv, summary.json, GIT_SHA.txt

Run from the S-RegNET directory:
    python utils/evaluate_synthseg_input.py \\
        --model <run_dir> --config config.yaml \\
        --num_samples 10 --device cuda:4 \\
        --output_dir <scratch>/synthseg_input_eval
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))          # repo-root modules, whatever the cwd is
sys.path.insert(0, str(_ROOT / 'utils'))

from git_provenance import write_git_sha
from inference import load_config, setup_inference
from losses import compute_dice_score
from visualize_run import (resolve_checkpoint, detect_affine, warp_template,
                           folding_stats, centroid_slices)
from visualize_mesh import (WM_LABEL, load_template_mesh, load_subject_white,
                            mm_per_norm, norm_to_world, push_from_flows, ref_for,
                            svf_inverse_flow, triangle_flip_fraction, render_sample,
                            save_deformed_surf)
from evaluate_all import sym_dist, hemi_scores, summarize
from mesh_flip_probe import self_intersections

SYNTH_C, GT_C, WARP_C = '#FFC000', '#32CD32', '#00BFFF'
AXIS_NAMES = ['axial (z)', 'coronal (y)', 'sagittal (x)']


def load_onehot(path, target_size):
    """One-hot .npy -> (1,5,D,H,W) float tensor, nearest-resized like SegDataset."""
    seg = torch.tensor(np.load(path), dtype=torch.float32)
    return F.interpolate(seg.unsqueeze(0), size=target_size, mode='nearest')


def render_seg(name, synth_wm, gt_wm, warped_wm, slices, row, output, dpi=140):
    """3 ortho slices at the GT WM centroid: GT WM mask as background, WM
    contours of the SynthSeg input, the GT, and the warped template on top."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#f5f5f0')
    for a, ax in enumerate(axes):
        s = int(slices[a])
        ax.set_facecolor('black')
        ax.imshow(np.take(gt_wm, s, axis=a), cmap='gray', interpolation='none',
                  vmin=0, vmax=1)
        for m, col in ((synth_wm, SYNTH_C), (gt_wm, GT_C), (warped_wm, WARP_C)):
            m2d = np.take(m, s, axis=a).astype(float)
            if m2d.any():
                ax.contour(m2d, levels=[0.5], colors=[col], linewidths=1.4)
        ax.set_title(f'{AXIS_NAMES[a]} @ {s}')
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend([Line2D([0], [0], color=c, lw=2) for c in (SYNTH_C, GT_C, WARP_C)],
                   ['SynthSeg WM (model input)', 'GT WM (scoring target)',
                    'warped template WM'],
                   loc='lower right', facecolor='gray', edgecolor='black', framealpha=0.9)
    fig.suptitle(f"{name} — flow from SynthSeg input   "
                 f"[WM Dice vs GT {row['dice_wm_gt']:.4f} | vs SynthSeg "
                 f"{row['dice_wm_synthseg']:.4f} | SynthSeg-vs-GT floor "
                 f"{row['synthseg_wm_vs_gt']:.4f} | fold {row['folding_pct']:.4f}%]",
                 fontsize=11, fontweight='bold')
    plt.subplots_adjust(wspace=0.05)
    fig.savefig(output, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


@torch.no_grad()
def mesh_eval(ctx, args, verts_t, verts_np, faces, n_lh, flows, affine, synth_seg,
              subject_dir, gt_wm_mask, name, out):
    """Push the template mesh through the SynthSeg-conditioned flow, score it,
    and draw the mesh figure. Returns the metric columns (may be partial when
    the subject has no white surface)."""
    ref_path = ref_for(subject_dir / ctx['cfg']['data']['seg_filename'])
    if not Path(ref_path).is_file():
        print(f"[synth] {name}: no {ref_path} — mesh skipped", flush=True)
        return {}
    ref = nib.load(ref_path)

    flow_inv, _aff, parity = svf_inverse_flow(ctx['model'], ctx['template_seg'],
                                              synth_seg)
    p = push_from_flows(verts_t, flows[0], flows[1], affine,
                        args.inv_iter, args.inv_alpha, flow_inv=flow_inv, parity=parity)
    v = p['pushes'][args.push]
    w = norm_to_world(v, ref)
    res_mm = float((p['inv_residual_norm'] * mm_per_norm(ref)).mean())

    row = {'flip_pct': triangle_flip_fraction(verts_np, v, faces),
           'svf_parity_max': parity}
    if not args.no_self_int:
        si, _ = self_intersections(w, faces, name)
        row.update({k: si[k] for k in ('si_faces', 'si_faces_pct', 'si_pairs',
                                       'si_clusters', 'si_largest')})

    dist, gt_pair = None, None
    gt_v, gt_f, gt_n_lh = load_subject_white(subject_dir, ref)
    if gt_v is not None:
        gt_pair = (gt_v, gt_f)
        gt_w = norm_to_world(gt_v, ref)
        per_hemi = sym_dist(w, gt_w, n_lh, gt_n_lh)
        row.update(hemi_scores(per_hemi, 'sym_'))
        und = sym_dist(norm_to_world(verts_np, ref), gt_w, n_lh, gt_n_lh)
        row['undeformed_mean_mm'] = hemi_scores(und, 'undeformed_')['undeformed_mean_mm']
        dist = {'both': np.concatenate([d for h in per_hemi for d in h]),
                'init_mean': row['undeformed_mean_mm']}

    curves = [((verts_np, faces), 'gold', 1.2)]
    if gt_pair is not None:
        curves.append((gt_pair, 'limegreen', 1.2))
    curves.append(((v, faces), 'deepskyblue', 1.7))
    render_sample(name, ctx['target_size'][0], gt_wm_mask.astype(np.float32), curves,
                  centroid_slices(gt_wm_mask), dist, row['flip_pct'], res_mm,
                  args.push, out / f'{name}_mesh.png', args.dpi)
    if not args.no_save_mesh:
        save_deformed_surf(v, faces, ref, out / f'{name}_deformed.white.surf')
    return row


def main():
    ap = argparse.ArgumentParser(
        description='Evaluate the deployment condition: flow from SynthSeg input, scored vs GT')
    ap.add_argument('--model', required=True,
                    help='run dir, a best_model.pth path, or a run name under output.base_dir')
    ap.add_argument('--config', default=None, help='config.yaml (default: ./config.yaml)')
    ap.add_argument('--output_dir', default=None, help='default <run_dir>/synthseg_eval')
    ap.add_argument('--val_txt', default=None, help='override config data.val_txt')
    ap.add_argument('--num_samples', type=int, default=10,
                    help='subjects, evenly spaced across the val list')
    ap.add_argument('--sample_idxs', default=None, help='comma list of val indices')
    ap.add_argument('--synthseg_filename', default='synthseg_onehot_v1.npy',
                    help='SynthSeg one-hot inside each subject dir (the model input)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--affine', choices=['auto', 'on', 'off'], default='auto')
    ap.add_argument('--push', choices=['svf_inv', 'numeric', 'rv', 'rv_noaffine'],
                    default='svf_inv',
                    help='svf_inv = exp(-v), the exact SVF inverse')
    ap.add_argument('--inv_iter', type=int, default=500)
    ap.add_argument('--inv_alpha', type=float, default=0.5)
    ap.add_argument('--template_surf', nargs=2, metavar=('LH', 'RH'), default=None)
    ap.add_argument('--no_mesh', action='store_true', help='voxel Dice only (fast)')
    ap.add_argument('--no_self_int', action='store_true')
    ap.add_argument('--no_save_mesh', action='store_true')
    ap.add_argument('--dpi', type=int, default=140)
    args = ap.parse_args()

    # A config name resolves against the S-RegNET dir, not the cwd.
    cfg_path = Path(args.config or 'config.yaml').expanduser()
    if not cfg_path.is_file() and not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    args.config = str(cfg_path)

    cfg = load_config(args.config)
    ckpt = resolve_checkpoint(args.model, cfg)
    use_affine = detect_affine(ckpt) if args.affine == 'auto' else args.affine == 'on'
    out = Path(args.output_dir).expanduser() if args.output_dir \
        else Path(ckpt).parent.parent / 'synthseg_eval'
    try:
        out.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise SystemExit(f"[synth] cannot write {out} — the run dir belongs to another "
                         f"user. Pass --output_dir <a dir you own>.")
    write_git_sha(out)

    print(f"[synth] checkpoint = {ckpt}")
    print(f"[synth] device = {args.device} | affine = {use_affine} | push = {args.push} | "
          f"input = {args.synthseg_filename} | output_dir = {out}", flush=True)

    ctx = setup_inference(ckpt, args.config, args.device, use_affine=use_affine, verbose=True)
    d, nc, dev = cfg['data'], ctx['num_classes'], ctx['device']
    template_subject = Path(d['template_seg_path']).parent.name

    verts_t = verts_np = faces = n_lh = None
    template_si = None
    if not args.no_mesh:
        verts_np, faces, n_lh = load_template_mesh(cfg, args.template_surf)
        verts_t = torch.tensor(verts_np, dtype=torch.float32, device=dev)
        print(f"[synth] template mesh {len(verts_np):,} verts ({n_lh:,} lh) | "
              f"{len(faces):,} faces", flush=True)
        if not args.no_self_int:
            # The floor the warp adds to — 0 for the repaired template.
            tpl_ref = nib.load(ref_for(d['template_seg_path']))
            template_si = self_intersections(norm_to_world(verts_np, tpl_ref), faces,
                                             f'template {template_subject}')[0]
            print(f"[synth] template self-int {template_si['si_faces']} faces "
                  f"({template_si['si_faces_pct']:.4f}%)", flush=True)

    subject_dirs = [Path(p).parent for p in
                    Path(args.val_txt or d['val_txt']).read_text().splitlines() if p.strip()]
    if args.sample_idxs:
        idxs = [int(x) for x in args.sample_idxs.split(',')]
    else:
        idxs = sorted(set(np.linspace(0, len(subject_dirs) - 1,
                                      min(args.num_samples, len(subject_dirs)))
                          .astype(int).tolist()))
    for i in idxs:
        if not (0 <= i < len(subject_dirs)):
            raise IndexError(f"sample index {i} out of range [0, {len(subject_dirs)})")
    print(f"[synth] {len(subject_dirs)} val subjects | evaluating {idxs}", flush=True)

    csv_path = out / 'metrics_per_subject.csv'
    rows, writer, fh = [], None, None
    for i in idxs:
        sd = subject_dirs[i]
        name = sd.name
        if name == template_subject:
            print(f"[synth] skip {name} (template subject)", flush=True)
            continue
        synth_path = sd / args.synthseg_filename
        if not synth_path.is_file():
            print(f"[synth] skip {name}: no {synth_path}", flush=True)
            continue

        synth_seg = load_onehot(synth_path, ctx['target_size']).to(dev)
        gt_seg = load_onehot(sd / d['seg_filename'], ctx['target_size']).to(dev)

        # The one substantive change vs evaluate_all: the flow is conditioned on
        # the SynthSeg seg, the score on the GT seg.
        with torch.no_grad():
            warped, flow_fw, flow_rv, _lam, affine = warp_template(ctx, synth_seg)
        dice_gt, fg_gt = compute_dice_score(warped, gt_seg, nc)
        dice_syn, _ = compute_dice_score(warped, synth_seg, nc)
        dice_floor, _ = compute_dice_score(synth_seg, gt_seg, nc)
        fold, min_det = folding_stats(flow_fw)

        row = {'subject': name, 'idx': int(i),
               **{f'dice_c{c}_gt': dice_gt[c] for c in range(nc)},
               'dice_wm_gt': dice_gt[WM_LABEL],
               'dice_fg_gt': fg_gt,
               'dice_wm_synthseg': dice_syn[WM_LABEL],
               'synthseg_wm_vs_gt': dice_floor[WM_LABEL],
               'folding_pct': fold, 'min_det': min_det}

        # Voxel half prints immediately — the mesh half below takes minutes per
        # subject (self-intersection narrowphase + KD-trees are CPU-bound).
        print(f"[synth] {len(rows) + 1:3d} {name:>20s} WM-vs-GT {row['dice_wm_gt']:.4f} | "
              f"vs-SynthSeg {row['dice_wm_synthseg']:.4f} | floor "
              f"{row['synthseg_wm_vs_gt']:.4f} | fold {row['folding_pct']:.4f}%",
              flush=True)

        gt_wm_mask = (gt_seg.argmax(1)[0].cpu().numpy() == WM_LABEL)
        render_seg(name,
                   synth_seg.argmax(1)[0].cpu().numpy() == WM_LABEL,
                   gt_wm_mask,
                   warped.argmax(1)[0].cpu().numpy() == WM_LABEL,
                   centroid_slices(gt_wm_mask), row, out / f'{name}_seg.png', args.dpi)

        if not args.no_mesh:
            row.update(mesh_eval(ctx, args, verts_t, verts_np, faces, n_lh,
                                 (flow_fw, flow_rv), affine, synth_seg, sd,
                                 gt_wm_mask, name, out))
            msg = f"[synth]     {name} mesh:"
            if 'sym_mean_mm' in row:
                msg += (f" sym {row['sym_mean_mm']:5.2f} mm | hd95 "
                        f"{row['sym_hd95_mm']:5.2f} | undef {row['undeformed_mean_mm']:5.2f}")
            if 'flip_pct' in row:
                msg += f" | flip {row['flip_pct']:.4f}%"
            if 'si_faces_pct' in row:
                msg += f" | self-int {row['si_faces_pct']:.4f}%"
            print(msg, flush=True)

        if fh is None:                              # header from the first row
            fh = open(csv_path, 'w', newline='')
            writer = csv.DictWriter(fh, fieldnames=list(row))
            writer.writeheader()
        assert writer is not None
        dropped = [k for k in row if k not in writer.fieldnames]
        if dropped:      # first row set the header; don't lose columns silently
            print(f"[synth] WARNING {name}: columns absent from the CSV header "
                  f"(first subject lacked them): {dropped}", flush=True)
        writer.writerow({k: row.get(k) for k in writer.fieldnames})
        fh.flush()                                  # partial results survive a kill
        rows.append(row)

    if fh is not None:
        fh.close()
    if not rows:
        raise SystemExit("no subjects evaluated")

    summary = {'checkpoint': ckpt, 'affine': use_affine, 'push': args.push,
               'synthseg_filename': args.synthseg_filename,
               'template_subject_excluded': template_subject,
               'template_self_int': template_si, 'n_subjects': len(rows),
               'stats': summarize(rows, 'SynthSeg-input eval'),
               'per_subject': rows}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))

    mean_wm = float(np.mean([r['dice_wm_gt'] for r in rows]))
    print(f"\n[synth] HEADLINE mean WM Dice vs GT = {mean_wm:.4f} over {len(rows)} subjects "
          f"(bar: 0.85)")
    if not args.no_mesh and not args.no_self_int:
        si_vals = [r['si_faces_pct'] for r in rows if 'si_faces_pct' in r]
        if si_vals and template_si is not None:
            print(f"[synth] mean pushed-mesh self-int = {float(np.mean(si_vals)):.4f}% "
                  f"(template baseline {template_si['si_faces_pct']:.4f}%)")
    print(f"[synth] per-subject CSV -> {out / 'metrics_per_subject.csv'}")
    print(f"[synth] summary        -> {out / 'summary.json'}")


if __name__ == '__main__':
    main()
