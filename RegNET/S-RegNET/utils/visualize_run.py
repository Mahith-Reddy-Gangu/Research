"""
Visualize a trained run: a WM error grid across N validation subjects plus
per-subject contour / deformation-field figures and a metrics summary.

Ported from the invertible-deform-SRegNET branch viz tools
(utils/wm_diff_grid.py for the error grid, utils/mesh_slice_figure.py for the
checkpoint/affine resolution and centroid slicing) and adapted to this branch's
SVF model, whose forward returns (flow_fw, flow_rv, lambda_map, affine_matrix).
The mesh-native parts of those tools are dropped: they need
wm_template.py / overlay_surface.py / model.push_points_to_sample, none of which
exist here. This is the voxel side — the same numbers training optimizes.

The warp replays training exactly (train.py:_warp_template): affine-align the
template (bilinear) then STN with flow_fw as a PULL field. Affine is
auto-detected from the checkpoint, so a run trained with affine on/off needs no
flag.

Outputs (into --output_dir):
    diff_grid.png          N×3 TP/FP/FN grid for the target class + warped outline
    <subj>_contours.png    template / target / warped contours (inference.py)
    <subj>_field.png       flow magnitude + per-axis components (inference.py)
    summary.json           per-subject and mean metrics
    GIT_SHA.txt            provenance

Run from the S-RegNET directory (needs torch):
    python visualize_run.py --model ~/shared_scratch/training_results/mahith_experiment/20260801_083414
    # --model accepts a run dir, a direct best_model.pth path, or a run name
    #   under output.base_dir
    # --num_samples 5              (default; evenly spaced across the val list)
    # --sample_idxs 0,12,44,120    (explicit val indices instead)
    # --target_class 3             (WM; 1=cortex 2=subGM 4=CSF)
    # --no_per_sample              (grid + summary only, skip per-subject PNGs)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from git_provenance import write_git_sha
from get_data import SegDataset
from inference import (load_config, setup_inference,
                       plot_contour_overlay, plot_deformation_field)
from losses import compute_dice_score, cycle_consistency_loss, jacobian_det

TP_COL = (0.55, 0.55, 0.55)     # hit
FP_COL = (0.95, 0.25, 0.25)     # over-reach (warped ∖ target)
FN_COL = (0.25, 0.55, 1.00)     # under-reach (target ∖ warped)
OUTLINE_COL = 'cyan'
AXIS_NAMES = ['axial (z)', 'coronal (y)', 'sagittal (x)']


# =============================================================================
# Checkpoint / affine resolution
# =============================================================================

def resolve_checkpoint(model, cfg):
    """Turn --model into a checkpoint path: a direct .pth, a run dir, or a run
    name under output.base_dir."""
    sub = cfg['output'].get('checkpoint_subdir', 'checkpoints')
    p = Path(model).expanduser()
    candidates = [p, p / sub / 'best_model.pth',
                  Path(cfg['output']['base_dir']) / model / sub / 'best_model.pth']
    for c in candidates:
        if c.is_file():
            return str(c)
    raise FileNotFoundError(
        f"Could not resolve --model '{model}' to a checkpoint. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates))


def detect_affine(ckpt_path):
    """Whether the checkpoint carries an affine head. The checkpoint — not
    config.yaml — is the source of truth: a run dir can outlive a config edit."""
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['model_state_dict']
    return any(k.startswith('affine_net') for k in sd)


# =============================================================================
# Warp + metrics
# =============================================================================

@torch.no_grad()
def warp_template(ctx, sample_seg):
    """Replay the critical invariant: affine first, then dense flow.
    Returns (warped_seg, flow_fw, flow_rv, lambda_map, affine_matrix)."""
    model, stn, template_seg = ctx['model'], ctx['stn'], ctx['template_seg']
    flow_fw, flow_rv, lambda_map, affine_matrix = model(template_seg, sample_seg)
    if affine_matrix is not None:
        grid = F.affine_grid(affine_matrix, template_seg.size(), align_corners=False)
        aligned = F.grid_sample(template_seg, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
        warped_seg = stn(aligned, flow_fw)
    else:
        warped_seg = stn(template_seg, flow_fw)
    return warped_seg, flow_fw, flow_rv, lambda_map, affine_matrix


def folding_stats(flow):
    """(% voxels with negative Jacobian determinant, min normalized det) —
    matches inference.py / train.py's val diagnostics."""
    _, _, D, H, W = flow.shape
    det_ref = (2.0 / D) * (2.0 / H) * (2.0 / W)
    det = jacobian_det(flow) / det_ref
    return (det < 0).float().mean().item() * 100.0, det.min().item()


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


@torch.no_grad()
def build_row(ctx, ds, idx, target_class):
    """Warp one subject; return the masks, metrics, and centroid slices."""
    device = ctx['device']
    sample = ds[idx]
    name = Path(ds.subject_dirs[idx]).name
    sample_seg = sample['sample_seg'].unsqueeze(0).to(device)

    warped_seg, flow_fw, flow_rv, lambda_map, _ = warp_template(ctx, sample_seg)

    warped_m = (warped_seg.argmax(1)[0] == target_class).cpu().numpy()
    target_m = (sample['sample_seg'].argmax(0) == target_class).numpy()
    fold_pct, min_det = folding_stats(flow_fw)
    dice_per_class, fg_dice = compute_dice_score(warped_seg, sample_seg, ctx['num_classes'])
    flow_mag = flow_fw.pow(2).sum(1).sqrt()

    return {
        'name': name, 'idx': int(idx),
        'warped_m': warped_m, 'target_m': target_m,
        'slices': centroid_slices(target_m),
        'metrics': {
            'dice_per_class': dice_per_class,
            'dice_fg_mean': fg_dice,
            'dice_target_class': dice_per_class[target_class],
            'folding_pct': fold_pct,
            'min_det': min_det,
            'fp_voxels': int(np.logical_and(warped_m, ~target_m).sum()),
            'fn_voxels': int(np.logical_and(~warped_m, target_m).sum()),
            'cycle': cycle_consistency_loss(flow_fw, flow_rv, ctx['stn']).item(),
            'flow_mag_mean': flow_mag.mean().item(),
            'flow_mag_max': flow_mag.max().item(),
            'lambda_mean': lambda_map.mean().item(),
            'lambda_std': lambda_map.std().item(),
        },
        # kept for the optional per-sample figures, dropped from summary.json
        '_tensors': (sample_seg, warped_seg, flow_fw),
    }


# =============================================================================
# Figures
# =============================================================================

def diff_rgb(w2d, g2d):
    """TP/FP/FN colour image from two boolean 2D slices."""
    rgb = np.zeros((*w2d.shape, 3), dtype=np.float32)
    rgb[w2d & g2d] = TP_COL
    rgb[w2d & ~g2d] = FP_COL
    rgb[~w2d & g2d] = FN_COL
    return rgb


def make_diff_grid(rows, output, class_name, dpi=130):
    """N×3 grid: where the warped template's target class agrees with the
    subject / over-reaches / under-reaches. A thin red/blue rim is the
    regularization tax; a solid blob is gross misalignment."""
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)

    for r, row in enumerate(rows):
        m = row['metrics']
        for a in range(3):
            ax = axes[r][a]
            s = int(row['slices'][a])
            w2d = np.take(row['warped_m'], s, axis=a).T
            g2d = np.take(row['target_m'], s, axis=a).T
            ax.imshow(diff_rgb(w2d, g2d), origin='lower', aspect='equal',
                      interpolation='nearest')
            if w2d.any():
                ax.contour(w2d.astype(float), levels=[0.5], colors=OUTLINE_COL,
                           linewidths=0.5, alpha=0.85)
            if r == 0:
                ax.set_title(f'{AXIS_NAMES[a]} @ {s}', fontsize=11)
            if a == 0:
                ax.set_ylabel(f"{row['name']}\n{class_name} Dice {m['dice_target_class']:.3f}\n"
                              f"fold {m['folding_pct']:.3f}%",
                              fontsize=8, rotation=0, ha='right', va='center', labelpad=38)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            else:
                ax.axis('off')

    legend = [
        Patch(facecolor=TP_COL, label='TP  warped ∩ target'),
        Patch(facecolor=FP_COL, label='FP  warped ∖ target (over-reach)'),
        Patch(facecolor=FN_COL, label='FN  target ∖ warped (under-reach)'),
        Line2D([0], [0], color=OUTLINE_COL, lw=1.5, label='warped outline'),
    ]
    fig.suptitle(f"{class_name} Dice error grid   [voxel pull-warp, matches training Dice; "
                 f"slices at target-class centroid]", fontsize=12, fontweight='bold')
    fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=9, framealpha=0.6)
    fig.tight_layout(rect=(0.04, 0.03, 1.0, 0.98))
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] saved: {output}")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description='Visualize a trained S-RegNET run')
    ap.add_argument('--model', required=True,
                    help='run dir, a direct best_model.pth path, or a run name under output.base_dir')
    ap.add_argument('--output_dir', default=None,
                    help='where figures land (default <run_dir>/viz_output)')
    ap.add_argument('--config', default=None, help='config.yaml (default: ./config.yaml)')
    ap.add_argument('--val_txt', default=None, help='override config data.val_txt')
    ap.add_argument('--num_samples', type=int, default=5,
                    help='subjects to visualize, evenly spaced across the val list')
    ap.add_argument('--sample_idxs', default=None,
                    help='comma list of val indices (overrides --num_samples)')
    ap.add_argument('--target_class', type=int, default=3,
                    help='class for the diff grid (default 3 = White Matter)')
    ap.add_argument('--device', default='cuda:0', help='e.g. cuda:0, cpu')
    ap.add_argument('--affine', choices=['auto', 'on', 'off'], default='auto',
                    help='affine stage; auto = detect the affine head from the checkpoint')
    ap.add_argument('--no_per_sample', action='store_true',
                    help='skip the per-subject contour / field PNGs')
    ap.add_argument('--dpi', type=int, default=130)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = resolve_checkpoint(args.model, cfg)
    use_affine = detect_affine(ckpt) if args.affine == 'auto' else args.affine == 'on'
    out = Path(args.output_dir).expanduser() if args.output_dir \
        else Path(ckpt).parent.parent / cfg['output'].get('viz_subdir', 'viz_output')
    out.mkdir(parents=True, exist_ok=True)
    write_git_sha(out)

    print(f"[viz] checkpoint = {ckpt}")
    print(f"[viz] device = {args.device} | affine = {use_affine} | output_dir = {out}", flush=True)

    ctx = setup_inference(ckpt, args.config, args.device, use_affine=use_affine, verbose=True)
    class_name = ctx['class_names'][args.target_class]

    d = cfg['data']
    ds = SegDataset(args.val_txt or d['val_txt'], d['template_seg_path'],
                    target_size=ctx['target_size'], seg_filename=d['seg_filename'],
                    preload=False)

    if args.sample_idxs:
        idxs = [int(x) for x in args.sample_idxs.split(',')]
    else:
        idxs = sorted(set(np.linspace(0, len(ds) - 1,
                                      min(args.num_samples, len(ds))).astype(int).tolist()))
    for i in idxs:
        if not (0 <= i < len(ds)):
            raise IndexError(f"sample index {i} out of range [0, {len(ds)})")
    print(f"[viz] {len(ds)} val subjects | visualizing {idxs} | diff class = "
          f"{args.target_class} ({class_name})", flush=True)

    rows = []
    for i in idxs:
        row = build_row(ctx, ds, i, args.target_class)
        m = row['metrics']
        print(f"[viz] {row['name']:>20s} (idx {i:4d}) | {class_name} Dice "
              f"{m['dice_target_class']:.4f} | fg Dice {m['dice_fg_mean']:.4f} | "
              f"folding {m['folding_pct']:.4f}% | FP {m['fp_voxels']:7d} | "
              f"FN {m['fn_voxels']:7d}", flush=True)

        if not args.no_per_sample:
            sample_seg, warped_seg, flow_fw = row['_tensors']
            plot_contour_overlay(ctx['template_seg'], sample_seg, warped_seg,
                                 out / f"{row['name']}_contours.png",
                                 target_class=args.target_class)
            plot_deformation_field(flow_fw.squeeze(0).cpu().numpy(),
                                   out / f"{row['name']}_field.png")
        row.pop('_tensors')
        rows.append(row)

    make_diff_grid(rows, out / 'diff_grid.png', class_name, args.dpi)

    keys = list(rows[0]['metrics'])
    means = {k: float(np.mean([r['metrics'][k] for r in rows]))
             for k in keys if k != 'dice_per_class'}
    means['dice_per_class'] = np.mean([r['metrics']['dice_per_class'] for r in rows],
                                      axis=0).tolist()
    (out / 'summary.json').write_text(json.dumps({
        'checkpoint': ckpt, 'affine': use_affine, 'target_class': args.target_class,
        'class_names': ctx['class_names'], 'sample_idxs': idxs,
        'per_sample': [{'name': r['name'], 'idx': r['idx'], **r['metrics']} for r in rows],
        'mean': means,
    }, indent=2))

    print(f"\n[viz] N={len(rows)} | mean {class_name} Dice {means['dice_target_class']:.4f} | "
          f"mean fg Dice {means['dice_fg_mean']:.4f} | mean folding {means['folding_pct']:.4f}% | "
          f"cycle {means['cycle']:.6f}")
    print(f"[viz] per-class Dice " + " | ".join(
        f"{n}: {v:.4f}" for n, v in zip(ctx['class_names'], means['dice_per_class'])))
    print(f"[viz] summary -> {out / 'summary.json'}")


if __name__ == '__main__':
    main()
