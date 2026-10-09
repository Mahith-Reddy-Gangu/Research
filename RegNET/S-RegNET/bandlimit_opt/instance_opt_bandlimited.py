"""
Band-limited instance optimization: does capping the delta-velocity's spatial
frequency remove the sub-voxel wrinkles that full-resolution instance
optimization added (mesh self-int 0.20% -> 1.48%) while keeping its voxel
gains (WM Dice 0.844 -> 0.875)?

Instead of optimizing the full 128^3 velocity,
the net's velocity is FROZEN and only a coarse d^3 correction is optimized,
trilinearly upsampled before use:

    v = v_net + upsample(delta),   delta in R^{d x d x d},  delta_0 = 0

One arm per coarse level d (default 128, 64, 32): d=128 reproduces the old
full-resolution behavior (control); d=64 caps added content at wavelength
>= 4 voxels; d=32 at >= 8 voxels. Plus a `baseline` arm: the net prediction
replayed exactly, untouched, untied.

Tied velocities (default on): a SINGLE velocity v = v_net_fw + upsample(delta)
with flow_fw = exp(+v), flow_rv = exp(-v) — the inverse is exact by
construction (the mesh push uses the inverse field); the net's independent
vel_rv is discarded. --no_tie_velocities keeps two independent deltas on the
two net fields like the old probe.

The trained model's affine matrix and lambda map are FROZEN; fp32, no
autocast. --fit_target defaults to 'input' (deployment condition: the
objective fits the SynthSeg input seg); metrics always score against GT.
Metrics are recorded every 50 iterations so overfitting over iterations is
visible. Reports numbers and figures only. Outputs into --output_dir:

    metrics.json                   everything: per-subject, per-arm,
                                   per-iteration-checkpoint + arm means
    loss_curves/<subj>_<arm>.json  per-term optimization curves
    flows/<subj>_<arm>.npz         raw UNet-scale vel_fw/vel_rv (+ affine),
                                   consumed by utils/push_optimized_mesh.py
    figures/<subj>_contours.png    arms x 3 ortho slices, GT vs warped WM
    GIT_SHA.txt                    provenance

Run from the S-RegNET directory (needs torch + scipy; GPU cluster):
    python bandlimit_opt/instance_opt_bandlimited.py \\
        --checkpoint <run_dir_or_best_model.pth> --config config.yaml \\
        --device cuda:1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))          # repo-root modules, whatever the cwd is
sys.path.insert(0, str(_ROOT / 'utils'))

from git_provenance import write_git_sha
from inference import load_config, setup_inference
from losses import SegRegistrationLoss, compute_dice_score
from train import _warp_template, _warp_sample_inverse
from visualize_run import (resolve_checkpoint, detect_affine, folding_stats,
                           centroid_slices)

GT_C, WARP_C = '#32CD32', '#00BFFF'
AXIS_NAMES = ['axial (z)', 'coronal (y)', 'sagittal (x)']
LOG_TERMS = ('dice', 'cross_entropy', 'bending', 'jacobian', 'displacement',
             'lambda_smoothness')
WM = 3                                  # white-matter channel in the 5-class one-hot
EDGE_BAND_VOX = 3                       # half-width of the edge-Dice band


def load_onehot(path, target_size):
    """One-hot .npy -> (1,5,D,H,W) float tensor, nearest-resized like SegDataset."""
    seg = torch.tensor(np.load(path), dtype=torch.float32)
    return F.interpolate(seg.unsqueeze(0), size=target_size, mode='nearest')


def integrate(vel, stn):
    """Exact scaling-and-squaring from model.forward: vel/2**7, 7 squarings."""
    flow = vel / (2 ** 7)
    for _ in range(7):
        flow = flow + stn(flow, flow)
    return flow


@torch.no_grad()
def net_init(model, template_seg, input_seg, affine=None):
    """Frozen forward pass of the trained pieces, mirroring model.forward up to
    (but not including) the integration. The affine warp of the template for
    the UNet input uses mode='nearest', exactly as model.forward does. An
    `affine` passed in replaces the affine net's prediction (the UNet then sees
    the template aligned by it)."""
    template_in = template_seg
    if model.use_affine:
        if affine is None:
            affine = model.affine_net(template_seg, input_seg)
        grid = F.affine_grid(affine, template_seg.size(), align_corners=False)
        template_in = F.grid_sample(template_seg, grid, mode='nearest',
                                    padding_mode='zeros', align_corners=False)
    vel, lambda_map = model.unet(torch.cat([template_in, input_seg], dim=1))
    if vel.shape[1] == 3:               # band-limited tied head: single v, rv = -v
        return vel, -vel, lambda_map, affine
    return vel[:, :3], vel[:, 3:], lambda_map, affine


def edge_band(wm, width=EDGE_BAND_VOX):
    """Voxels within `width` voxels (Euclidean) of the GT WM boundary. The
    boundary is the WM surface shell (wm & ~erode(wm)); the band is the EDT of
    that shell thresholded at `width` — symmetric inside/outside, ~2·width+1
    voxels thick."""
    shell = wm & ~binary_erosion(wm)
    return distance_transform_edt(~shell) <= width


def band_dice(pred, gt, band, eps=1e-5):
    """Hard Dice of two boolean masks restricted to the band."""
    p, g = pred[band], gt[band]
    inter = np.logical_and(p, g).sum()
    return float((2.0 * inter + eps) / (p.sum() + g.sum() + eps))


def surface_distances(pred, gt):
    """Mean symmetric surface distance and 95th-percentile Hausdorff (voxels)
    between the boundary shells (mask & ~erode) of two boolean masks, pooling
    both EDT directions."""
    pred_b = pred & ~binary_erosion(pred)
    gt_b = gt & ~binary_erosion(gt)
    if not pred_b.any() or not gt_b.any():
        return float('nan'), float('nan')
    d = np.concatenate([distance_transform_edt(~gt_b)[pred_b],
                        distance_transform_edt(~pred_b)[gt_b]])
    return float(d.mean()), float(np.percentile(d, 95))
METRIC_KEYS = ('wm_dice', 'edge_dice', 'msd_vox', 'hd95_vox', 'folding_pct',
               'worst_det', 'inverse_residual_vox')
LOG_EVERY = 25                          # loss-curve sampling
METRIC_EVERY = 50                       # full-metric checkpoints


def lowpass(vel, d, target_size):
    """Trilinear down-up resampling through a d^3 grid (the same filter the
    band-limited head bakes in). Applied to the NET INIT when --lowpass_init
    is set; the baseline arm always replays the unfiltered net."""
    down = F.interpolate(vel, size=(d, d, d), mode='trilinear',
                         align_corners=False)
    return F.interpolate(down, size=target_size, mode='trilinear',
                         align_corners=False)


def build_velocities(delta, s):
    """Coarse delta -> full-res raw UNet-scale velocities (pre /2**7).
    Tied: v = v_net_fw + upsample(delta), returned as (+v, -v) so exp(-v) is
    the exact inverse of exp(+v). Untied: independent deltas on both fields."""
    up = F.interpolate(delta, size=s['target_size'], mode='trilinear',
                       align_corners=False)
    if s['tied']:
        v = s['vel_fw0'] + up
        return v, -v
    return s['vel_fw0'] + up[:, :3], s['vel_rv0'] + up[:, 3:]


def loss_and_terms(vel_fw, vel_rv, s):
    """One objective evaluation: integrate both velocities, replay the training
    warps (affine first, then dense flow — train.py invariant), score against
    fit_seg with the training weights — exactly the old probe's control arm."""
    flow_fw = integrate(vel_fw, s['stn'])
    flow_rv = integrate(vel_rv, s['stn'])
    warped_fw = _warp_template(s['template_seg'], flow_fw, s['affine'], s['stn'])
    warped_rv = _warp_sample_inverse(s['fit_seg'], flow_rv, s['affine'], s['stn'])
    # affine_matrix=None: the affine is frozen, so its reg terms are constants
    # with no gradient — same reason lambda_prior is zero-weighted in loss_fn.
    total, terms = s['loss_fn'](warped_fw, s['fit_seg'], warped_rv, s['template_seg'],
                                flow_fw, flow_rv, s['lambda_map'],
                                affine_matrix=None, return_components=True)
    return total, terms


def inverse_residual_vox(flow_fw, flow_rv, stn):
    """Mean composition error |fw o rv| and |rv o fw| in voxels — the
    cycle_consistency_loss composition, reported as a mean magnitude instead
    of a mean square. Flows are normalized grid units; one unit = size/2 vox."""
    _, _, D, H, W = flow_fw.shape
    scale = torch.tensor([W / 2.0, H / 2.0, D / 2.0], device=flow_fw.device,
                         dtype=flow_fw.dtype).view(1, 3, 1, 1, 1)
    fw_rv = flow_fw + stn(flow_rv, flow_fw)
    rv_fw = flow_rv + stn(flow_fw, flow_rv)
    r1 = (fw_rv * scale).pow(2).sum(1).sqrt().mean()
    r2 = (rv_fw * scale).pow(2).sum(1).sqrt().mean()
    return float(0.5 * (r1 + r2))


@torch.no_grad()
def arm_metrics(vel_fw, vel_rv, s):
    """Identical metric computation for all arms. Metrics always score vs GT."""
    flow_fw = integrate(vel_fw, s['stn'])
    flow_rv = integrate(vel_rv, s['stn'])
    warped = _warp_template(s['template_seg'], flow_fw, s['affine'], s['stn'])
    dice_per_class, _ = compute_dice_score(warped, s['gt_seg'], s['num_classes'])
    warped_wm = (warped.argmax(1)[0] == WM).cpu().numpy()
    fold_pct, worst_det = folding_stats(flow_fw)
    msd, hd95 = surface_distances(warped_wm, s['gt_wm'])
    return {'wm_dice': dice_per_class[WM],
            'edge_dice': band_dice(warped_wm, s['gt_wm'], s['band']),
            'msd_vox': msd, 'hd95_vox': hd95,
            'folding_pct': fold_pct, 'worst_det': worst_det,
            'inverse_residual_vox':
                inverse_residual_vox(flow_fw, flow_rv, s['stn'])}, warped_wm


def _log(name, arm, it, n, total, terms):
    vals = ' | '.join(f'{k} {float(terms[k]):.5f}' for k in LOG_TERMS)
    print(f'[probe] {name} {arm} iter {it}/{n} total {float(total):.5f} | {vals}',
          flush=True)


def _record(curve, it, total, terms):
    curve['iters'].append(int(it))
    curve['total'].append(float(total))
    for k in LOG_TERMS:
        curve['terms'][k].append(float(terms[k]))


def _print_metrics(name, arm, it, met):
    print(f"[probe] {name} {arm} @ iter {it}: WM {met['wm_dice']:.4f} | "
          f"edge {met['edge_dice']:.4f} | msd {met['msd_vox']:.3f} vox | "
          f"hd95 {met['hd95_vox']:.3f} vox | fold {met['folding_pct']:.4f}% | "
          f"worst det {met['worst_det']:.3f} | "
          f"inv res {met['inverse_residual_vox']:.4f} vox", flush=True)


def optimize_arm(arm, d, n_iters, name, s, args, metric_iters):
    """Optimize a zero-initialized coarse d^3 delta on the frozen net velocity.
    fp32 throughout, no autocast; the delta is the only leaf with gradients.
    Full metrics are recorded at iter 0 and every METRIC_EVERY iterations."""
    ch = 3 if s['tied'] else 6
    delta = torch.zeros(1, ch, d, d, d, device=s['dev'], requires_grad=True)
    opt = torch.optim.Adam([delta], lr=args.lr)
    curve = {'iters': [], 'total': [], 'terms': {k: [] for k in LOG_TERMS}}
    checkpoints = {}

    vel_fw, vel_rv = build_velocities(delta.detach(), s)
    checkpoints[0], _ = arm_metrics(vel_fw, vel_rv, s)
    _print_metrics(name, arm, 0, checkpoints[0])

    for it in range(n_iters):
        opt.zero_grad()
        vel_fw, vel_rv = build_velocities(delta, s)
        total, terms = loss_and_terms(vel_fw, vel_rv, s)
        total.backward()
        opt.step()
        step = it + 1
        if step == 1 or step % LOG_EVERY == 0 or step == n_iters:
            total_d = total.detach()
            terms_d = {k: terms[k].detach() for k in LOG_TERMS}
            _record(curve, step, total_d, terms_d)
            _log(name, arm, step, n_iters, total_d, terms_d)
        if step in metric_iters:
            vel_fw, vel_rv = build_velocities(delta.detach(), s)
            checkpoints[step], _ = arm_metrics(vel_fw, vel_rv, s)
            _print_metrics(name, arm, step, checkpoints[step])

    vel_fw, vel_rv = build_velocities(delta.detach(), s)
    return vel_fw.detach(), vel_rv.detach(), curve, checkpoints


def render_contours(name, gt_wm, arms, warped_by_arm, metrics_by_arm, slices,
                    output, dpi=140):
    """len(arms) rows x 3 ortho slices at the GT-WM centroid: GT WM mask as
    grayscale background, GT WM contour green, warped-template WM contour blue."""
    n = len(arms)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5.7 * n), squeeze=False,
                             facecolor='#f5f5f0')
    for r, arm in enumerate(arms):
        w, m = warped_by_arm[arm], metrics_by_arm[arm]
        for a in range(3):
            ax = axes[r][a]
            si = int(slices[a])
            ax.set_facecolor('black')
            ax.imshow(np.take(gt_wm, si, axis=a), cmap='gray', interpolation='none',
                      vmin=0, vmax=1)
            for mask, col in ((gt_wm, GT_C), (w, WARP_C)):
                m2d = np.take(mask, si, axis=a).astype(float)
                if m2d.any():
                    ax.contour(m2d, levels=[0.5], colors=[col], linewidths=1.4)
            if r == 0:
                ax.set_title(f'{AXIS_NAMES[a]} @ {si}')
            ax.set_xticks([]); ax.set_yticks([])
        axes[r][0].set_ylabel(
            f"{arm}\nWM {m['wm_dice']:.4f} | edge {m['edge_dice']:.4f}\n"
            f"hd95 {m['hd95_vox']:.2f} vox | fold {m['folding_pct']:.4f}%\n"
            f"inv res {m['inverse_residual_vox']:.3f} vox",
            fontsize=9, rotation=0, ha='right', va='center', labelpad=50)
    axes[0][0].legend([Line2D([0], [0], color=c, lw=2) for c in (GT_C, WARP_C)],
                      ['GT WM (scoring target)', 'warped template WM'],
                      loc='lower right', facecolor='gray', edgecolor='black',
                      framealpha=0.9)
    fig.suptitle(f'{name} — band-limited instance optimization of the velocity field',
                 fontsize=12, fontweight='bold')
    plt.subplots_adjust(wspace=0.05, hspace=0.08)
    fig.savefig(output, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def save_flows(out, name, arm, vel_fw, vel_rv, affine):
    """push_optimized_mesh.py reads vel_fw, vel_rv, and affine_matrix (when
    present) from these keys."""
    extra = {} if affine is None else \
        {'affine_matrix': affine.squeeze(0).cpu().numpy()}
    np.savez(out / 'flows' / f'{name}_{arm}.npz',
             vel_fw=vel_fw.squeeze(0).cpu().numpy(),
             vel_rv=vel_rv.squeeze(0).cpu().numpy(), **extra)


def metric_iters_for(n_iters):
    """Full-metric checkpoint iterations for an n_iters arm."""
    return sorted(set(range(METRIC_EVERY, n_iters + 1, METRIC_EVERY)) | {n_iters})


def probe_subject(name, subject_dir, ctx, loss_fn, args, out, opt_arms):
    d = ctx['cfg']['data']
    dev, ts, stn = ctx['device'], ctx['target_size'], ctx['stn']
    input_name = d.get('input_seg_filename') or d['seg_filename']
    input_seg = load_onehot(subject_dir / input_name, ts).to(dev)
    gt_seg = load_onehot(subject_dir / d['seg_filename'], ts).to(dev)

    vel_fw0, vel_rv0, lambda_map, affine = net_init(ctx['model'],
                                                    ctx['template_seg'], input_seg)

    # Optionally low-pass the net init for the OPT arms (baseline stays raw).
    opt_fw0, opt_rv0 = vel_fw0, vel_rv0
    if args.lowpass_init:
        opt_fw0 = lowpass(vel_fw0, args.lowpass_init, ctx['target_size'])
        opt_rv0 = lowpass(vel_rv0, args.lowpass_init, ctx['target_size'])

    # fit_seg drives the objective; gt_seg only ever scores metrics.
    fit_seg = gt_seg if args.fit_target == 'gt' else input_seg
    gt_wm = (gt_seg.argmax(1)[0] == WM).cpu().numpy()
    slices = centroid_slices(gt_wm)

    s = {'template_seg': ctx['template_seg'], 'gt_seg': gt_seg, 'fit_seg': fit_seg,
         'lambda_map': lambda_map, 'affine': affine, 'stn': stn,
         'loss_fn': loss_fn, 'num_classes': ctx['num_classes'],
         'target_size': ts, 'dev': dev, 'tied': args.tie_velocities,
         'vel_fw0': opt_fw0, 'vel_rv0': opt_rv0,
         'gt_wm': gt_wm, 'band': edge_band(gt_wm)}

    arms = ['baseline'] + [a for a, _, _ in opt_arms]
    results, warped_by_arm, metrics_by_arm = {}, {}, {}
    for arm, d_, n_iters in [('baseline', None, 0)] + opt_arms:
        t0 = time.time()
        if arm == 'baseline':
            # Net prediction replayed exactly: untied, the net's own two fields.
            vel_fw, vel_rv = vel_fw0, vel_rv0
            met, warped_wm = arm_metrics(vel_fw, vel_rv, s)
            _print_metrics(name, arm, 0, met)
            with torch.no_grad():
                total, terms = loss_and_terms(vel_fw, vel_rv, s)
            curve = {'iters': [], 'total': [], 'terms': {k: [] for k in LOG_TERMS}}
            _record(curve, 0, total, terms)
            checkpoints = {0: met}
        else:
            vel_fw, vel_rv, curve, checkpoints = optimize_arm(
                arm, d_, n_iters, name, s, args, metric_iters_for(n_iters))
            met, warped_wm = arm_metrics(vel_fw, vel_rv, s)

        (out / 'loss_curves' / f'{name}_{arm}.json').write_text(json.dumps(curve))
        save_flows(out, name, arm, vel_fw, vel_rv, affine)
        warped_by_arm[arm], metrics_by_arm[arm] = warped_wm, met
        results[arm] = {'checkpoints': {str(k): v for k, v in checkpoints.items()},
                        'final': met, 'seconds': round(time.time() - t0, 1)}

    render_contours(name, gt_wm, arms, warped_by_arm, metrics_by_arm, slices,
                    out / 'figures' / f'{name}_contours.png')

    del input_seg, gt_seg, vel_fw0, vel_rv0, vel_fw, vel_rv, lambda_map, s
    return results


def main():
    ap = argparse.ArgumentParser(
        description='Band-limited per-subject instance optimization: frozen net '
                    'velocity + coarse-grid delta (one arm per coarse level)')
    ap.add_argument('--checkpoint', required=True,
                    help='run dir, a best_model.pth path, or a run name under output.base_dir')
    ap.add_argument('--config', default='config.yaml',
                    help='config.yaml (a bare name resolves against the S-RegNET dir)')
    ap.add_argument('--output_dir', default='bandlimit_opt/results_v1',
                    help='output dir (a relative path resolves against the S-RegNET dir)')
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--num_subjects', type=int, default=3,
                    help='subjects from the head of val_txt')
    ap.add_argument('--subjects', default=None,
                    help='comma list of subject dir names from val_txt '
                         '(overrides --num_subjects)')
    ap.add_argument('--coarse_levels', default='128,64,32',
                    help='comma list of delta-grid sizes d; one optimization arm '
                         'per level (d=128 reproduces the old full-res behavior)')
    ap.add_argument('--iters', default='150',
                    help='optimization steps per arm; a comma list runs one arm '
                         'per (level x count), named d<level>i<count>')
    ap.add_argument('--lr', type=float, default=1e-3,
                    help='Adam lr on the coarse delta (normalized grid units)')
    ap.add_argument('--tie_velocities', dest='tie_velocities',
                    action='store_true', default=True,
                    help='single velocity v: flow_fw = exp(+v), flow_rv = exp(-v) '
                         '(default on)')
    ap.add_argument('--no_tie_velocities', dest='tie_velocities',
                    action='store_false',
                    help='two independent deltas on the two net fields, like the '
                         'old probe')
    ap.add_argument('--lowpass_init', type=int, default=0,
                    help='if >0, low-pass the net-init velocities through a '
                         'd^3 grid (trilinear down-up) before the OPT arms; '
                         'the baseline arm always replays the raw net')
    ap.add_argument('--fit_target', choices=('gt', 'input'), default='input',
                    help="seg the optimization fits against: 'input' (deployment "
                         "condition — the objective never touches the GT seg) or "
                         "'gt' (probe condition); metrics always score vs GT")
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_file() and not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    args.config = str(cfg_path)

    cfg = load_config(args.config)
    ckpt = resolve_checkpoint(args.checkpoint, cfg)
    use_affine = detect_affine(ckpt)
    levels = [int(x) for x in args.coarse_levels.split(',') if x]
    iters_list = [int(x) for x in str(args.iters).split(',') if x]
    # Single count keeps the old d<level> arm names; a list disambiguates.
    opt_arms = [(f'd{d_}' if len(iters_list) == 1 else f'd{d_}i{n}', d_, n)
                for d_ in levels for n in iters_list]
    out = Path(args.output_dir).expanduser()
    if not out.is_absolute():
        out = _ROOT / out
    for sub in ('loss_curves', 'flows', 'figures'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    write_git_sha(out)

    print(f'[probe] checkpoint = {ckpt}')
    print(f'[probe] device = {args.device} | affine = {use_affine} | '
          f'tied = {args.tie_velocities} | levels = {levels} | '
          f'iters = {iters_list} (metrics every {METRIC_EVERY}) | lr = {args.lr} | '
          f'fit_target = {args.fit_target} | output_dir = {out}', flush=True)

    ctx = setup_inference(ckpt, args.config, args.device,
                          use_affine=use_affine, verbose=True)

    # Training weights minus the frozen heads' terms: lambda_prior sees only
    # the frozen lambda, the affine terms only the frozen affine — constants.
    weights = dict(cfg['loss'])
    class_weights = weights.pop('class_weights', None)
    weights['lambda_prior'] = 0.0
    loss_fn = SegRegistrationLoss(weights=weights, class_weights=class_weights)

    # Subject selection: head of val_txt, template subject excluded, input seg
    # required.
    d = cfg['data']
    input_name = d.get('input_seg_filename') or d['seg_filename']
    subject_dirs = [Path(p).parent for p in
                    Path(d['val_txt']).read_text().splitlines() if p.strip()]
    template_subject = Path(d['template_seg_path']).parent.name
    if args.subjects:
        by_name = {sd.name: sd for sd in subject_dirs}
        wanted = [x for x in args.subjects.split(',') if x]
        missing = [x for x in wanted if x not in by_name]
        if missing:
            raise SystemExit(f"[probe] subjects not in {d['val_txt']}: {missing}")
        chosen = [by_name[x] for x in wanted]
    else:
        chosen = [sd for sd in subject_dirs
                  if sd.name != template_subject
                  and (sd / input_name).is_file()][:args.num_subjects]
    if not chosen:
        raise SystemExit('[probe] no subjects to probe')
    print(f'[probe] {len(subject_dirs)} val subjects | probing '
          f'{[sd.name for sd in chosen]} | input = {input_name} | '
          f"supervision = {d['seg_filename']}", flush=True)

    arms = ['baseline'] + [a for a, _, _ in opt_arms]
    subjects = {}
    for i, sd in enumerate(chosen):
        print(f'[probe] === subject {i + 1}/{len(chosen)}: {sd.name} ===',
              flush=True)
        subjects[sd.name] = probe_subject(sd.name, sd, ctx, loss_fn, args, out,
                                          opt_arms)
        if 'cuda' in args.device:
            torch.cuda.empty_cache()

    arm_means = {arm: {k: float(np.mean([subjects[n][arm]['final'][k]
                                         for n in subjects]))
                       for k in METRIC_KEYS}
                 for arm in arms}
    metrics = {'probe': {**vars(args), 'checkpoint': ckpt, 'affine': use_affine,
                         'coarse_levels': levels, 'arms': arms,
                         'iters': iters_list,
                         'loss_weights': {k: float(v) for k, v in weights.items()}},
               'n_subjects': len(chosen),
               'arm_means': arm_means,
               'subjects': subjects}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    base_mean = arm_means['baseline']['wm_dice']
    print(f'[probe] replay sanity: baseline mean WM Dice vs GT = {base_mean:.4f} '
          f'— reference: ckpt 20260801_083414 measured ~0.844 on the standard '
          f'3 subjects; a large gap means the replay path is broken and these '
          f'numbers are void', flush=True)
    for arm in arms:
        m = arm_means[arm]
        print(f"[probe] mean {arm:>9s}: WM {m['wm_dice']:.4f} | "
              f"edge {m['edge_dice']:.4f} | msd {m['msd_vox']:.3f} vox | "
              f"hd95 {m['hd95_vox']:.3f} vox | fold {m['folding_pct']:.4f}% | "
              f"inv res {m['inverse_residual_vox']:.4f} vox", flush=True)
    print(f"[probe] metrics -> {out / 'metrics.json'}")


if __name__ == '__main__':
    main()
