"""
Distillation targets: the band-limited "combo" per-subject optimization run
over the TRAINING set, saving the tied velocity it ends at so the network can
be retrained to predict it in one forward pass.

The combo arm exactly as instance_opt_bandlimited.py runs it (--lowpass_init 96
--coarse_levels 64 --iters 300 --lr 1e-3 --fit_target input, tied): low-pass
the net's forward velocity through a 96^3 grid, then optimize a zero-initialized
64^3 tied delta for 300 Adam steps against the SynthSeg INPUT seg, fp32 with no
autocast, affine matrix and lambda map frozen. Saved per subject:

    v* = lp96(vel_fw) + up(delta*)          (3, 128, 128, 128) float32

at raw UNet scale — before the /2**7 of scaling-and-squaring — so
flow_fw = exp(+v*) and flow_rv = exp(-v*) replay the combo's warp exactly.
Only v* is saved: -v* and both flows are derived from it.

Each target lands next to the subject's segs as <subject_dir>/distill_vel_v1.npy,
reusing the per-subject-filename convention (seg_filename, input_seg_filename)
so the dataset only needs one optional filename. Per-subject QC json (final
loss + terms, WM Dice vs the fit seg and vs GT, folding %), summary.json and
GIT_SHA.txt land in --output_dir.

RESUME-SAFE: a subject whose target .npy AND QC json both exist is skipped, so
the ~8 GPU-hour pass over 330 subjects survives restarts.

Run from the S-RegNET directory (needs torch; GPU cluster):
    python bandlimit_opt/generate_distill_targets.py \\
        --checkpoint 20260801_083414 \\
        --config config.yaml --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))          # repo-root modules, whatever the cwd is
sys.path.insert(0, str(_ROOT / 'utils'))
sys.path.insert(0, str(_HERE))          # instance_opt_bandlimited

from git_provenance import write_git_sha
from inference import load_config, setup_inference
from losses import SegRegistrationLoss, compute_dice_score
from train import _warp_template
from visualize_run import resolve_checkpoint, detect_affine, folding_stats
from instance_opt_bandlimited import (LOG_TERMS, WM, build_velocities, integrate,
                                      load_onehot, loss_and_terms, lowpass, net_init)

TARGET_NAME = 'distill_vel_v1.npy'      # written next to each subject's segs
LOWPASS_INIT = 96                       # lp96 of the net init — the combo's filter
COARSE = 64                             # tied delta grid: content >= 4 voxels
JACOBIAN = 0.005                        # not config.yaml's 0.15: the tied inverse exp(-v)
                                        # was never trained, its raw term (~1445) would
                                        # outweigh dice ~500:1 and stall the fit
ITERS, LR = 300, 1e-3                   # the combo's schedule
LOG_EVERY = 100                         # progress lines within one subject
QC_KEYS = ('final_loss', 'wm_dice_vs_input', 'wm_dice_vs_gt', 'folding_pct')


@torch.no_grad()
def qc_metrics(vel, s):
    """QC of the final tied velocity, on the probe's warp replay (affine first,
    then exp(+v)): WM Dice against the seg the optimization fit and against GT,
    plus folding of the forward flow."""
    flow_fw = integrate(vel, s['stn'])
    warped = _warp_template(s['template_seg'], flow_fw, s['affine'], s['stn'])
    dice_fit, _ = compute_dice_score(warped, s['fit_seg'], s['num_classes'])
    dice_gt, _ = compute_dice_score(warped, s['gt_seg'], s['num_classes'])
    fold_pct, worst_det = folding_stats(flow_fw)
    return {'wm_dice_vs_input': dice_fit[WM], 'wm_dice_vs_gt': dice_gt[WM],
            'folding_pct': fold_pct, 'worst_det': worst_det}


def fit_subject(name, subject_dir, ctx, loss_fn):
    """One combo fit. Returns the tied velocity v* (1, 3, D, H, W) and its QC
    dict. Same state dict `s` as instance_opt_bandlimited's d64 arm, so
    build_velocities/loss_and_terms behave identically here."""
    d = ctx['cfg']['data']
    dev, ts, stn = ctx['device'], ctx['target_size'], ctx['stn']
    input_name = d.get('input_seg_filename') or d['seg_filename']
    input_seg = load_onehot(subject_dir / input_name, ts).to(dev)
    gt_seg = load_onehot(subject_dir / d['seg_filename'], ts).to(dev)

    vel_fw0, vel_rv0, lambda_map, affine = net_init(ctx['model'],
                                                    ctx['template_seg'], input_seg)

    s = {'template_seg': ctx['template_seg'], 'gt_seg': gt_seg,
         # fit_target='input' — the deployment condition; gt_seg only scores QC.
         'fit_seg': input_seg,
         'lambda_map': lambda_map, 'affine': affine, 'stn': stn,
         'loss_fn': loss_fn, 'num_classes': ctx['num_classes'],
         'target_size': ts, 'dev': dev, 'tied': True,
         'vel_fw0': lowpass(vel_fw0, LOWPASS_INIT, ts),
         'vel_rv0': lowpass(vel_rv0, LOWPASS_INIT, ts)}

    delta = torch.zeros(1, 3, COARSE, COARSE, COARSE, device=dev, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=LR)
    for it in range(ITERS):
        opt.zero_grad()
        vel_fw, vel_rv = build_velocities(delta, s)
        total, terms = loss_and_terms(vel_fw, vel_rv, s)
        total.backward()
        opt.step()
        step = it + 1
        if step == 1 or step % LOG_EVERY == 0 or step == ITERS:
            vals = ' | '.join(f'{k} {float(terms[k].detach()):.5f}' for k in LOG_TERMS)
            print(f'[targets] {name} iter {step}/{ITERS} '
                  f'total {float(total.detach()):.5f} | {vals}', flush=True)

    # Drop the graph before the QC forward pass — it is the largest allocation
    # alive at this point and the QC pass needs the room.
    final = {'final_loss': float(total.detach()),
             'final_terms': {k: float(terms[k].detach()) for k in LOG_TERMS}}
    del total, terms

    vel, _ = build_velocities(delta.detach(), s)
    qc = {**final, **qc_metrics(vel, s)}
    del input_seg, gt_seg, vel_fw0, vel_rv0, lambda_map, s
    return vel.detach(), qc


def main():
    ap = argparse.ArgumentParser(
        description='Generate per-subject distillation targets: the band-limited '
                    'combo velocity (lp96 net init + optimized 64^3 tied delta) '
                    'for every train_txt subject')
    ap.add_argument('--checkpoint', required=True,
                    help='run dir, a best_model.pth path, or a run name under output.base_dir')
    ap.add_argument('--config', default='config.yaml',
                    help='config.yaml (a bare name resolves against the S-RegNET dir)')
    ap.add_argument('--output_dir', default=None,
                    help='QC + provenance dir (default: <output.base_dir>/../'
                         'distill_targets_v1); the targets themselves always go '
                         'next to the subject segs')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--num_subjects', type=int, default=0,
                    help='0 = every eligible train_txt subject; >0 takes that '
                         'many from the head (the 2-subject verification)')
    ap.add_argument('--target_name', default=TARGET_NAME,
                    help='per-subject output filename (default: %(default)s); '
                         'the GT-input variant writes distill_vel_gt_v1.npy so '
                         'the SynthSeg targets are never overwritten')
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_file() and not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    args.config = str(cfg_path)

    cfg = load_config(args.config)
    ckpt = resolve_checkpoint(args.checkpoint, cfg)
    use_affine = detect_affine(ckpt)
    out = (Path(args.output_dir).expanduser() if args.output_dir else
           Path(cfg['output']['base_dir']).parent / 'distill_targets_v1')
    if not out.is_absolute():
        out = _ROOT / out
    (out / 'qc').mkdir(parents=True, exist_ok=True)
    write_git_sha(out)

    print(f'[targets] checkpoint = {ckpt}')
    print(f'[targets] device = {args.device} | affine = {use_affine} | '
          f'lowpass_init = {LOWPASS_INIT} | coarse = {COARSE} | iters = {ITERS} | '
          f'lr = {LR} | target = {args.target_name} | output_dir = {out}', flush=True)

    ctx = setup_inference(ckpt, args.config, args.device,
                          use_affine=use_affine, verbose=True)

    # Training weights minus the frozen heads' terms: lambda_prior sees only
    # the frozen lambda, the affine terms only the frozen affine — constants.
    weights = dict(cfg['loss'])
    class_weights = weights.pop('class_weights', None)
    weights['lambda_prior'] = 0.0
    weights['jacobian'] = JACOBIAN
    loss_fn = SegRegistrationLoss(weights=weights, class_weights=class_weights)

    # Same selection rule as the probes, over train_txt instead of val_txt:
    # template subject excluded, input seg required.
    d = cfg['data']
    input_name = d.get('input_seg_filename') or d['seg_filename']
    subject_dirs = [Path(p).parent for p in
                    Path(d['train_txt']).read_text().splitlines() if p.strip()]
    template_subject = Path(d['template_seg_path']).parent.name
    chosen = [sd for sd in subject_dirs
              if sd.name != template_subject and (sd / input_name).is_file()]
    if args.num_subjects:
        chosen = chosen[:args.num_subjects]
    if not chosen:
        raise SystemExit('[targets] no subjects to process')
    print(f'[targets] {len(subject_dirs)} train subjects | {len(chosen)} eligible '
          f'| input = {input_name} | supervision = {d["seg_filename"]}', flush=True)

    for i, sd in enumerate(chosen):
        target_path = sd / args.target_name
        qc_path = out / 'qc' / f'{sd.name}.json'
        if target_path.is_file() and qc_path.is_file():
            continue                    # resume: this subject is already done
        print(f'[targets] === {i + 1}/{len(chosen)}: {sd.name} ===', flush=True)
        vel, qc = fit_subject(sd.name, sd, ctx, loss_fn)
        # .npy first, QC json second — the pair is the resume marker, so a kill
        # between the two costs one recomputed subject and never a stale pair.
        np.save(target_path, vel.squeeze(0).cpu().numpy().astype(np.float32))
        qc_path.write_text(json.dumps(
            {'subject': sd.name, 'target': str(target_path), **qc}, indent=2))
        print(f"[targets] {sd.name}: loss {qc['final_loss']:.5f} | WM vs input "
              f"{qc['wm_dice_vs_input']:.4f} | vs GT {qc['wm_dice_vs_gt']:.4f} | "
              f"fold {qc['folding_pct']:.4f}% -> {target_path}", flush=True)
        del vel
        if 'cuda' in args.device:
            torch.cuda.empty_cache()

    # Means over every QC json in the run dir, i.e. including subjects fitted
    # by an earlier (resumed) invocation.
    qcs = [json.loads(p.read_text()) for p in sorted((out / 'qc').glob('*.json'))]
    means = {k: float(np.mean([q[k] for q in qcs])) for k in QC_KEYS}
    (out / 'summary.json').write_text(json.dumps(
        {'checkpoint': ckpt, 'config': args.config, 'affine': use_affine,
         'lowpass_init': LOWPASS_INIT, 'coarse': COARSE, 'iters': ITERS, 'lr': LR,
         'target_name': args.target_name, 'n_eligible': len(chosen), 'n_targets': len(qcs),
         'qc_means': means,
         'loss_weights': {k: float(v) for k, v in weights.items()}}, indent=2))
    print(f'[targets] {len(qcs)} targets on disk / {len(chosen)} eligible | mean '
          + ' | '.join(f'{k} {v:.4f}' for k, v in means.items()), flush=True)
    print(f"[targets] summary -> {out / 'summary.json'}")


if __name__ == '__main__':
    main()
