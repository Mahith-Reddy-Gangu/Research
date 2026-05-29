"""
Compare seg-only registration quality + topology against OASIS GT.

For each OASIS subject in INDEX_RANGE we:
  1. Score the raw SynthSeg seg against the GT (Dice, AHD, per-class
     connected-component count).
  2. Register the template to the SynthSeg seg; score the warped template
     against the GT (full loss set incl. Dice, AHD, folding %, regs);
     report per-class component count of the warp.
  3. Write a per-subject directory under OUTPUT_DIR with
        comparison.png       — 3 views × 4 cols (Template / SynthSeg /
                                Warped / GT), per-column Dice + components
        deformation_field.png
        warped_seg.nii.gz
        losses.json          — the model's full metric set vs GT
  4. Write {OUTPUT_DIR}/summary.json with per-sample metrics +
     template_components + aggregate means.

Folding % is meaningful only for a deformation field.  For the static
segs (SynthSeg, GT, warp) we use per-class 3D 26-connectivity component
count as a topology proxy: clean anatomy has 1–2 components per class;
SynthSeg often produces dozens of spurious islands that a diffeomorphic
warp of a clean template cannot introduce.
"""

from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as ndi_label

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from inference import (
    setup_inference, load_seg_input,
    compute_all_losses, compute_avg_hausdorff_distance,
    plot_deformation_field, save_warped_nifti,
    CLASS_COLORS,
)
from losses import compute_dice_score

# =============================================================================
# CONFIGURE HERE
# =============================================================================

# training_seg_acm/20260528_115247
CHECKPOINT = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/training_seg_acm/20260528_115247/checkpoints/best_model.pth"

OASIS_SCANS_DIR = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans"
SYNTHSEG_DIR    = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/anna_data/oasis_dataset/oasis_synthseg_output/output"

# Outputs land next to the checkpoint, under .../<TIMESTAMP>/compare_synthseg_vs_gt_results/
OUTPUT_DIR = Path(CHECKPOINT).parent.parent / "compare_synthseg_vs_gt_results"

# Subject IDs come from val.txt (one absolute path to seg4_onehot.npy per line,
# in shuffled order — preserved here). Parse the subject from the path.
VAL_TXT = Path(__file__).parent / "val.txt"
DEVICE = "cuda:5"
USE_AFFINE = False    # must match the checkpoint


def load_val_indices(val_txt):
    """Return the subject ids (e.g. '0146') in val.txt order. Each line points
    at .../OASIS_OAS1_<idx>_MR1/seg4_onehot.npy."""
    ids = []
    for line in Path(val_txt).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        subject = Path(line).parent.name              # OASIS_OAS1_0146_MR1
        ids.append(subject.split('_')[2])             # 0146
    return ids

# =============================================================================


def per_class_components(seg_chw, num_classes):
    """3D 26-connectivity component count per foreground class. Input is
    a (C, D, H, W) one-hot tensor."""
    labels = seg_chw.argmax(dim=0).cpu().numpy()
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    counts = []
    for c in range(1, num_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            counts.append(0)
            continue
        _, n = ndi_label(mask, structure=structure)
        counts.append(int(n))
    return counts


def plot_synthseg_comparison(template_seg, synthseg, warped_seg, gt_seg,
                             ss_dice, m_dice,
                             ss_components, w_components, gt_components,
                             save_path):
    """3 views (axial/coronal/sagittal) × 4 cols
    (Template / SynthSeg input / Warped Template / GT).  Per-column header
    carries the Dice (vs GT) + component count so the figure tells the
    SynthSeg-topology-vs-clean-warp story without consulting the JSON."""
    t_lab  = template_seg.squeeze().argmax(dim=0).cpu().numpy()
    ss_lab = synthseg.squeeze().argmax(dim=0).cpu().numpy()
    w_lab  = warped_seg.squeeze().argmax(dim=0).cpu().numpy()
    gt_lab = gt_seg.squeeze().argmax(dim=0).cpu().numpy()
    D, H, W = t_lab.shape

    views = [
        ('Axial',    lambda v: v[D // 2, :, :]),
        ('Coronal',  lambda v: v[:, H // 2, :]),
        ('Sagittal', lambda v: v[:, :, W // 2]),
    ]
    cols = [
        ('Template',                                                t_lab),
        (f'SynthSeg\nDice={ss_dice:.3f}  comp={ss_components}',     ss_lab),
        (f'Warped Template\nDice={m_dice:.3f}  comp={w_components}', w_lab),
        (f'GT\ncomp={gt_components}',                               gt_lab),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    for row_idx, (view_name, getter) in enumerate(views):
        for col_idx, (col_title, labels) in enumerate(cols):
            ax = axes[row_idx, col_idx]
            ax.imshow(CLASS_COLORS[getter(labels)], aspect='equal')
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(col_title, fontsize=11)
            if col_idx == 0:
                ax.text(-0.05, 0.5, view_name, transform=ax.transAxes,
                        rotation=90, ha='right', va='center',
                        fontsize=12, fontweight='bold')

    fig.suptitle('SynthSeg vs Warped Template vs Ground Truth',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def main():
    ctx = setup_inference(CHECKPOINT, config_path=None, device=DEVICE,
                          use_affine=USE_AFFINE, verbose=True)

    target_size  = ctx['target_size']
    num_classes  = ctx['num_classes']
    device       = ctx['device']
    template_seg = ctx['template_seg']
    model        = ctx['model']
    stn          = ctx['stn']

    template_components = per_class_components(template_seg.squeeze(0), num_classes)
    print(f"\nTemplate components per fg class (1..{num_classes - 1}): "
          f"{template_components}\n")

    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    indices = load_val_indices(VAL_TXT)
    print(f"Loaded {len(indices)} subject ids from {VAL_TXT}\n")

    results = []
    skipped = []

    for idx in indices:
        synthseg_path = f"{SYNTHSEG_DIR}/OASIS_OAS1_{idx}_MR1/orig_synthseg.nii.gz"
        gt_path       = f"{OASIS_SCANS_DIR}/OASIS_OAS1_{idx}_MR1/seg4_onehot.npy"

        if not (Path(synthseg_path).exists() and Path(gt_path).exists()):
            print(f"[{idx}] SKIP — missing input")
            skipped.append(idx)
            continue

        print(f"[{idx}] Running...", flush=True)

        synthseg, _            = load_seg_input(synthseg_path, target_size, num_classes)
        gt,       nifti_affine = load_seg_input(gt_path,       target_size, num_classes)
        synthseg_b = synthseg.unsqueeze(0).to(device)
        gt_b       = gt.unsqueeze(0).to(device)

        # SynthSeg vs GT (no model involved).
        ss_dice_pc, ss_dice = compute_dice_score(synthseg_b, gt_b, num_classes)
        ss_ahd_pc,  ss_ahd  = compute_avg_hausdorff_distance(synthseg_b, gt_b, num_classes)
        ss_components = per_class_components(synthseg, num_classes)
        gt_components = per_class_components(gt,       num_classes)

        # Model: register template → synthseg, then score warped vs GT.
        final_flow, lambda_map, affine_matrix = model(template_seg, synthseg_b)
        if affine_matrix is not None:
            affine_grid = F.affine_grid(affine_matrix, template_seg.size(),
                                        align_corners=False)
            aligned_seg = F.grid_sample(template_seg, affine_grid, mode='bilinear',
                                        padding_mode='zeros', align_corners=False)
            warped_seg = stn(aligned_seg, final_flow)
        else:
            warped_seg = stn(template_seg, final_flow)

        model_losses = compute_all_losses(
            warped_seg, gt_b, final_flow,
            lambda_map, affine_matrix, num_classes=num_classes,
        )
        warped_components = per_class_components(warped_seg.squeeze(0), num_classes)

        # Per-subject outputs.
        sub_dir = output_root / f"OASIS_OAS1_{idx}_MR1"
        sub_dir.mkdir(parents=True, exist_ok=True)

        plot_synthseg_comparison(
            template_seg, synthseg_b, warped_seg, gt_b,
            ss_dice=ss_dice, m_dice=model_losses['dice_score'],
            ss_components=ss_components, w_components=warped_components,
            gt_components=gt_components,
            save_path=sub_dir / 'comparison.png',
        )
        plot_deformation_field(final_flow.squeeze(0).cpu().numpy(),
                               sub_dir / 'deformation_field.png')
        save_warped_nifti(warped_seg, nifti_affine,
                          str(sub_dir / 'warped_seg.nii.gz'))
        with open(sub_dir / 'losses.json', 'w') as f:
            json.dump(model_losses, f, indent=2)

        r = {
            'idx': idx,
            'synthseg_dice_vs_gt':     ss_dice,
            'synthseg_dice_per_class': ss_dice_pc,
            'synthseg_ahd_vs_gt':      ss_ahd,
            'synthseg_ahd_per_class':  ss_ahd_pc,
            'synthseg_components':     ss_components,
            'gt_components':           gt_components,
            'model_dice_vs_gt':        model_losses['dice_score'],
            'model_dice_per_class':    model_losses['dice_per_class'],
            'model_ahd_vs_gt':         model_losses['avg_hausdorff'],
            'model_ahd_per_class':     model_losses['avg_hausdorff_per_class'],
            'model_warped_components': warped_components,
            'model_folding_pct':       model_losses['folding_pct'],
            'model_min_det':           model_losses['min_det'],
            'model_fold_voxels':       model_losses['fold_voxels'],
        }
        results.append(r)

        print(f"  SynthSeg vs GT : Dice={ss_dice:.4f}  AHD={ss_ahd:.4f}  "
              f"components={ss_components}")
        print(f"  Model    vs GT : Dice={model_losses['dice_score']:.4f}  "
              f"AHD={model_losses['avg_hausdorff']:.4f}  "
              f"folding={model_losses['folding_pct']:.4f}%  "
              f"(min det={model_losses['min_det']:.4f})  "
              f"warped_components={warped_components}")

        del synthseg_b, gt_b, final_flow, warped_seg
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"Processed : {len(results)} / {len(indices)}")
    if skipped:
        print(f"Skipped   : {skipped}")
    if not results:
        return

    def mean(k):
        vals = [r[k] for r in results if r[k] == r[k]]   # filter NaN
        return float(np.mean(vals)) if vals else float('nan')

    print(f"\nMean SynthSeg Dice (vs GT) : {mean('synthseg_dice_vs_gt'):.4f}")
    print(f"Mean SynthSeg AHD  (vs GT) : {mean('synthseg_ahd_vs_gt'):.4f}")
    print(f"Mean Model    Dice (vs GT) : {mean('model_dice_vs_gt'):.4f}")
    print(f"Mean Model    AHD  (vs GT) : {mean('model_ahd_vs_gt'):.4f}")
    print(f"Mean Model    Folding %    : {mean('model_folding_pct'):.4f}%")

    ss_comp = np.array([r['synthseg_components']     for r in results])
    m_comp  = np.array([r['model_warped_components'] for r in results])
    gt_comp = np.array([r['gt_components']           for r in results])
    print(f"\nMean per-class components (lower = cleaner topology):")
    print(f"  GT         : {gt_comp.mean(axis=0).tolist()}")
    print(f"  SynthSeg   : {ss_comp.mean(axis=0).tolist()}")
    print(f"  Model warp : {m_comp.mean(axis=0).tolist()}")
    print(f"  Template   : {template_components}")

    summary_path = output_root / 'summary.json'
    summary_path.write_text(json.dumps({
        'checkpoint': CHECKPOINT,
        'template_components': template_components,
        'per_sample': results,
        'aggregate': {
            'mean_synthseg_dice_vs_gt':     mean('synthseg_dice_vs_gt'),
            'mean_synthseg_ahd_vs_gt':      mean('synthseg_ahd_vs_gt'),
            'mean_model_dice_vs_gt':        mean('model_dice_vs_gt'),
            'mean_model_ahd_vs_gt':         mean('model_ahd_vs_gt'),
            'mean_model_folding_pct':       mean('model_folding_pct'),
            'mean_synthseg_components':     ss_comp.mean(axis=0).tolist(),
            'mean_model_warped_components': m_comp.mean(axis=0).tolist(),
            'mean_gt_components':           gt_comp.mean(axis=0).tolist(),
        },
        'skipped': skipped,
    }, indent=2))
    print(f"\nResults → {summary_path}")


if __name__ == "__main__":
    main()
