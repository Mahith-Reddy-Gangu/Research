"""
Run inference for OASIS_OAS1_{IDX}_MR1, IDX = 0001..0100.
Loads the model + template ONCE, then loops in-process over every sample.
Prints per-sample Mean Dice and overall average at the end.
"""

from pathlib import Path

from inference import setup_inference, run_inference_on_sample

# =============================================================================
# CONFIGURE HERE
# =============================================================================

# CHECKPOINT = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/training_mri_acm_affine_aug/20260516_114138/checkpoints/best_model.pth"
CHECKPOINT = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/training_mri_no_acm/20260524_134154/checkpoints/best_model.pth"
# CHECKPOINT = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/training_mri_acm_fixed/20260426_113359/checkpoints/best_model.pth"
# CHECKPOINT = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/training_mri_acm_affine/20260427_022537/checkpoints/best_model.pth"
# OASIS
# INPUT_MRI_TEMPLATE  = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/scans/OASIS_OAS1_{idx}_MR1/brain.npy"
# INPUT_SEG_TEMPLATE  = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/anna_data/oasis_dataset/oasis_synthseg_output/output/OASIS_OAS1_{idx}_MR1/orig_synthseg.nii.gz"
# OUTPUT_DIR_TEMPLATE = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/anna_data/nishchay_results/OASIS_OAS1_{idx}_MR1"

# FOMO60K - Make sure to put flag = "fomo60k" in inference.py too
INPUT_MRI_TEMPLATE  = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/soham_data/fomo-60k/sub_{idx}/ses_1/t1.nii.gz"
INPUT_SEG_TEMPLATE  = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/anna_data/fomo60k_synthseg_data/output/output/sub_{idx}/ses_1/t1_synthseg.nii.gz"
OUTPUT_DIR_TEMPLATE = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/anna_data/nishchay_results_fomo60k/sub_{idx}/ses_1"

DEVICE = "cuda:5"
USE_AFFINE = True

# =============================================================================

# Load template + model ONCE; reuse across all samples.
ctx = setup_inference(CHECKPOINT, config_path=None, device=DEVICE,
                      use_affine=USE_AFFINE, verbose=True)

dice_scores = {}
avd_scores = {}
folding_scores = {}   # per-sample folding % (voxels with det(J) < 0)
min_dets = {}         # per-sample worst (most-negative) normalized det
skipped = []

# for i in range(1, 100):
#     idx = f"{i:04d}"
#     if i > 10:
#         print("DONE")
#         break
for i in range(1, 41):
    idx = f"{i:01d}"
    input_mri = INPUT_MRI_TEMPLATE.format(idx=idx)
    input_seg = INPUT_SEG_TEMPLATE.format(idx=idx)
    output_dir = OUTPUT_DIR_TEMPLATE.format(idx=idx)

    if not Path(input_mri).exists() or not Path(input_seg).exists():
        print(f"[{idx}] SKIP — input files not found")
        skipped.append(idx)
        continue

    print(f"[{idx}] Running inference...", flush=True)

    try:
        losses = run_inference_on_sample(
            ctx, input_mri, input_seg, output_dir,
            losses_only=False, verbose=False,
        )
    except Exception as e:
        print(f"[{idx}] FAILED: {e}")
        skipped.append(idx)
        continue

    dice = losses.get("dice_score")
    dice_scores[idx] = dice
    print(f"[{idx}] Mean Dice: {dice:.4f}")
    
    avd = losses.get("avg_hausdorff")
    avd_scores[idx] = avd
    print(f"[{idx}] Mean AVD: {avd:.4f}")

    fold = losses.get("folding_pct")
    folding_scores[idx] = fold
    min_dets[idx] = losses.get("min_det")
    print(f"[{idx}] Folding %: {fold:.4f}%  (min det = {min_dets[idx]:.4f})")


# Summary
print("\n" + "=" * 50)
print(f"Processed : {len(dice_scores)} / 100")
print(f"Skipped   : {len(skipped)} {skipped if skipped else ''}")

avg_dice = sum(dice_scores.values()) / len(dice_scores)
best_idx  = max(dice_scores, key=dice_scores.get)
worst_idx = min(dice_scores, key=dice_scores.get)
print(f"Avg Dice  : {avg_dice:.4f}")
print(f"Best      : {best_idx}  ({dice_scores[best_idx]:.4f})")
print(f"Worst     : {worst_idx}  ({dice_scores[worst_idx]:.4f})")

avg_avd = sum(avd_scores.values()) / len(avd_scores)
best_idx  = max(avd_scores, key=avd_scores.get)
worst_idx = min(avd_scores, key=avd_scores.get)
print(f"Avg Dice  : {avg_avd:.4f}")
print(f"Best      : {best_idx}  ({avd_scores[best_idx]:.4f})")
print(f"Worst     : {worst_idx}  ({avd_scores[worst_idx]:.4f})")

avg_folding = sum(folding_scores.values()) / len(folding_scores)
best_idx  = min(folding_scores, key=folding_scores.get)   # least folding = best
worst_idx = max(folding_scores, key=folding_scores.get)   # most folding  = worst
print(f"Avg Folding : {avg_folding:.4f}%")
print(f"Best        : {best_idx}  ({folding_scores[best_idx]:.4f}%)")
print(f"Worst       : {worst_idx}  ({folding_scores[worst_idx]:.4f}%, "
      f"min det = {min_dets[worst_idx]:.4f})")

print("=" * 50)
