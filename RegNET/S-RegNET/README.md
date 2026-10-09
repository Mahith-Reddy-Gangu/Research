# S-RegNET — topology-correct white-matter surfaces from a SynthSeg segmentation

A user gives us a white-matter segmentation from SynthSeg. It is topologically
dirty: extra handles, islands, holes. We never build a mesh from it. Instead we
keep one clean template (OASIS 0406, white surface repaired to zero
self-intersections) and deform the template's segmentation until it matches the
upload. The template mesh is then carried through the inverse of that
deformation into the subject's space. The output is the subject's white-matter
surface with the template's topology. Everything runs on a 128³ grid (~2 mm).

The shared model is **distilled net + 50 refinement steps**:

1. One forward pass of the distilled network → a velocity field `v`.
2. 50 Adam steps fitting a coarse 64³ correction `δ` to this one subject → `v + up(δ)`. ~20 s on one GPU.
3. Push the template mesh with `exp(−v)`; warp the template seg with `exp(+v)`.

| Method (n=10 val, SynthSeg in, scored vs GT) | WM Dice | self-int. faces | surface dist | cost/subject |
|---|---|---|---|---|
| Old SVF net (exp6), forward pass | 0.856 | 0.209 % | 1.145 mm | < 1 s |
| Distilled net, forward pass | 0.842 | 0.022 % | 1.038 mm | < 1 s |
| **Distilled net + 50 steps** | **0.877** | **0.063 %** | **0.973 mm** | ~20 s |

## 1. Setup

On the lab cluster everything is in place:

```bash
source ~/venv/bin/activate            # torch 2.8 + cu128, scipy, nibabel, pyyaml, matplotlib
cd ~/code/seg-seg-reg/RegNET/S-RegNET # run every command from this directory
```

Elsewhere: `pip install torch numpy scipy nibabel matplotlib pyyaml tqdm scikit-image`
(scikit-image only for the topology tools).

`config.yaml` is the single source of truth (paths, sizes, loss weights).
Edit it; there are no training CLI flags. Diagnostic scripts take
`--checkpoint/--model`, `--config`, `--output_dir`, `--device`.

Rules that will save you a day:

- **Write every output under your own scratch** (`~/shared_scratch/...` =
  `/shared/scratch/0/home/<you>/`). The plain home quota is tiny and a full
  home breaks git and the file sync.
- **Commit before a real run.** Every script stamps `GIT_SHA.txt` into its
  output dir and flags a dirty tree. A dirty run cannot be reproduced.
- `nvidia-smi` first; the box is shared. Pass `--device cuda:N` explicitly.

## 2. Data

Root: `OASIS=/shared/scratch/0/home/v_nishchay_nilabh/oasis_data`. One
directory per subject:

```
$OASIS/scans/OASIS_OAS1_0146_MR1/
    seg35.nii.gz             35-label aseg map — the source the 5-class GT is built from
    seg4_white.nii.gz        GT 5-class seg (0 bg, 1 cortex, 2 other brain,
                             3 white-surface interior, 4 CSF)              ← convert_one_hot.py
    seg4_white_onehot.npy    (5, D, H, W) uint8 — the SUPERVISION target   ← convert_one_hot.py
    synthseg_white_onehot_v1.npy  (5, D, H, W) uint8 — the network INPUT   ← convert_synthseg_one_hot.py
    distill_vel_v1.npy       (3, 128, 128, 128) float32 — distillation target (train subjects only)
$OASIS/meshes/OASIS_OAS1_0146_MR1/{lh,rh}.white.surf     FreeSurfer white surfaces (eval only)
$OASIS/meshes/OASIS_OAS1_0406_MR1/repaired/template_wm_mesh_repaired.npz   the template mesh
$OASIS/train.txt, val.txt    one seg .npy path per line (330 / 82 subjects); only
                             the subject directory is used, so the filename is stale but harmless
```

Regenerate from raw if you bring new subjects (each step writes next to the
subject's files; the list files only need to name the subject directories):

```bash
# GT: seg35 -> 5-class map + one-hot (class 3 = the white-surface interior)
python convert_one_hot.py --lists $OASIS/train.txt $OASIS/val.txt

# SynthSeg (FreeSurfer labels) -> the SAME 5 classes; label_mapping.py holds the remap
python convert_synthseg_one_hot.py --lists $OASIS/train.txt $OASIS/val.txt \
    --synthseg_dir $OASIS/oasis_synthseg/oasis_data_synthseg_version1 \
    --synthseg_name norm_synthseg.nii.gz --out_name synthseg_white_onehot_v1.npy

# template mesh: FreeSurfer's 0406 ?h.white self-intersects in 660 faces; this moves
# vertices (never faces) until zero cross, and writes template_wm_mesh_repaired.npz
python utils/repair_template_mesh.py --config config.yaml
```

The template subject (0406) must not appear in train/val; the eval scripts skip
it anyway. `template_seg_path` and `template_wm_mesh_path` in `config.yaml`
point at it.

## 3. Run the shared model

Checkpoints (`RUNS=/shared/scratch/0/home/v_nishchay_nilabh/training_results`):

| name | path | head |
|---|---|---|
| distilled (use this) | `$RUNS/distill_v1/20260824_051202/checkpoints/best_model.pth` | `bandlimited` (`bandlimit_opt/config_distill.yaml`) |
| exp6 (old SVF net) | `$RUNS/mahith_experiment/20260801_083414/checkpoints/best_model.pth` | 6-channel (`config.yaml`) |

Always pair a checkpoint with the config it was trained with: the config's
`model.head` decides which network is built.

**Step 1+2 — forward pass + 50 refinement steps** on the first 10 val subjects
(add `--subjects OASIS_OAS1_0146_MR1,...` to pick):

```bash
OUT=~/shared_scratch/distill50_n10
python bandlimit_opt/instance_opt_bandlimited.py \
    --checkpoint $RUNS/distill_v1/20260824_051202 --config bandlimit_opt/config_distill.yaml \
    --coarse_levels 64 --iters 50 --lowpass_init 96 --num_subjects 10 \
    --output_dir $OUT --device cuda:0
```

Arms: `baseline` = forward pass only, `d64` = + 50 steps. Look at
`metrics.json` → `arm_means.d64.wm_dice` (~0.877). `flows/<subj>_d64.npz`
holds the final velocity; `figures/` the contour overlays.

**Step 3 — push the template mesh and score it:**

```bash
python utils/push_optimized_mesh.py --probe_dir $OUT --config bandlimit_opt/config_distill.yaml \
    --arms baseline,d64 --output_dir $OUT/mesh_eval --device cuda:0
```

`mesh_metrics.csv` has per subject × arm: `si_faces_pct` (self-intersecting
faces, ~0.06 %), `sym_mean_mm` (distance to the subject's own white surface,
~0.97 mm), `inverse_residual_vox`. `<subj>_d64_deformed.white.surf` is the
deliverable — open it in freeview on the subject's `seg4.nii.gz`.

Optional figures and the topology table (CPU):

```bash
python bandlimit_opt/render_deformed_mesh.py --config bandlimit_opt/config_distill.yaml \
    --rows baseline=$OUT/mesh_eval:baseline,refined=$OUT/mesh_eval:d64 --output_dir $OUT/renders
python utils/topology_before_after.py --mesh_eval_dir $OUT/mesh_eval --arm d64 \
    --config bandlimit_opt/config_distill.yaml --output_dir $OUT/topology
```

Single-file inference (warped seg NIfTI + loss dump, no mesh):
`python inference.py --input_seg <seg.npy> --checkpoint <best_model.pth> --config <its config> --output_dir <dir>`.
Full voxel + mesh sweeps: `utils/evaluate_synthseg_input.py` (deployment
condition) and `utils/evaluate_all.py` (every subject, `--self_int` for topology).

## 4. Train it yourself

Three stages, one GPU each. Stage A is ~1–2 days, B ~8 h, C ~5 h.

**A. Baseline SVF net** — `config.yaml` as committed (SynthSeg in, GT
supervision, affine on, 6-channel head, 200 epochs):

```bash
python train.py                        # -> output.base_dir/<YYYYMMDD_HHMMSS>/{checkpoints,logs}
```

Watch `logs/`: val Dice, `folding_pct`, per-term `loss_components.jsonl`.
exp6 is a run of this stage.

**B. Distillation targets** — run the 300-step band-limited fit once per
training subject with the stage-A checkpoint and save the final velocity
`v* = lp96(v_net) + up(δ*)` as `<subject>/distill_vel_v1.npy`. Resume-safe.

```bash
python bandlimit_opt/generate_distill_targets.py \
    --checkpoint $RUNS/mahith_experiment/20260801_083414 --config config.yaml --device cuda:0
```

**C. Distilled net** — new head, warm start from the stage-A checkpoint, one
extra loss term (`vel_mse`). `resume.checkpoint_path` in the config names the
stage-A checkpoint; the affine branch is frozen automatically.

```bash
python train.py --config bandlimit_opt/config_distill.yaml
```

Every deviation from `config.yaml` is commented inside `config_distill.yaml`.
Evaluate the result with section 3 (val subjects have no targets, so their
numbers are honest).

## 5. Surface + mesh supervision (192³)

The pushed template mesh is scored as a mesh, not as voxels: a symmetric
surface distance plus three mesh regularisers, all built from the subject's
SynthSeg segmentation at deployment (no ground truth needed). At 192³ this is
a fine-tune of the existing net; per subject it is the 50-step refinement's
objective. The tuned 50-step arm takes mesh↔GT symmetric mean 0.91 → 0.66 mm,
HD95 2.26 → 1.74 mm and self-intersecting faces 0.047 % → 0.005 % at n = 10,
for 0.882 → 0.856 WM Dice.

The objective:

```
L(δ) = w_d·Dice + w_ce·CE                                      voxel data term, unchanged
     + w_f ·[ mean|S_in(φ(V))|   + ½·mean top-5% ]               template → subject
     + w_r ·[ mean|S_tpl(φ⁻¹(P))| + ½·mean top-5% ]               subject → template
     + w_e · mean (log |e'|/|e|)²                                 edge-length ratio
     + w_l · mean relu(|Δ'v| − |Δv|)                              Laplacian excess
     + w_n · mean relu((1 − n'a·n'b) − (1 − na·nb))               normal excess
     + bending(δ) + jacobian(δ)                                   velocity terms, unchanged
```

`S_in` is the signed distance to the SynthSeg WM mask (EDT, Gaussian σ 0.75
vox, clamp ±10 mm). `S_tpl` is the precomputed distance to the template's
white surface. `V` is the template vertices, `P` the subject's WM↔cortex
interface points (marching cubes on the smoothed WM mask at 0.5, kept where
the smoothed cortex channel > 0.2). The mesh terms are excess over the
affine-aligned template, so the template's own curvature is free.

**Data prep and training:**

```bash
python utils/make_white_sdf.py --lists train,val --template --workers 32 \
    --output_dir ~/shared_scratch/meshsup_v2/sdf_qc     # ~0.5 GB RAM/worker
python train.py --config config_meshsup.yaml
```

The generator writes `white_sdf_v1.npy` + `white_surf_v1.npz` next to each
subject's segs and the template SDF beside the template mesh (the data is
already on the cluster; it is only re-run for new subjects). The run
warm-starts the 192³ checkpoint weights-only and writes both
`best_model.pth` (val Dice) and `best_surface.pth` (val surface) —
`best_surface.pth` is the one to evaluate.

**Refinement and evaluation:**

```bash
python bandlimit_opt/instance_opt_surface.py --checkpoint <ckpt> --config config_meshsup.yaml \
    --num_subjects 10 --arms dice@50,symM@50 --coarse_levels 96 --lowpass_init 144 \
    --output_dir <out> --device cuda:0
python utils/push_optimized_mesh.py --probe_dir <out> --config config_meshsup.yaml \
    --output_dir <out>/mesh_eval --arms baseline,dice50,symM50 --device cuda:0 \
    --cortex_labels_dir /shared/scratch/0/home/v_nishchay_nilabh/bandlimit_opt/fsaverage_labels
python utils/dir_split.py <out>/mesh_eval --config config_meshsup.yaml
```

`symM`'s defaults are the tuned arm (Dice ×2, surface weights annealed over
25 steps, top-10 % tails, every interface point); `dice` is the deployed
baseline and `dsdf` / `sym` are the one-sided and no-mesh-regulariser
ablations — run with the same tuned knobs as `symM` (Dice ×2, anneal 25,
top-10 % tails, all points), so they are ablations of the tuned arm and not
comparable to older sweep rows; 50 steps is the knee at ~9 s per subject.
`summary.json` carries `sym_mean_mm` / `sym_hd95_mm` / `si_faces_pct`,
`arms.csv` the WM Dice and folding, `dir_split` the template→GT vs
GT→template split.

## 6. How the model works

- **Input**: template one-hot (5 ch) ⊕ subject one-hot (5 ch) = 10 channels.
  An `AffineNet` pre-aligns the template first; a 3D U-Net then predicts a
  stationary velocity field and a per-voxel smoothness weight λ.
- **Warp**: `φ = exp(v)` by scaling and squaring (`v/2⁷`, 7 self-compositions).
  Order is always **affine, then dense flow**; every consumer replays both.
- **Band-limited tied head** (`model.head: bandlimited`, the distilled net):
  `v = lp96(base) + up(residual)`. `base` is a 1×1 conv at 128³ passed through a
  fixed 128→96→128 trilinear resample (erases wavelengths < ~2.7 voxels — the
  band that carried ~95 % of mesh self-intersection). `residual` is a zero-init
  1×1 conv on the 64³ decoder level, upsampled. The reverse field is `−v`, so
  `exp(−v)` — what the mesh rides — is the exact inverse of `exp(+v)`. The old
  6-channel head predicted two independent fields that disagreed by ~4 voxels.
- **Loss** (`config.yaml → loss`): Dice 0.8 + CE 0.2 (both directions),
  bending 0.02, Jacobian anti-fold, displacement 0.01, λ-smoothness and
  λ-prior 0.05, affine regulariser. Distillation adds
  `300 · mean‖v − v*‖²` and lowers Jacobian to 0.005 (the never-trained tied
  inverse starts with a huge raw Jacobian term; the regression is the main
  anti-fold force).
- **Refinement** (`instance_opt_bandlimited.py`): net, affine and λ frozen;
  only a 64³ `δ` is optimised against the *input* (SynthSeg) seg with the
  training loss minus `vel_mse`. 50 steps is the knee: 300 gains 0.005 mm and
  doubles self-intersection.
- **Topology by construction**: the mesh push moves vertices only; faces are
  the template's, so handles/components are the template's for every subject.
  Self-intersection is the one thing that can still fail — that is the metric.

## 7. File map

| file | role |
|---|---|
| `config.yaml` / `bandlimit_opt/config_distill.yaml` | baseline run / distilled run (source of truth) |
| `config_meshsup.yaml` | 192³ surface + mesh supervised fine-tune (§5) |
| `model.py` | `AffineNet`, `UNet`, `BandLimitedHead`, `SegRegistrationNet`, `SpatialTransformer`, warm-start loader |
| `losses.py` | Dice, CE, bending, Jacobian, displacement, λ terms, `vel_mse`, surface + mesh regularisers (§5); `SegRegistrationLoss` |
| `get_data.py` | `SegDataset`: one-hot loading, nearest resize to 128³, optional `target_vel`, white SDF pair (§5) |
| `train.py` | training loop, AMP, warm-up cosine LR, checkpoints, provenance |
| `inference.py` | single-subject inference; `setup_inference` is reused by every tool |
| `convert_one_hot.py`, `convert_synthseg_one_hot.py`, `label_mapping.py` | data generation |
| `utils/make_white_sdf.py` | signed-distance volumes to each white surface — the surface loss's target |
| `bandlimit_opt/instance_opt_bandlimited.py` | the 50-step refinement (and the 300-step fit) |
| `bandlimit_opt/instance_opt_surface.py` | the 50-step refinement with the surface + mesh objective (arms `dice`/`dsdf`/`sym`/`symM`) |
| `bandlimit_opt/generate_distill_targets.py` | distillation targets over `train.txt` |
| `utils/push_optimized_mesh.py` | template mesh → subject space; self-int, flips, mm distance |
| `utils/dir_split.py` | m2g / g2m direction split of the mesh distances |
| `bandlimit_opt/render_deformed_mesh.py` | 3D renders + slice overlays of pushed meshes |
| `utils/topology_before_after.py` | handles/components: SynthSeg surface vs pushed mesh |
| `utils/repair_template_mesh.py` | one-off template surface repair |
| `utils/evaluate_all.py`, `utils/evaluate_synthseg_input.py` | full-split voxel + mesh evaluation |
| `utils/visualize_run.py`, `utils/visualize_mesh.py`, `utils/mesh_flip_probe.py`, `utils/synthseg_topology_baseline.py`, `compute_self_intersection.py`, `compare_synthseg_vs_gt.py`, `random_deform.py` | older diagnostics, still runnable |
| `git_provenance.py` | writes `GIT_SHA.txt` into every run dir |
