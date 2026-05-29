# Unsupervised Segmentation Map Registration

This repository implements an unsupervised deep learning framework for registering a surface-derived segmentation template to a target segmentation volume. The model learns a 3D deformation field using volumetric similarity and smoothness constraints, without requiring paired training data. It supports multiple surface-aware loss functions and operates on medical datasets such as OASIS.

---

**Dataset**

We use the [Neurite-OASIS](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) brain MRI dataset. The `.npz` files are preprocessed into one-hot encoded `.npy` volumes using `convert_one_hot.py`. Each one-hot volume has **5 channels**, corresponding to:  
- Background  
- Cortex  
- Subcortical GM  
- White Matter  
- CSF

---

**Setup**

```bash
git clone https://github.com/karthiknm/Segmentation-Map-Registration.git
cd Segmentation-Map-Registration
pip install -r requirements.txt  # or install dependencies manually
```

---

**Training**

All paths and hyperparameters live in `config.yaml`; edit it rather than passing flags.

```bash
python train.py                      # uses config.yaml in this directory
python train.py --config my.yaml     # use a custom config
```

The input to the model is constructed by concatenating the 5-channel template and 5-channel fixed map, resulting in a **10-channel input**. The U-Net is initialized with `in_channels=10`. Affine pre-alignment (`AffineNet`) is optional via `affine.enabled` in the config; when on, the warp is applied affine-first, then dense flow.

---

**File Overview**

- `train.py`: Main training loop. Loads `config.yaml`, builds the model + `SpatialTransformer`, trains with validation, and saves checkpoints.
- `model.py`: Network architecture — `AffineNet`, `UNet`, `SegRegistrationNet`, and a `SpatialTransformer` (trilinear sampling for the warp; the seg itself is resized with nearest-neighbor in the dataset).
- `losses.py`: Atomic losses (`dice_loss`, `cross_entropy_loss`, `bending_energy_loss`, `jacobian_det_loss`, `displacement_loss`, lambda-adaptive smoothness, affine regularizers) plus the `SegRegistrationLoss` module that combines them.
- `get_data.py`: Loads `.npy` one-hot maps and resizes them to `(128, 128, 128)` using **nearest-neighbor interpolation**.
- `convert_one_hot.py`: Converts `.nii.gz` segmentations into 5-channel `.npy` format.
- `inference.py` / `compute_self_intersection.py` / `compare_synthseg_vs_gt.py`: standalone evaluation and diagnostic tools (run manually).

---

**Loss Functions Used**

`train.py` uses the `SegRegistrationLoss` module from `losses.py`, weighted by `loss_weights` in `config.yaml`. It combines:
- Dice Loss
- Cross Entropy Loss
- Bending Energy Loss
- Jacobian Determinant Loss (prevents folding)
- Displacement and lambda-adaptive smoothness
- Affine regularizers (only when `affine.enabled`)

---

**Notes**

- One fixed template is warped to match each subject's segmentation.
- The `SpatialTransformer` performs dense warping using bilinear interpolation on tensors in normalized coordinates.
- Label-preserving nearest-neighbor interpolation is used only during dataset resizing (`get_data.py`).
- Be sure to adjust paths and GPU IDs in `train.py` as needed.

---

**Logging**

Training metrics are written to a timestamped run directory under `output.base_dir`
(`checkpoints/` and `logs/`), and each run records its git commit via `git_provenance.py`.

---

**Example Command**

```bash
# Point config.yaml at your train/val lists and template, then:
python train.py
```

---

