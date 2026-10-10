# Self-Supervised Pretraining for S-RegNET

This document provides a comprehensive technical breakdown of the Self-Supervised Learning (SSL) pipeline integrated into S-RegNET. It details the mathematical formulation, implementation constraints, and architectural modifications made to achieve zero-topology-break, high-fidelity deformations.

---

## 1. Conceptual Motivation

Traditionally, image registration models rely on indirect image similarity metrics (like LNCC) or segmentation overlap (Dice) to align subjects. In the S-RegNET architecture, the inputs are strict one-hot 5-class segmentations (SynthSeg). Relying purely on Dice loss for cross-subject alignment has limitations: it cannot perfectly supervise the internal sub-voxel topology of the deformation field.

**The Self-Supervised Solution:** We implement **Equivariance on Synthetic Warps**. By applying a known, mathematically rigorous synthetic deformation to an input image and asking the network to recover that exact deformation, we create a fully self-supervised task. Because we generated the target field ourselves, we have access to the exact Ground Truth (GT) transformation, allowing us to perform direct supervised regression on the velocity field without requiring human annotations.

---

## 2. On-The-Fly Data Generation (`pretrain.py`)

Generating realistic synthetic warps requires strict adherence to the physical constraints of the network's architecture. The target fields are generated on-the-fly during training in `generate_synthetic_batch`.

### 2.1 Band-Limited Velocity Sampling
The S-RegNET model employs a `BandLimitedHead` that hard-limits the spatial frequency of the predicted velocity field to $\frac{3}{4}$ of the grid (e.g., $96^3$ on a $128^3$ grid). 
If our synthetic deformations contain high-frequency noise that the `BandLimitedHead` physically cannot output, the network is trained against an impossible target. 

To solve this, we sample our initial synthetic Stationary Velocity Field (SVF), $v_{coarse}$, at exactly the `lowpass_size` ($96^3$) from a standard normal distribution $\mathcal{N}(0, \sigma)$, and then upsample it to the full $128^3$ resolution via trilinear interpolation. This guarantees the synthetic target perfectly respects the network's theoretical capacity.

### 2.2 Scaling and Squaring Integration
The band-limited $v_{gt}$ is passed through the same 7-step scaling-and-squaring integration used by the network:
$$ \Phi = \text{exp}(v_{gt}) = (v_{gt} / 2^7)^{2^7} $$
This guarantees the resulting deformation flow is diffeomorphic (topology-preserving).

### 2.3 Strict One-Hot Interpolation
PyTorch's Spatial Transformer Networks typically warp volumes using `bilinear` or `trilinear` interpolation. Applying this to one-hot segmentations results in blurred, continuous probabilities (e.g., $0.6$ GM, $0.4$ WM). 
To prevent the network from learning a corrupted data distribution during SSL pretraining, the synthetic target generation explicitly forces `mode='nearest'` inside `F.grid_sample`. This ensures the synthetic supervision targets remain strictly binary (one-hot), exactly mirroring the real SynthSeg input distribution.

---

## 3. The Loss Functions

During pretraining, the model takes the original `SynthSeg` input and the `synthetic_target`, and predicts the deformation required to map between them. The supervision relies on four terms:

### 3.1 Velocity MSE (Primary Signal)
Instead of applying a loss to the final integrated flow field, we regress directly against the pre-integration SVF. 
$$ \mathcal{L}_{MSE} = ||v_{pred} - v_{gt}||^2 $$
**Why this is critical:** Operating in velocity space means the backpropagation gradient completely bypasses the 7 squaring steps. This results in a highly stable, computationally cheap gradient. Furthermore, matching the pre-integration velocity prevents high-frequency errors from hiding within the exponential map smoothing.

### 3.2 Dice Loss (Integration Safety Net)
A minor error in the predicted velocity field compounds exponentially through the scaling-and-squaring steps. A standard cross-entropy / Dice loss is applied between the warped output and the synthetic target to catch accumulated integration drift that velocity MSE might under-penalize.

### 3.3 Jacobian Determinant & Bending Energy (Topology Guards)
Even with MSE on a diffeomorphic target, neural networks can find non-smooth shortcut solutions. We retain the network's native `bending_energy` and `jacobian_det_loss` (a top-K barrier on local folding) during pretraining to aggressively penalize any local topology breaks.

---

## 4. Mesh Quality Optimization (`losses.py`)

The overarching goal is to achieve high-quality, topologically correct White Matter (WM) surface meshes. Folding in the output mesh (measured by `si_faces_pct`) occurs when the local deformation violently pinches, causing adjacent triangles to invert or collapse.

To directly combat this during the mesh-supervised fine-tuning stage, a **Face Area Collapse Penalty** (`mesh_area`) was introduced to the `mesh_regularizers`.
1. It computes the area of every triangle on the WM mesh using cross-products.
2. It compares the pushed area to the reference template area.
3. It applies a harsh ReLU penalty to any triangle whose area drops below 10% of its original reference area.

This physical constraint guarantees the optimizer cannot collapse geometry to minimize local surface distances, directly driving self-intersections toward zero.

---

## 5. Execution Workflow

1. **Pretraining:** Execute the SSL pipeline to learn base deformations natively.
   ```bash
   CUDA_VISIBLE_DEVICES=0 python pretrain.py --config config_pretrain.yaml
   ```
2. **Fine-Tuning:** Inject the pretrained weights (`best_pretrain_model.pth`) into `config_meshsup.yaml` via the `resume` block.
3. **Mesh Supervision:** Run the standard `train.py` using `config_meshsup.yaml` to activate the `mesh_area` penalty and refine the surfaces.
4. **Evaluation:** Use `evaluate_all.py --self_int` to verify Dice improvements and zero topology violations.
