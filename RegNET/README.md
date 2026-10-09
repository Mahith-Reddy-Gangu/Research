# RegNET

Deformable registration models for brain imaging. Two self-contained subprojects,
each with its own model, losses, affine pre-alignment, training loop, and config.

| Folder       | Input modality | Description |
|--------------|----------------|-------------|
| **M-RegNET** | MRI            | Registers a sample MRI to a fixed template MRI via a dense deformation field; tissue-class segmentations guide registration through attention. See `M-RegNET/README.md` and `M-RegNET/docs/handoff.md`. |
| **S-RegNET** | Segmentation   | Segmentation-map-only registration (5-channel one-hot template + sample) with a vanilla UNet (flow + lambda heads) and optional affine pre-alignment. |

Each subproject reads paths and hyperparameters from its own `config.yaml`.
