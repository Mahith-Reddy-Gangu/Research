"""
FreeSurfer integer-label -> 5-class remapping (background, cortex, other brain,
white matter, CSF), plus the seg35 -> FreeSurfer decode. NumPy only, so the
converters and inference share one definition without heavy deps.

Class 3 is the WHITE-SURFACE INTERIOR, not the WM tissue label: the deliverable
is a genus-0 mesh pushed from the template, so the voxel class the flow is fit to
must be the solid that lh/rh.white bound. Measured per seg35 label against a
ray-parity fill of both surfaces (OASIS_OAS1_0199_MR1),
that solid is cerebral WM (98%) PLUS the lateral ventricles (99.6%), thalamus
(96%), caudate (100%), putamen (98%), pallidum (100%), accumbens (93%), ventral
DC (91%), choroid plexus (93%) and vessel (100%) — they sit inside the WM mass,
so the closed surface encloses them. It EXCLUDES cerebellum and brainstem (0.0%
and 0.3% inside) and hippocampus/amygdala (7.5% / 3.0%), which the old mapping
put in the WM class. Class 3 as defined here scores Dice 0.962 lh / 0.966 rh
against the fill; the residual is the white surface sitting ~half a voxel out on
the cortex side, sub-voxel once resized to 128^3.
"""

import numpy as np

# FreeSurfer/SynthSeg label -> 5-class (0 bg, 1 cortex, 2 other brain,
# 3 white-surface interior, 4 CSF). Unlisted labels fall through to background.
LABEL_MAPPING = {
    0: 0, 24: 0,
    3: 1, 42: 1,
    7: 2, 46: 2, 8: 2, 47: 2, 16: 2, 17: 2, 53: 2, 18: 2, 54: 2,
    2: 3, 41: 3, 4: 3, 43: 3, 5: 3, 44: 3, 10: 3, 49: 3, 11: 3, 50: 3,
    12: 3, 51: 3, 13: 3, 52: 3, 26: 3, 58: 3, 28: 3, 60: 3, 30: 3, 62: 3,
    31: 3, 63: 3,
    14: 4, 15: 4,
}

# seg35.nii.gz holds the aseg structures sorted by FreeSurfer label and
# renumbered 1..35 with 24 (CSF) dropped — 1..19 left+midline, 20..35 right.
# Verified on OASIS_OAS1_0199_MR1: mirrored voxel counts, centroid laterality,
# and the seg4.nii.gz class totals reproduced exactly from these groupings.
SEG35_TO_FS = dict(zip(
    range(1, 36),
    [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28, 30, 31,
     41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63]))


def remap_to_5class(int_volume):
    """Map a FreeSurfer-label volume to 0..4 (unlisted -> background). Returns a
    new array, same shape and dtype as int_volume."""
    remapped = np.zeros_like(int_volume)
    for src, dst in LABEL_MAPPING.items():
        remapped[int_volume == src] = dst
    return remapped


def remap_seg35_to_5class(seg35_volume):
    """Same, for a seg35-indexed volume (OASIS scan dirs' seg35.nii.gz)."""
    remapped = np.zeros_like(seg35_volume)
    for src, fs in SEG35_TO_FS.items():
        remapped[seg35_volume == src] = LABEL_MAPPING.get(fs, 0)
    return remapped
