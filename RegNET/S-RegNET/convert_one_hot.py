"""
Convert every listed subject's seg35.nii.gz to the 5-class GT map this project
trains on (0 bg / 1 cortex / 2 other brain / 3 white-surface interior / 4 CSF)
and write both the integer volume and its 5-channel uint8 one-hot next to it.

The shipped seg4.nii.gz is NOT used: its WM label is the WM tissue class
(cerebral + cerebellar WM + brainstem + ventral DC), which is a different solid
from the one lh/rh.white bound. label_mapping.LABEL_MAPPING documents the
measured difference; the .nii.gz is rebuilt from seg35 so that grouping is one
definition shared with the SynthSeg twin, convert_synthseg_one_hot.py.

Both files are written because downstream tools (utils/visualize_mesh.ref_for)
take the .nii.gz sibling of a one-hot .npy as the geometry reference. Native
resolution; SegDataset resizes at load time.

Usage:
    python convert_one_hot.py --lists /path/train.txt /path/val.txt

Each list line is a subject's seg4_onehot.npy path (the train.txt/val.txt
format); only its directory is used.
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
import nibabel as nib
from tqdm import tqdm

from label_mapping import remap_seg35_to_5class

LABELS = [0, 1, 2, 3, 4]


def to_one_hot(seg, labels=LABELS):
    one_hot = np.zeros((len(labels), *seg.shape), dtype=np.uint8)
    for i, label in enumerate(labels):
        one_hot[i] = seg == label
    return one_hot


def convert(subject_dir, seg_name, out_seg_name, out_name):
    img = nib.load(os.path.join(subject_dir, seg_name))
    seg = remap_seg35_to_5class(np.asanyarray(img.dataobj).astype(np.int16))
    nib.save(nib.Nifti1Image(seg.astype(np.uint8), img.affine, img.header),
             os.path.join(subject_dir, out_seg_name))
    np.save(os.path.join(subject_dir, out_name), to_one_hot(seg))


def _convert_one(job):
    convert(*job)


def main():
    ap = argparse.ArgumentParser(description="seg35 .nii.gz -> 5-class map + one-hot .npy")
    ap.add_argument('--lists', nargs='+', required=True,
                    help='train/val txt files; each line = a subject seg .npy path')
    ap.add_argument('--seg_name', default='seg35.nii.gz')
    ap.add_argument('--out_seg_name', default='seg4_white.nii.gz')
    ap.add_argument('--out_name', default='seg4_white_onehot.npy')
    ap.add_argument('--workers', type=int, default=8,
                    help='parallel subjects; the 84 MB one-hot write dominates')
    args = ap.parse_args()

    paths = []
    for lst in args.lists:
        with open(lst) as f:
            paths += [ln.strip() for ln in f if ln.strip()]

    jobs = [(os.path.dirname(p), args.seg_name, args.out_seg_name, args.out_name)
            for p in paths]
    with Pool(args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(_convert_one, jobs), total=len(jobs),
                      desc="Converting GT segs", ncols=80):
            pass


if __name__ == "__main__":
    main()
