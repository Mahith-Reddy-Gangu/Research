"""
Segmentation Dataset for Registration

Loads:
- Template segmentation (5 channels, one-hot)
- Sample segmentation  (5 channels, one-hot)

Resizes to a fixed target_size using mode='nearest' so the seg stays
strictly one-hot. No spatial augmentation here — misalignment between
template and sample is the registration signal.

File layout expected (per subject directory):
    seg4_onehot.npy  — shape (5, D, H, W) one-hot uint8/float

Configuration:
    Reads config.yaml for default paths and target_size when run as a
    script. The Dataset class itself takes explicit arguments.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
import yaml
from pathlib import Path


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None


class SegDataset(Dataset):
    """
    Dataset for segmentation-only registration.

    Returns dict with:
        - template_seg: (5, D, H, W) one-hot (shared across all samples)
        - sample_seg:   (5, D, H, W) one-hot
    """

    def __init__(
        self,
        data_list_file: str,
        template_seg_path: str,
        target_size=(128, 128, 128),
        seg_filename="seg4_onehot.npy",
        preload=True,
    ):
        """
        Args:
            data_list_file: text file with one path per line; each path
                points at a subject's seg .npy. Subject dir is inferred
                via os.path.dirname.
            template_seg_path: absolute path to the template seg .npy.
            target_size: (D, H, W) resize target.
            seg_filename: filename inside each subject dir.
            preload: if True, load every subject's seg into RAM at init
                to eliminate per-epoch I/O on slow filesystems.
        """
        with open(data_list_file, 'r') as f:
            seg_paths = f.read().splitlines()

        self.subject_dirs = [os.path.dirname(p) for p in seg_paths]
        self.seg_filename = seg_filename
        self.target_size = target_size

        self.template_seg = self._load_seg(template_seg_path, target_size)

        self._seg_cache = None
        if preload:
            self._preload_all()

    def _preload_all(self):
        """Load every sample seg into RAM once."""
        from tqdm import tqdm
        n = len(self.subject_dirs)
        print(f"Preloading {n} seg volumes into RAM...")
        self._seg_cache = []
        for subject_dir in tqdm(self.subject_dirs, desc="Preloading", ncols=80):
            seg = self._load_seg(os.path.join(subject_dir, self.seg_filename), self.target_size)
            self._seg_cache.append(seg)
        print(f"Preloading complete. RAM cached {n} seg volumes.")

    def _load_seg(self, path, target_size):
        """Load and preprocess a one-hot seg volume."""
        seg = np.load(path)
        seg = torch.tensor(seg, dtype=torch.float32)

        seg = F.interpolate(
            seg.unsqueeze(0),
            size=target_size,
            mode='nearest',
        ).squeeze(0)

        # One-hot integrity check. Valid only because every load/warp
        # path here uses mode='nearest'; switching to bilinear/trilinear
        # anywhere upstream would make voxels sum to <1 and trip this.
        assert torch.all(torch.sum(seg, dim=0) == 1), f"Invalid one-hot at {path}"
        return seg

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        if self._seg_cache is not None:
            sample_seg = self._seg_cache[idx]
        else:
            subject_dir = self.subject_dirs[idx]
            sample_seg = self._load_seg(os.path.join(subject_dir, self.seg_filename),
                                        self.target_size)

        return {
            'template_seg': self.template_seg,
            'sample_seg': sample_seg,
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    config = load_config()

    if config is None:
        raise Exception("Config file not found. Check config.yaml.")

    data_list = config['data']['train_txt']
    template = config['data']['template_seg_path']
    target_size = tuple(config['model']['target_size'])

    if os.path.exists(data_list) and os.path.exists(template):
        print(f"Loading dataset from config.yaml...")
        print(f"  Data list: {data_list}")
        print(f"  Template:  {template}")
        print(f"  Target size: {target_size}")
        print()

        dataset = SegDataset(data_list, template, target_size=target_size, preload=False)
        print(f"Dataset size: {len(dataset)}")

        sample = dataset[0]
        print(f"Template seg shape: {sample['template_seg'].shape}")
        print(f"Sample seg shape:   {sample['sample_seg'].shape}")
        print(f"Sample seg sum-per-voxel range: "
              f"[{sample['sample_seg'].sum(0).min().item()}, "
              f"{sample['sample_seg'].sum(0).max().item()}] (should be 1.0)")
    else:
        print("Test data not found. Check paths in config.yaml.")
