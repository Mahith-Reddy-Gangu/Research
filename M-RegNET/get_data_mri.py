"""
MRI Dataset for Registration

Loads:
- Template MRI (1 channel) - grayscale brain scan
- Template Segmentation (5 channels) - one-hot encoded
- Sample MRI (1 channel) - target brain scan
- Sample Segmentation (5 channels) - for evaluation

File structure expected:
    subject_dir/
        brain.npy        - MRI scan (D, H, W) normalized to [0,1]
        seg4_onehot.npy  - Segmentation (5, D, H, W) one-hot

Configuration:
    Uses config.yaml for default paths and augmentation parameters.
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


def detect_and_correct_inversion(mri):
    """
    Detect and correct inverted MRI volumes.

    Brain MRI (any weighting) has a dark background that dominates the
    volume.  After [0,1] normalisation the median intensity should be
    well below 0.5.  If it is above, the image is inverted and we flip
    it back.  The threshold is intentionally generous -- a median above
    0.5 is unambiguous inversion for brain scans.

    Args:
        mri: Tensor of shape (1, D, H, W) in [0, 1].
    Returns:
        Corrected tensor (same shape), bool indicating whether inversion
        was applied.
    """
    median_val = mri.median().item()
    if median_val > 0.5:
        mri = 1.0 - mri
        return mri, True
    return mri, False


class MRIDataset(Dataset):
    """
    Dataset for MRI-based registration with segmentation guidance.

    Returns dict with:
        - template_mri: (1, D, H, W)
        - template_seg: (5, D, H, W)
        - sample_mri: (1, D, H, W)
        - sample_seg: (5, D, H, W)
    """

    def __init__(
        self,
        data_list_file: str,
        template_mri_path: str,
        template_seg_path: str,
        target_size=(128, 128, 128),
        mri_filename="brain.npy",
        seg_filename="seg4_onehot.npy",
        spatial_aug_config=None,
        preload=True,
    ):
        """
        Args:
            data_list_file: Text file with paths to subject segmentation files
            template_mri_path: Path to template MRI .npy file
            template_seg_path: Path to template segmentation .npy file
            target_size: Target volume size
            mri_filename: MRI filename in each subject directory
            seg_filename: Segmentation filename in each subject directory
            preload: If True, preload all volumes into RAM at init (eliminates per-epoch I/O)
        """
        with open(data_list_file, 'r') as f:
            seg_paths = f.read().splitlines()

        self.subject_dirs = [os.path.dirname(p) for p in seg_paths]

        self.mri_filename = mri_filename
        self.seg_filename = seg_filename
        self.target_size = target_size

        # Spatial augmentation (sample only — synthesizes misalignment for AffineNet)
        sa = spatial_aug_config or {}
        self.spatial_augmentation = sa.get('enabled', False)
        self.spatial_rotation_deg = sa.get('rotation_deg', 10.0)
        self.spatial_translation_voxels = sa.get('translation_voxels', 10)
        self.spatial_scale_range = sa.get('scale_range', [1.0, 1.0])  # per-axis; 1.1 = 10% larger anatomy
        self.spatial_elastic_control_points = sa.get('elastic_control_points', 4)
        self.spatial_elastic_magnitude_voxels = sa.get('elastic_magnitude_voxels', 3.0)

        self.template_mri = self._load_mri(template_mri_path, target_size)
        self.template_seg = self._load_seg(template_seg_path, target_size)

        # Preload all subject volumes into RAM to eliminate per-batch NFS I/O
        self._mri_cache = None
        self._seg_cache = None
        if preload:
            self._preload_all()

    def _preload_all(self):
        """Load all subject volumes into RAM once to eliminate per-epoch NFS I/O."""
        from tqdm import tqdm
        n = len(self.subject_dirs)
        print(f"Preloading {n} subjects into RAM...")
        self._mri_cache = []
        self._seg_cache = []
        for subject_dir in tqdm(self.subject_dirs, desc="Preloading", ncols=80):
            mri = self._load_mri(os.path.join(subject_dir, self.mri_filename), self.target_size)
            seg = self._load_seg(os.path.join(subject_dir, self.seg_filename), self.target_size)
            self._mri_cache.append(mri)
            self._seg_cache.append(seg)
        print(f"Preloading complete. RAM cached {n} MRI + {n} seg volumes.")

    def _load_mri(self, path, target_size):
        """Load and preprocess MRI volume."""
        mri = np.load(path)
        mri = torch.tensor(mri, dtype=torch.float32)
        
        if mri.ndim == 3:
            mri = mri.unsqueeze(0)
        
        mri_min, mri_max = mri.min(), mri.max()
        if mri_max - mri_min > 0:
            mri = (mri - mri_min) / (mri_max - mri_min)
        
        mri = F.interpolate(
            mri.unsqueeze(0),
            size=target_size,
            mode='trilinear',
            align_corners=False
        ).squeeze(0)
        
        mri, _ = detect_and_correct_inversion(mri)
        return mri

    def _load_seg(self, path, target_size):
        """Load and preprocess segmentation volume."""
        seg = np.load(path)
        seg = torch.tensor(seg, dtype=torch.float32)

        seg = F.interpolate(
            seg.unsqueeze(0),
            size=target_size,
            mode='nearest'
        ).squeeze(0)

        return seg

    def _augment_spatial(self, mri, seg):
        """Apply identical random rotation + translation + elastic warp to mri and seg."""
        D, H, W = self.target_size

        rot_max = self.spatial_rotation_deg * np.pi / 180.0
        ax, ay, az = (torch.rand(3) * 2 - 1).tolist()
        ax, ay, az = ax * rot_max, ay * rot_max, az * rot_max

        cx, sx = np.cos(ax), np.sin(ax)
        cy, sy = np.cos(ay), np.sin(ay)
        cz, sz = np.cos(az), np.sin(az)
        Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
        Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
        Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)
        R = Rz @ Ry @ Rx

        # Anisotropic scale: config value 1.1 means "anatomy 10% larger along that axis".
        # affine_grid maps output->input coords, so theta needs the inverse (1/s).
        sl, sh = self.spatial_scale_range
        s = torch.rand(3) * (sh - sl) + sl
        S = torch.diag(1.0 / s.to(torch.float32))
        A = R @ S

        tv = self.spatial_translation_voxels
        t_vox = (torch.rand(3) * 2 - 1) * tv
        t_norm = t_vox * 2.0 / torch.tensor([D, H, W], dtype=torch.float32)

        theta = torch.cat([A, t_norm.unsqueeze(1)], dim=1).unsqueeze(0)  # (1, 3, 4)

        target_shape = (1, 1, D, H, W)
        grid = F.affine_grid(theta, target_shape, align_corners=False)  # (1, D, H, W, 3)

        k = self.spatial_elastic_control_points
        mag = self.spatial_elastic_magnitude_voxels
        if k > 0 and mag > 0:
            low = torch.randn(1, 3, k, k, k) * mag
            up = F.interpolate(low, size=(D, H, W), mode='trilinear', align_corners=False)
            elastic = up.permute(0, 2, 3, 4, 1)  # (1, D, H, W, 3)
            scale = torch.tensor([2.0 / W, 2.0 / H, 2.0 / D], dtype=torch.float32)
            grid = grid + elastic * scale

        mri_w = F.grid_sample(mri.unsqueeze(0), grid, mode='bilinear',
                              padding_mode='border', align_corners=False).squeeze(0)
        seg_w = F.grid_sample(seg.unsqueeze(0), grid, mode='nearest',
                              padding_mode='border', align_corners=False).squeeze(0)
        return mri_w, seg_w

    def __len__(self):
        return len(self.subject_dirs)
    
    def __getitem__(self, idx):
        if self._mri_cache is not None:
            sample_mri = self._mri_cache[idx]
            sample_seg = self._seg_cache[idx]
        else:
            subject_dir = self.subject_dirs[idx]
            sample_mri = self._load_mri(os.path.join(subject_dir, self.mri_filename), self.target_size)
            sample_seg = self._load_seg(os.path.join(subject_dir, self.seg_filename), self.target_size)

        template_mri = self.template_mri

        if self.spatial_augmentation:
            sample_mri, sample_seg = self._augment_spatial(sample_mri, sample_seg)

        return {
            'template_mri': template_mri,
            'template_seg': self.template_seg,
            'sample_mri': sample_mri,
            'sample_seg': sample_seg,
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Load config for test paths
    config = load_config()
    
    if config is not None:
        data_list = config['data']['train_txt']
        template_mri = config['data']['template_mri_path']
        template_seg = config['data']['template_seg_path']
        target_size = tuple(config['model']['target_size'])
    else:
        raise Exception("Config file not found. Check paths in config.yaml or run convert_brain_mri.py first.")
    
    if os.path.exists(data_list) and os.path.exists(template_mri):
        print(f"Loading dataset from config.yaml...")
        print(f"  Data list: {data_list}")
        print(f"  Template MRI: {template_mri}")
        print(f"  Template Seg: {template_seg}")
        print(f"  Target size: {target_size}")
        print()
        
        dataset = MRIDataset(data_list, template_mri, template_seg, target_size=target_size)
        print(f"Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        print(f"Template MRI shape: {sample['template_mri'].shape}")
        print(f"Template Seg shape: {sample['template_seg'].shape}")
        print(f"Sample MRI shape: {sample['sample_mri'].shape}")
        print(f"Sample Seg shape: {sample['sample_seg'].shape}")
    else:
        print("Test data not found. Check paths in config.yaml or run convert_brain_mri.py first.")