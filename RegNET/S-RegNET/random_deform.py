import cornucopia
import monai
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torch
from monai.transforms import EnsureChannelFirst


def show_slices(data, axis='z', index=None, title="Fixed Map", show_all_labels=True):
    """
    Visualize 2D slices of a 4D label volume.

    Parameters:
        data: np.ndarray
            Shape (labels, D, H, W)
        axis: str
            'x', 'y', or 'z' — axis to slice along
        index: int or None
            Slice index. If None, use center slice.
        title: string
            Title for the plot.
        show_all_labels: bool
            If True, overlays all labels in color. If False, shows one at a time.
    """
    assert data.ndim == 4, "Data must be (labels, D, H, W)"
    labels, D, H, W = data.shape

    if axis == 'z':
        axis_idx = 1
        max_idx = D
    elif axis == 'y':
        axis_idx = 2
        max_idx = H
    elif axis == 'x':
        axis_idx = 3
        max_idx = W
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    if index is None:
        index = max_idx // 2

    fig, ax = plt.subplots(figsize=(6, 6))

    if show_all_labels:
        # Combine all labels into one color image
        combined = np.zeros((H, W, 3), dtype=np.float32)

        # Define fixed colormap for up to 5 labels
        colormaps = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
        ]

        for i in range(min(labels, 5)):
            if axis == 'z':
                slice2d = data[i, index, :, :]
            elif axis == 'y':
                slice2d = data[i, :, index, :]
            elif axis == 'x':
                slice2d = data[i, :, :, index]

            color = np.array(colormaps[i])
            combined += np.expand_dims(slice2d, -1) * color

        combined = np.clip(combined, 0, 1)
        ax.imshow(combined)
        ax.set_title(f"{title} Overlayed Labels @ {axis.upper()}={index}")
    else:
        for i in range(labels):
            if axis == 'z':
                slice2d = data[i, index, :, :]
            elif axis == 'y':
                slice2d = data[i, :, index, :]
            elif axis == 'x':
                slice2d = data[i, :, :, index]

            ax.imshow(slice2d, cmap='gray')
            ax.set_title(f"{title} Label {i}, {axis.upper()}={index}")
            plt.pause(0.5)
            ax.clear()

    ax.axis('off')
    plt.tight_layout()
    plt.show()




# CHANGE AS DESIRED
data = np.load('neurite-oasis.v1.0/OASIS_OAS1_0406_MR1/seg4_onehot.npy')

# Copy data to be deformed
cdata = np.copy(data)

transforms = monai.transforms.Compose([
    monai.transforms.Rand3DElastic(prob=1, sigma_range=(3, 5), magnitude_range=(80, 100), mode="nearest")
    # monai.transforms.RandAffine(),
    # monai.transforms.RandRotate(range_x=np.pi/8, prob=0.5),
    # monai.transforms.RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    # monai.transforms.RandGaussianNoise(prob=0.3)
])

# Deform the data
deformed_data = transforms(cdata)

# Display the original and deformed data (change index to show slices at different positions)
show_slices(data, index=140)
show_slices(deformed_data, index=140, title="Deformed Map")
