import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Txt file where each line points to a subject segmentation map 
input_txt = "/local/scratch/v_karthik_mohan/template.txt"

labels = [0, 1, 2, 3, 4]
# Labels in thisc case are Background, Cortex, Subcortical GM, White Matter, CSF

def to_one_hot(segmentation, labels):
    one_hot = np.zeros((len(labels), *segmentation.shape), dtype=np.uint8)  
    for i, label in enumerate(labels):
        one_hot[i] = (segmentation == label).astype(np.uint8)
    return one_hot

with open(input_txt, 'r') as file:
    paths = file.read().splitlines()

for path in tqdm(paths, desc="Processing segmentation maps"):
    img = nib.load(path)
    seg_map = img.get_fdata().astype(np.int16) 
    one_hot_map = to_one_hot(seg_map, labels)

    # Saving in same subject directory
    output_path = path.replace("seg4.nii.gz", "seg4_onehot.npy")

    np.save(output_path, one_hot_map)

