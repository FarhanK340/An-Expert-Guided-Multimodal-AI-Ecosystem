import nibabel as nib
import numpy as np
import sys
from pathlib import Path

# Path from logs: h:/FYP/synapsedownloads/Brats2024/BratsGLI/training_data1_v2/BraTS-GLI-02193-100/BraTS-GLI-02193-100-seg.nii.gz
# Finding a file...
base_dir = Path("h:/FYP/synapsedownloads/Brats2024/BratsGLI/training_data1_v2")
if not base_dir.exists():
    print(f"Dir not found: {base_dir}")
    sys.exit(1)

# Find first seg file
seg_files = list(base_dir.rglob("*-seg.nii.gz"))
if not seg_files:
    print("No seg files found.")
    sys.exit(1)

target_file = seg_files[0]
print(f"Inspecting file: {target_file}")

img = nib.load(str(target_file))
data = img.get_fdata()
unique = np.unique(data)
print(f"Unique values: {unique}")
