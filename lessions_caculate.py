#!/usr/bin/env python3
import subprocess
from pathlib import Path

T1_RAW    = "/home/data/vub_ms/BIDS/sub-BRUMEG0921/ses-01/anat/sub-BRUMEG0921_ses-01_T1w.nii.gz"
FLAIR_RAW = "/home/data/vub_ms/BIDS/sub-BRUMEG0921/ses-01/anat/sub-BRUMEG0921_ses-01_FLAIR.nii.gz"
WORK      = Path("/home/dknguyen/Documents/2_Works/1_FLAIR_MS/data/vub_ms/lession_volume/")
THREADS   = 14

WORK.mkdir(parents=True, exist_ok=True)
(WORK / "samseg").mkdir(exist_ok=True)

def run(cmd): subprocess.run(cmd, shell=True, check=True)

# STEP 0 – Resample cả hai về 1mm isotropic (giữ FOV, chỉ đổi voxel size)
t1_1mm    = WORK / "T1_1mm.nii.gz"
flair_1mm = WORK / "FLAIR_1mm.nii.gz"
run(f"mri_convert {T1_RAW}    {t1_1mm}    --conform")  # FreeSurfer: 1mm iso + 256³
run(f"mri_convert {FLAIR_RAW} {flair_1mm} --conform")

# STEP 1 – Coreg FLAIR_1mm → T1_1mm (rigid, mutual info)
fl2t1 = WORK / "flair2t1.nii.gz"
run(f"flirt -in {flair_1mm} -ref {t1_1mm} -out {fl2t1} -omat {WORK}/flair2t1.mat -dof 6 -cost mutualinfo")

# STEP 2 – SAMSEG (cả hai giờ cùng space + cùng resolution)
run(f"samseg --i {t1_1mm} --i {fl2t1} --o {WORK}/samseg --lesion --threads {THREADS} --save-posteriors")

# STEP 3 – Lesion mask
mask = WORK / "lesion_mask.nii.gz"
run(f"mri_binarize --i {WORK}/samseg/seg.mgz --match 99 --o {mask}")

# STEP 4 – Volume (tính trên 1mm³ voxel → mm³ = số voxel)
n, vol = subprocess.check_output(f"fslstats {mask} -V", shell=True).decode().split()
print(f"✓ Lesion volume: {float(vol):.2f} mm³  ({float(vol)/1000:.4f} mL)  –  {n} voxels")