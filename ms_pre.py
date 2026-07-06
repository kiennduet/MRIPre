import os
import subprocess
import nibabel as nib
import numpy as np

# =================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (CHỈ SỬA Ở ĐÂY)
# =================================================================
T1_RAW = "/home/data/vub_ms/BIDS/sub-BRUMEG0921/ses-01/anat/sub-BRUMEG0921_ses-01_T1w.nii.gz"
FLAIR_RAW = "/home/data/vub_ms/BIDS/sub-BRUMEG0921/ses-01/anat/sub-BRUMEG0921_ses-01_FLAIR.nii.gz"
OUTPUT_DIR = "/home/dknguyen/Documents/2_Works/1_FLAIR_MS/data/vub_ms/lession_filling/sub-BRUMEG0921"
MNI_REF = "/home/dknguyen/Documents/2_Works/1_FLAIR_MS/FLightcase/MRIPre/submodules/Wood_2022/Data/MNI152_T1_1mm_brain.nii"


if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# =================================================================
# 2. BẮT ĐẦU CHẠY THỬ NGHIỆM
# =================================================================

# # --- BƯỚC 1: REORIENT (Chuẩn hóa hướng) ---
# print("\n[STEP 1] Reorienting images...")
# T1_RE = f"{OUTPUT_DIR}/t1_reorient.nii.gz"
# FL_RE = f"{OUTPUT_DIR}/flair_reorient.nii.gz"
# subprocess.run(f"fslreorient2std {T1_RAW} {T1_RE}", shell=True)
# subprocess.run(f"fslreorient2std {FLAIR_RAW} {FL_RE}", shell=True)

# # --- BƯỚC 2: COREGISTRATION (Đồng bộ FLAIR vào T1) ---
# print("[STEP 2] Registering FLAIR to T1 space...")
# FL2T1 = f"{OUTPUT_DIR}/flair_to_t1.nii.gz"
# MAT_COREIG = f"{OUTPUT_DIR}/flair2t1.mat"
# subprocess.run(f"flirt -in {FL_RE} -ref {T1_RE} -out {FL2T1} -omat {MAT_COREIG} -dof 6 -cost mutualinfo", shell=True)

# # --- BƯỚC 3: SAMSEG (Tìm tổn thương MS) ---
# print("[STEP 3] Running SAMSEG (this will take a while)...")
# SAMSEG_DIR = f"{OUTPUT_DIR}/samseg_work"
# if not os.path.exists(SAMSEG_DIR): os.makedirs(SAMSEG_DIR)

# # Convert sang .mgz để SAMSEG chạy ổn định nhất
# subprocess.run(f"mri_convert {T1_RE} {SAMSEG_DIR}/t1.mgz", shell=True)
# subprocess.run(f"mri_convert {FL2T1} {SAMSEG_DIR}/flair.mgz", shell=True)

# # Chạy lệnh samseg từ help menu của bạn
# subprocess.run(f"samseg --i {SAMSEG_DIR}/t1.mgz --i {SAMSEG_DIR}/flair.mgz --o {SAMSEG_DIR} --lesion --threads 8 --save-posteriors", shell=True)

# --- BƯỚC 4: LESION MASK EXTRACTION ---
print("[STEP 4] Extracting Lesion Mask...")
LESION_MASK = f"{OUTPUT_DIR}/lesion_mask.nii.gz"
# Thử lấy từ lesions.mgz (xác suất), nếu không có thì lấy từ seg.mgz (label 77)
if os.path.exists(f"{SAMSEG_DIR}/lesions.mgz"):
    subprocess.run(f"mri_binarize --i {SAMSEG_DIR}/lesions.mgz --min 0.5 --o {LESION_MASK}", shell=True)
else:
    subprocess.run(f"mri_binarize --i {SAMSEG_DIR}/seg.mgz --match 77 --o {LESION_MASK}", shell=True)

# --- BƯỚC 5: LESION FILLING (Vá ảnh T1) ---
print("[STEP 5] Filling Lesions on T1...")
T1_FILLED = f"{OUTPUT_DIR}/t1_filled.nii.gz"
subprocess.run(f"fsl_fill_lesions -i {T1_RE} -l {LESION_MASK} -o {T1_FILLED}", shell=True)

# --- BƯỚC 6: ROBUST STANDARDIZATION (Quy trình cho GAN) ---
print("[STEP 6] Removing neck and conforming to 1mm...")
T1_ROBUST = f"{OUTPUT_DIR}/t1_robust.nii.gz"
T1_1MM = f"{OUTPUT_DIR}/t1_1mm_conform.nii.gz"
subprocess.run(f"robustfov -i {T1_FILLED} -r {T1_ROBUST}", shell=True)
subprocess.run(f"mri_convert {T1_ROBUST} {T1_1MM} --conform", shell=True)

# --- BƯỚC 7: SKULL STRIPPING ---
print("[STEP 7] Skull stripping...")
MASK_BRAIN = f"{OUTPUT_DIR}/brain_mask.nii.gz"
T1_STRIPPED = f"{OUTPUT_DIR}/t1_stripped.nii.gz"
subprocess.run(f"mri_watershed {T1_1MM} {MASK_BRAIN}", shell=True)
subprocess.run(f"mri_mask {T1_1MM} {MASK_BRAIN} {T1_STRIPPED}", shell=True)

# --- BƯỚC 8: ALIGNMENT TO MNI (DOF 12) ---
print("[STEP 8] Aligning to MNI152 (12 DOF)...")
T1_AFFINE = f"{OUTPUT_DIR}/t1_mni_affine.nii.gz"
subprocess.run(f"flirt -in {T1_STRIPPED} -ref {MNI_REF} -out {T1_AFFINE} -dof 12 -searchrx -180 180 -searchry -180 180 -searchrz -180 180", shell=True)

# --- BƯỚC 9: CROP (160x192x224) ---
print("[STEP 9] Final Cropping...")
FINAL_FILE = f"{OUTPUT_DIR}/sub-test_final_GAN.nii.gz"
cog = subprocess.check_output(f"fslstats {T1_AFFINE} -c", shell=True).decode().split()
mx, my, mz = float(cog[0]), float(cog[1]), float(cog[2])
cx, cy, cz = int(max(0, min(256-160, mx-80))), int(max(0, min(256-192, my-96))), int(max(0, min(256-224, mz-112)))
subprocess.run(f"fslroi {T1_AFFINE} {FINAL_FILE} {cx} 160 {cy} 192 {cz} 224", shell=True)

# --- BƯỚC 10: INTENSITY NORMALIZATION (0-1) ---
print("[STEP 10] Normalizing Intensity 0-1...")
img = nib.load(FINAL_FILE)
data = img.get_fdata()
data = np.maximum(data, 0)
p99 = np.percentile(data, 99)
if p99 > 0: data = np.clip(data / p99, 0, 1)
nib.save(nib.Nifti1Image(data.astype(np.float32), img.affine, img.header), FINAL_FILE)

print(f"\n✅ THỬ NGHIỆM HOÀN TẤT! File kết quả: {FINAL_FILE}")