import os
import subprocess


# =================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =================================================================
T1_PREPROC = "/home/data/vub_ms/BIDS/derivatives/Wood_2022_skull-stripped/sub-BRUMEG0921/ses-01/anat/sub-BRUMEG0921_ses-01_T1w.nii.gz"    # File T1 đã xử lý
FLAIR_PREPROC = "/home/data/vub_ms/BIDS/derivatives/Wood_2022_skull-stripped/sub-BRUMEG0921/ses-01/anat/sub-BRUMEG0921_ses-01_FLAIR.nii.gz" # File FLAIR đã align vào T1
OUTPUT_DIR = "/home/dknguyen/Documents/2_Works/1_FLAIR_MS/data/vub_ms/lession_filling/sub-BRUMEG0921_preco"


if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def run_cmd(cmd):
    print(f"--> Chạy: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


# # --- BƯỚC 2: COREGISTRATION (Đồng bộ FLAIR vào T1) ---
print("[STEP 0] Registering FLAIR to T1 space...")
FL2T1 = f"{OUTPUT_DIR}/flair_to_t1.nii.gz"
MAT_COREIG = f"{OUTPUT_DIR}/flair2t1.mat"
subprocess.run(f"flirt -in {FLAIR_PREPROC} -ref {T1_PREPROC} -out {FL2T1} -omat {MAT_COREIG} -dof 6 -cost mutualinfo", shell=True)

# # =================================================================
# # BƯỚC 1: SAMSEG - PHÂN ĐOẠN TỔN THƯƠNG
# # =================================================================
# print("\n[BƯỚC 1] Đang chạy SAMSEG (Tìm tổn thương MS)...")
SAMSEG_WORK = os.path.join(OUTPUT_DIR, "samseg_tmp")
if not os.path.exists(SAMSEG_WORK): os.makedirs(SAMSEG_WORK)

# Chạy SAMSEG trực tiếp trên dữ liệu preprocessed (đã 1mm nên sẽ nhanh hơn)
# --lesion: Bật chế độ tìm tổn thương
# --save-posteriors: Xuất file xác suất lesions.mgz
run_cmd(f"samseg --i {T1_PREPROC} --i {FL2T1} --o {SAMSEG_WORK} --lesion --threads 12 --save-posteriors")

# =================================================================
# BƯỚC 2: TẠO MẶT NẠ NHỊ PHÂN (LESION MASK)
# =================================================================
print("\n[BƯỚC 2] Đang tạo Lesion Mask...")
LESION_MASK = os.path.join(OUTPUT_DIR, "lesion_mask.nii.gz")
run_cmd(f"mri_binarize --i {SAMSEG_WORK}/seg.mgz --match 99 --o {LESION_MASK}")

# Kiểm tra xem mask có trống không
vol = subprocess.check_output(f"fslstats {LESION_MASK} -V", shell=True).decode().split()[0]
if float(vol) == 0:
    print("⚠️ CẢNH BÁO: Không tìm thấy tổn thương.")

# =================================================================
# BƯỚC 3: LESION FILLING - VÁ ẢNH T1
# =================================================================
print("\n[BƯỚC 3] Đang thực hiện Lesion Filling trên ảnh T1...")
T1_FILLED = os.path.join(OUTPUT_DIR, "t1_preproc_filled.nii.gz")

# fsl_fill_lesions dùng thuật toán nội suy để lấp đầy vùng mask bằng chất trắng bình thường
run_cmd(f"lesion_filling -i {T1_PREPROC} -l {LESION_MASK} -o {T1_FILLED}")

print(f"\n✅ HOÀN THÀNH!")
print(f"Mặt nạ tổn thương: {LESION_MASK}")
print(f"Ảnh T1 đã vá: {T1_FILLED}")
