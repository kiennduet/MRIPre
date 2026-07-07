import subprocess

# --- CẤU HÌNH ---
raw_mgz   = "/home/dknguyen/Documents/2_Works/1_FLAIR_MS/data/dataTransfer/sub-p127547_ses-20190306/mode01_bias_corrected.mgz"
seg_mgz   = "/home/dknguyen/Documents/2_Works/1_FLAIR_MS/data/dataTransfer/sub-p127547_ses-20190306/seg.mgz"
label_id  = 99
output_png = "overlay_label99.png"

def run(cmd):
    print(f">> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def check_output(cmd):
    return subprocess.check_output(cmd, shell=True).decode().strip()

# Bước 1: Convert .mgz → .nii.gz (FreeSurfer mri_convert)
run(f"mri_convert {raw_mgz} raw.nii.gz")
run(f"mri_convert {seg_mgz} seg.nii.gz")

# Bước 2: Trích label 99 → binary mask
run(f"fslmaths seg.nii.gz -thr {label_id} -uthr {label_id} -bin mask_label{label_id}.nii.gz")

# Bước 3: Lấy thống kê ảnh nền
raw_min = check_output("fslstats raw.nii.gz -P 2")
raw_max = check_output("fslstats raw.nii.gz -P 98")

# Bước 4: Overlay — label 99 tô màu đỏ-vàng, label khác trong suốt
run(f"overlay 1 0 raw.nii.gz {raw_min} {raw_max} mask_label{label_id}.nii.gz 0.5 1 rendered_overlay.nii.gz")

# Bước 5: Xuất PNG lightbox
run(f"slicer rendered_overlay.nii.gz -S 2 1500 {output_png}")
print(f"Saved: {output_png}")
