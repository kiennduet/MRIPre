
MNI_TEMPLATE = "/home/dknguyen/Documents/2_Works/1_FLAIR_MS/FLightcase/MRIPre/submodules/Wood_2022/Data/MNI152_T1_1mm_brain.nii"

import cmd
import os
import subprocess
import shutil

fastsufer = """
export FREESURFER_HOME=$HOME/software/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export FS_LICENSE=/home/dknguyen/software/freesurfer/.license
recon-all -version
"""


def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout.strip()

def process_mri_stylegan3d(input_raw, output_dir):

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    base = os.path.basename(input_raw).split('.')[0]
    
    std_1mm = os.path.join(output_dir, f"{base}_1mm.nii.gz")
    stripped = os.path.join(output_dir, f"{base}_stripped.nii.gz")
    mask = os.path.join(output_dir, f"{base}_mask.nii.gz")
    affine = os.path.join(output_dir, f"{base}_affine.nii.gz")
    final = os.path.join(output_dir, f"{base}_final.nii.gz")
    mni_template = os.path.expandvars("$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz")

    print(f"--- Đang xử lý an toàn cho: {base} ---")

    reoriented = os.path.join(output_dir, "temp_reorient.nii.gz")
    roi_file = os.path.join(output_dir, "temp_robustfov.nii.gz")

    print(f"--- Đang xử lý Robust cho: {base} ---")

    # BƯỚC 0a: Đưa về hướng chuẩn (Tránh lệch trục)
    run_cmd(f"fslreorient2std {input_raw} {reoriented}")

    # BƯỚC 0b: Tự động cắt bớt cổ (Quan trọng nhất để sửa lỗi của bạn)
    # Lệnh này sẽ bỏ phần cổ thừa, giúp FLIRT tập trung vào não
    run_cmd(f"robustfov -i {reoriented} -r {roi_file}")

    # 1. Conform 1mm
    subprocess.run(f"mri_convert {roi_file} {std_1mm} --conform", shell=True)

    # 2. Skull-strip
    subprocess.run(f"mri_watershed {std_1mm} {mask}", shell=True)
    subprocess.run(f"mri_mask {std_1mm} {mask} {stripped}", shell=True)

    # 3. Affine Alignment (Thêm tìm kiếm rộng để tránh lệch)
    subprocess.run(f"flirt -in {stripped} -ref {mni_template} -out {affine} -dof 12 -searchrx -180 180 -searchry -180 180 -searchrz -180 180", shell=True)

    # 4. TÌM TÂM BẰNG BOUNDING BOX (Chính xác hơn COG)
    # Lệnh này trả về: x_min x_size y_min y_size z_min z_size
    bbox = run_cmd(f"fslstats {affine} -w")
    items = bbox.split()
    xmin_b, xsz_b = int(items[0]), int(items[1])
    ymin_b, ysz_b = int(items[2]), int(items[3])
    zmin_b, zsz_b = int(items[4]), int(items[5])

    # Tính tâm hình học của bộ não
    mid_x = xmin_b + (xsz_b / 2)
    mid_y = ymin_b + (ysz_b / 2)
    mid_z = zmin_b + (zsz_b / 2)

    # KÍCH THƯỚC 
    dx, dy, dz = 160,192,224
    
    # Tính tọa độ bắt đầu cắt
    crop_x = int(max(0, mid_x - dx/2))
    crop_y = int(max(0, mid_y - dy/2))
    crop_z = int(max(0, mid_z - dz/2))

    # Sửa lỗi nếu hộp vượt quá 256
    if crop_x + dx > 256: crop_x = 256 - dx
    if crop_y + dy > 256: crop_y = 256 - dy
    if crop_z + dz > 256: crop_z = 256 - dz

    print(f"Vùng não thực tế tìm thấy: X:{xsz_b} Y:{ysz_b} Z:{zsz_b}")
    print(f"Quyết định cắt tại: x={crop_x}, y={crop_y}, z={crop_z} với kích thước {dx}x{dy}x{dz}")

    # 5. Cắt
    subprocess.run(f"fslroi {affine} {final} {crop_x} {dx} {crop_y} {dy} {crop_z} {dz}", shell=True)

    print(f"✅ HOÀN THÀNH: {final}")

# --- CHẠY THỬ ---
if __name__ == "__main__":
    run_cmd(fastsufer)
    file_raw = "/home/data/vub_ms/BIDS/sub-BRUMEG0947/ses-01/anat/sub-BRUMEG0947_ses-01_T1w.nii.gz" # Đường dẫn file của bạn
    thu_muc_ra = "./output_160×192×224_dof12"
    print("Starting...")
    process_mri_stylegan3d(file_raw, thu_muc_ra)

