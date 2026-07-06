
MNI_TEMPLATE = "/home/dknguyen/Documents/2_Works/1_FLAIR_MS/FLightcase/MRIPre/submodules/Wood_2022/Data/MNI152_T1_1mm_brain.nii"

import cmd
import os
import subprocess
import shutil

from nbconvert import export


def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout.strip()

def process_mri(input_raw, output_dir):

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


def process_mri_ms_lesion_filled(t1_raw, flair_raw, output_dir):
    """
    Quy trình tiền xử lý MRI cho bệnh MS:
    Coregistration -> Lesion Segmentation -> Lesion Filling -> Standard Pipeline
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    base = os.path.basename(t1_raw).split('.')[0]
    
    # 1. Định nghĩa file trung gian
    t1_reorient = os.path.join(output_dir, f"{base}_t1_re.nii.gz")
    flair_reorient = os.path.join(output_dir, f"{base}_fl_re.nii.gz")
    flair_to_t1 = os.path.join(output_dir, f"{base}_fl_reg_t1.nii.gz")
    samseg_dir = os.path.join(output_dir, "samseg_out")
    lesion_mask = os.path.join(output_dir, f"{base}_lesion_mask.nii.gz")
    t1_filled = os.path.join(output_dir, f"{base}_t1_filled.nii.gz")
    final = os.path.join(output_dir, f"{base}_final.nii.gz")

    print(f"--- 🚀 Bắt đầu quy trình MS chuẩn (SAMSEG) cho: {base} ---")

    # BƯỚC 1: Reorient
    run_cmd(f"fslreorient2std {t1_raw} {t1_reorient}")
    run_cmd(f"fslreorient2std {flair_raw} {flair_reorient}")

    # BƯỚC 2: Coregister FLAIR to T1 (Rigid - 6 DOF)
    # Rất quan trọng để samseg nhận diện đúng tổn thương trên cả 2 ảnh
    run_cmd(f"flirt -in {flair_reorient} -ref {t1_reorient} -out {flair_to_t1} -dof 6 -cost mutualinfo")

# BƯỚC 3: Lesion Segmentation bằng SAMSEG
    print("-> Đang chạy SAMSEG để tìm tổn thương MS (T1 + FLAIR)...")
    # Chạy samseg (đảm bảo thư mục samseg_dir đã được tạo hoặc sạch sẽ)
    run_cmd(f"samseg --i {t1_reorient} --i {flair_to_t1} --o {samseg_dir} --ms")
    
    # Định nghĩa các file đầu ra tiềm năng của SAMSEG
    samseg_lesion_prob = os.path.join(samseg_dir, "lesions.mgz") # File xác suất tổn thương
    samseg_seg_file = os.path.join(samseg_dir, "seg.mgz")         # File phân đoạn tổng quát
    
    if os.path.exists(samseg_lesion_prob):
        # ƯU TIÊN: Dùng file lesions.mgz vì nó chứa thông tin chi tiết hơn
        # Ngưỡng --min 0.5 thường là chuẩn để tạo mặt nạ nhị phân từ xác suất
        print("-> Tìm thấy lesions.mgz, đang tạo mặt nạ tổn thương (threshold=0.5)...")
        run_cmd(f"mri_binarize --i {samseg_lesion_prob} --min 0.5 --o {lesion_mask}")
        
    elif os.path.exists(samseg_seg_file):
        # DỰ PHÒNG: Nếu không có lesions.mgz, trích xuất nhãn 77 từ seg.mgz
        print("-> Không tìm thấy lesions.mgz, đang trích xuất Label 77 từ seg.mgz...")
        run_cmd(f"mri_binarize --i {samseg_seg_file} --match 77 --o {lesion_mask}")
        
    else:
        print(f"❌ Lỗi: SAMSEG không tạo ra kết quả tại {samseg_dir}")
        return False # Hoặc raise Exception tùy vào cấu trúc code của bạn


    # BƯỚC 4: Lesion Filling (Dùng mặt nạ tổn thương để vá ảnh T1)
    print("-> Đang thực hiện Lesion Filling...")
    run_cmd(f"fsl_fill_lesions -i {t1_reorient} -l {lesion_mask} -o {t1_filled}")

    # BƯỚC 5: Quy trình tiền xử lý GAN tiêu chuẩn (RobustFOV -> Conform -> Align -> Crop)
    # (Đoạn này giữ nguyên logic mạnh mẽ nhất chúng ta đã xây dựng)
    roi_file = os.path.join(output_dir, "temp_roi.nii.gz")
    std_1mm = os.path.join(output_dir, "temp_1mm.nii.gz")
    stripped = os.path.join(output_dir, "temp_stripped.nii.gz")
    mask = os.path.join(output_dir, "temp_mask.nii.gz")
    affine = os.path.join(output_dir, "temp_affine.nii.gz")
    mni_template = os.path.expandvars("$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz")

    run_cmd(f"robustfov -i {t1_filled} -r {roi_file}")
    run_cmd(f"mri_convert {roi_file} {std_1mm} --conform")
    run_cmd(f"mri_watershed {std_1mm} {mask}")
    run_cmd(f"mri_mask {std_1mm} {mask} {stripped}")
    
    # Align DOF 12 (nhờ lấp tổn thương nên sẽ không bị méo)
    run_cmd(f"flirt -in {stripped} -ref {mni_template} -out {affine} -dof 6 -searchrx -180 180 -searchry -180 180 -searchrz -180 180")

    # Crop an toàn 160x192x224 dựa trên Center of Gravity
    cog = run_cmd(f"fslstats {affine} -c").split()
    mx, my, mz = float(cog[0]), float(cog[1]), float(cog[2])
    cx, cy, cz = int(max(0, min(256-160, mx-80))), int(max(0, min(256-192, my-96))), int(max(0, min(256-224, mz-112)))
    
    run_cmd(f"fslroi {affine} {final} {cx} 160 {cy} 192 {cz} 224")

    print(f"✅ HOÀN THÀNH: {final}")

# --- CHẠY THỬ ---
if __name__ == "__main__":

    freesurfer = """
    export FREESURFER_HOME=$HOME/software/freesurfer
    source $FREESURFER_HOME/SetUpFreeSurfer.sh
    export FS_LICENSE=/home/dknguyen/software/freesurfer/.license
    """

    # file_raw = "/home/data/vub_ms/BIDS/sub-BRUMEG0947/ses-01/anat/sub-BRUMEG0947_ses-01_T1w.nii.gz" # Đường dẫn file của bạn
    # thu_muc_ra = "./output_160×192×224_dof12"
    # print("Starting...")
    # process_mri(file_raw, thu_muc_ra)

    t1_raw = "/home/data/vub_ms/BIDS/sub-BRUMEG0921/ses-01/anat/sub-BRUMEG0921_ses-01_T1w.nii.gz"
    flair_raw = "/home/data/vub_ms/BIDS/sub-BRUMEG0921/ses-01/anat/sub-BRUMEG0921_ses-01_FLAIR.nii.gz"
    output_dir = "/home/dknguyen/Documents/2_Works/1_FLAIR_MS/data/vub_ms/lession_filling/sub-BRUMEG0921"
    process_mri_ms_lesion_filled(t1_raw, flair_raw, output_dir);
