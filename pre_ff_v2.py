import os
import subprocess
import pandas as pd

# Đường dẫn template và phần mềm (Cấu hình một lần ở đây)
MNI_TEMPLATE = "/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"
FSL_LICENSE = "/home/dknguyen/software/freesurfer/.license"

def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"❌ LỖI: {res.stderr}")
    return res.stdout.strip()

def process_mri_stylegan3d_prof_method(input_raw, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    base = os.path.basename(input_raw).split('.')[0]
    
    # Định nghĩa các file output
    reoriented = os.path.join(output_dir, f"{base}_reorient.nii.gz")
    roi_file = os.path.join(output_dir, f"{base}_robust.nii.gz")
    std_1mm = os.path.join(output_dir, f"{base}_1mm.nii.gz")
    mask = os.path.join(output_dir, f"{base}_mask.nii.gz")
    stripped = os.path.join(output_dir, f"{base}_stripped.nii.gz")
    
    # Các file liên quan đến phương pháp của Giáo sư
    affine_out = os.path.join(output_dir, f"{base}_affine_MNI.nii.gz")
    affine_mat = os.path.join(output_dir, f"{base}_T_native2mni.mat")
    inv_mat = os.path.join(output_dir, f"{base}_T_mni2native.mat")
    native_atlas = os.path.join(output_dir, f"{base}_atlas_in_native.nii.gz")
    
    # File kết quả cuối cho GAN
    final = os.path.join(output_dir, f"{base}_final_stylegan.nii.gz")

    print(f"--- 🚀 Bắt đầu quy trình cho: {base} ---")

    # BƯỚC 0: Tiền xử lý Robust (Reorient + RobustFOV)
    run_cmd(f"fslreorient2std {input_raw} {reoriented}")
    run_cmd(f"robustfov -i {reoriented} -r {roi_file}")

    # BƯỚC 1: Conform 1mm
    run_cmd(f"mri_convert {roi_file} {std_1mm} --conform")

    # BƯỚC 2: Skull-strip
    run_cmd(f"mri_watershed {std_1mm} {mask}")
    run_cmd(f"mri_mask {std_1mm} {mask} {stripped}")

    # BƯỚC 3: Affine Alignment (Ti lệ 12 DOF) - ĐÂY LÀ Ti
    print(f"-> Đang tính toán ma trận chuyển đổi Ti (12 DOF)...")
    run_cmd(f"flirt -in {stripped} -ref {MNI_TEMPLATE} -out {affine_out} -omat {affine_mat} -dof 12 -searchrx -180 180 -searchry -180 180 -searchrz -180 180")

    # BƯỚC 4: THỰC HIỆN PHƯƠNG PHÁP CỦA GIÁO SƯ (Inverse Atlas Mapping)
    print(f"-> Đang thực hiện phương pháp của Giáo sư (Inverse Mapping)...")
    
    # 4.1. Nghịch đảo ma trận Ti thành Ti^-1
    run_cmd(f"convert_xfm -omat {inv_mat} -inverse {affine_mat}")
    
    # 4.2. Áp dụng Ti^-1 để đưa Atlas MNI về không gian Native
    # Ở đây dùng MNI_TEMPLATE (brain mask) làm Atlas cơ bản để đo Whole Brain
    run_cmd(f"flirt -in {MNI_TEMPLATE} -ref {stripped} -applyxfm -init {inv_mat} -out {native_atlas} -interp nearestneighbour")
    
    # 4.3. Đo thể tích trên ảnh Native (mm3) bằng fslstats
    # Kết quả trả về: <số voxel> <thể tích mm3>
    vol_results = run_cmd(f"fslstats {stripped} -k {native_atlas} -V")
    native_vol_mm3 = vol_results.split()[1]
    print(f"✅ Thể tích thực đo được (Native Volume): {native_vol_mm3} mm3")

    # BƯỚC 5: Crop ảnh cho GAN (Sử dụng ảnh đã Affine ở bước 3)
    bbox = run_cmd(f"fslstats {affine_out} -w")
    items = bbox.split()
    mid_x, mid_y, mid_z = int(items[0]) + int(items[1])/2, int(items[2]) + int(items[3])/2, int(items[4]) + int(items[5])/2
    
    dx, dy, dz = 160, 192, 224
    cx, cy, cz = int(max(0, min(256-dx, mid_x-dx/2))), int(max(0, min(256-dy, mid_y-dy/2))), int(max(0, min(256-dz, mid_z-dz/2)))
    
    run_cmd(f"fslroi {affine_out} {final} {cx} {dx} {cy} {dy} {cz} {dz}")

    # LƯU KẾT QUẢ VÀO CSV
    res_df = pd.DataFrame([{
        "Subject": base,
        "Native_Volume_mm3": native_vol_mm3,
        "Matrix_Size": f"{dx}x{dy}x{dz}",
        "Space": "MNI152_Normalized"
    }])
    res_df.to_csv(os.path.join(output_dir, f"{base}_report.csv"), index=False)

    print(f"--- 🎉 HOÀN THÀNH: {base} ---")

if __name__ == "__main__":
    # Đảm bảo source môi trường trước khi chạy
    file_raw = "/home/data/vub_ms/BIDS/sub-BRUMEG0947/ses-01/anat/sub-BRUMEG0947_ses-01_T1w.nii.gz"
    thu_muc_ra = "./output_prof_method"
    process_mri_stylegan3d_prof_method(file_raw, thu_muc_ra)