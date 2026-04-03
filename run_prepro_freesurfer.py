import os
import subprocess
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np

def setup_environment(fs_home, fsl_dir, fs_license):
    """Thiết lập biến môi trường cho FreeSurfer và FSL"""
    os.environ["FREESURFER_HOME"] = fs_home
    os.environ["FSLDIR"] = fsl_dir
    os.environ["FS_LICENSE"] = fs_license
    os.environ["PATH"] += f":{fs_home}/bin:{fsl_dir}/bin"
    
    mni_path = os.path.join(fsl_dir, "data/standard/MNI152_T1_1mm_brain.nii.gz")
    return mni_path

def run_cmd(cmd):
    """Thực thi lệnh shell"""
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        return f"ERROR: {res.stderr.strip()}"
    return res.stdout.strip()

def normalize_intensity_01(input_p, output_p):
    """Chuẩn hóa cường độ ảnh về dải 0-1 sử dụng 99th percentile để tránh outliers"""
    img = nib.load(input_p)
    data = img.get_fdata()

    # Loại bỏ giá trị âm (nếu có do nội suy)
    data = np.maximum(data, 0)

    # Sử dụng phân vị thứ 99 để xác định giá trị tối đa (tránh các điểm nhiễu cực sáng)
    p99 = np.percentile(data, 99)
    p0 = np.min(data)

    if p99 - p0 > 0:
        data = (data - p0) / (p99 - p0)
    
    # Cắt các giá trị vượt quá 1 (do ban đầu ta dùng percentile 99)
    data = np.clip(data, 0, 1)

    # Lưu lại file với kiểu dữ liệu float32 để nhẹ và chuẩn cho Deep Learning
    new_img = nib.Nifti1Image(data.astype(np.float32), img.affine, img.header)
    nib.save(new_img, output_p)

def process_file(input_p, output_p, mni_ref, res, dof):
    """Quy trình xử lý lõi nâng cao: Reorient -> RobustFOV -> Conform -> Skull-strip -> Align -> Crop"""
    out_dir = os.path.dirname(output_p)
    os.makedirs(out_dir, exist_ok=True)
    
    # Tạo tên file tạm trong thư mục output
    tmp_base = output_p.replace(".nii.gz", "_TMP")
    
    # Danh sách các file trung gian
    reoriented = f"{tmp_base}_reorient.nii.gz"
    robust_roi = f"{tmp_base}_robust.nii.gz"
    std_1mm    = f"{tmp_base}_1.nii.gz"
    mask       = f"{tmp_base}_m.nii.gz"
    stripped   = f"{tmp_base}_s.nii.gz"
    affine     = f"{tmp_base}_a.nii.gz"
    resampled  = f"{tmp_base}_r.nii.gz"

    try:
        # BƯỚC 0: Chuẩn hóa hướng và loại bỏ cổ thừa (Sửa lỗi lệch cổ)
        # fslreorient2std: Đưa ảnh về hướng chuẩn RAS/FSL
        run_cmd(f"fslreorient2std {input_p} {reoriented}")
        
        # robustfov: Tự động nhận diện vùng não và cắt bỏ phần cổ/vai thừa
        # Đây là lệnh quan trọng nhất để FLIRT không bị nhầm cổ là não
        run_cmd(f"robustfov -i {reoriented} -r {robust_roi}")
        
        # BƯỚC 1: Conform 1mm (Dùng file đã cắt cổ làm đầu vào)
        run_cmd(f"mri_convert {robust_roi} {std_1mm} --conform")
        
        # BƯỚC 2: Skull-strip (Tách sọ)
        run_cmd(f"mri_watershed {std_1mm} {mask}")
        run_cmd(f"mri_mask {std_1mm} {mask} {stripped}")
        
        # BƯỚC 3: Alignment (Đưa về không gian MNI)
        # Thêm các tham số search để tìm kiếm kỹ hơn nếu não bị nghiêng
        run_cmd(f"flirt -in {stripped} -ref {mni_ref} -out {affine} -dof {dof} "
                f"-searchrx -180 180 -searchry -180 180 -searchrz -180 180")

        # BƯỚC 4: Resample (Nếu chọn 2mm)
        working_file = affine
        grid_lim = 256
        if res == 2:
            run_cmd(f"mri_convert {affine} {resampled} --voxsize 2 2 2")
            working_file = resampled
            grid_lim = 128

        # BƯỚC 5: Crop (Cắt theo chuẩn StyleGAN3D)
        dx, dy, dz = (160, 192, 224) if res == 1 else (80, 96, 112)
        bbox = run_cmd(f"fslstats {working_file} -w")
        if "ERROR" in bbox or not bbox: return False
            
        it = bbox.split()
        mx, my, mz = int(it[0])+(int(it[1])/2), int(it[2])+(int(it[3])/2), int(it[4])+(int(it[5])/2)
        
        cx = int(max(0, min(grid_lim - dx, mx - dx/2)))
        cy = int(max(0, min(grid_lim - dy, my - dy/2)))
        cz = int(max(0, min(grid_lim - dz, mz - dz/2)))

        run_cmd(f"fslroi {working_file} {output_p} {cx} {dx} {cy} {dy} {cz} {dz}")

        # BƯỚC 6: CHUẨN HÓA CƯỜNG ĐỘ (MỚI THÊM)
        if os.path.exists(output_p):
            print(f"    [+] Normalizing intensity to 0-1...")
            normalize_intensity_01(output_p, output_p) # Ghi đè lên file output cuối cùng
        
        return os.path.exists(output_p)        

        
    finally:
        # Dọn dẹp TẤT CẢ file tạm
        temp_files = [reoriented, robust_roi, std_1mm, mask, stripped, affine, resampled]
        for f in temp_files:
            if os.path.exists(f): 
                os.remove(f)

def main():
    parser = argparse.ArgumentParser(description="BIDS Preprocessing for 3D-StyleGAN (External Output)")
    
    # Input & Output
    parser.add_argument("--bids_dir", required=True, help="Thư mục gốc BIDS (Read-only)")
    parser.add_argument("--out_dir", required=True, help="Thư mục lưu kết quả (Bạn có quyền ghi)")
    
    # Params
    parser.add_argument("--res", type=int, choices=[1, 2], default=2, help="Độ phân giải (1mm hoặc 2mm)")
    parser.add_argument("--dof", type=int, choices=[6, 12], default=12, help="DOF cho Alignment (12 hoặc 6)")
    parser.add_argument("--skip", action="store_true", help="Bỏ qua nếu file đã tồn tại")
    
    # Software paths
    parser.add_argument("--fs_home", default="/home/dknguyen/software/freesurfer")
    parser.add_argument("--fsl_dir", default="/usr/local/fsl")
    parser.add_argument("--license", default="/home/dknguyen/software/freesurfer/.license")

    args = parser.parse_args()

    # Setup
    mni_ref = setup_environment(args.fs_home, args.fsl_dir, args.license)
    bids_root = Path(args.bids_dir).resolve()
    deriv_root = Path(args.out_dir).resolve()
    
    # Tìm file T1w
    t1w_list = list(bids_root.glob("sub-*/anat/*_T1w.nii.gz")) + \
               list(bids_root.glob("sub-*/ses-*/anat/*_T1w.nii.gz"))

    print(f"--- BIDS External Output Pipeline ---")
    print(f"Input: {bids_root}")
    print(f"Output: {deriv_root}")
    print(f"Found {len(t1w_list)} files.")

    for t1w_path in t1w_list:
        # Lấy phần đường dẫn tương đối từ bids_root (VD: sub-01/ses-01/anat/...)
        relative_p = t1w_path.relative_to(bids_root)
        
        # Tạo tên file output mới
        out_name = t1w_path.name
        
        # Đường dẫn đích = Thư mục đầu ra + cấu hình tương đối
        final_out_path = deriv_root / relative_p.parent / out_name

        if args.skip and final_out_path.exists():
            print(f"[-] Skip: {t1w_path.name}")
            continue

        print(f"[*] Processing: {t1w_path.name}")
        if process_file(str(t1w_path), str(final_out_path), mni_ref, args.res, args.dof):
            print(f"    Saved to -> {final_out_path}")

if __name__ == "__main__":
    main()
    
    # python3 run_prepro_freesurfer.py --bids_dir /home/data/vub_ms/BIDS --out_dir /home/dknguyen/Documents/2_Works/1_FLAIR_MS/data/vub_ms/derivatives --res 1 --dof 6
