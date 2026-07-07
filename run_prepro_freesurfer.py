import os
import subprocess
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np


def run_cmd(cmd, step=""):
    """Thực thi lệnh shell, raise lỗi ngay nếu thất bại (thay vì âm thầm bỏ qua)"""
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"[{step}] thất bại: {res.stderr.strip()[:300]}")
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

    return 100 * float(np.mean(data >= 1.0))  # % voxel bị clip ở đỉnh (QC)


def process_file(input_p, output_p, mni_ref, res, dof, bias_correct=True):
    """Quy trình xử lý lõi nâng cao: Reorient -> RobustFOV -> Conform -> (Bias correct) -> Skull-strip -> Align -> Crop"""
    out_dir = os.path.dirname(output_p)
    os.makedirs(out_dir, exist_ok=True)

    # Tạo tên file tạm trong thư mục output
    tmp_base = output_p.replace(".nii.gz", "_TMP")

    # Danh sách các file trung gian
    reoriented = f"{tmp_base}_reorient.nii.gz"
    robust_roi = f"{tmp_base}_robust.nii.gz"
    std_1mm    = f"{tmp_base}_1.nii.gz"
    biascorr   = f"{tmp_base}_n3.nii.gz"
    mask       = f"{tmp_base}_m.nii.gz"
    stripped   = f"{tmp_base}_s.nii.gz"
    affine     = f"{tmp_base}_a.nii.gz"
    resampled  = f"{tmp_base}_r.nii.gz"
    crop       = f"{tmp_base}_crop.nii.gz"

    try:
        # BƯỚC 0: Chuẩn hóa hướng và loại bỏ cổ thừa (Sửa lỗi lệch cổ)
        # fslreorient2std: Đưa ảnh về hướng chuẩn RAS/FSL
        run_cmd(f"fslreorient2std {input_p} {reoriented}", "reorient2std")

        # robustfov: Tự động nhận diện vùng não và cắt bỏ phần cổ/vai thừa
        # Đây là lệnh quan trọng nhất để FLIRT không bị nhầm cổ là não
        run_cmd(f"robustfov -i {reoriented} -r {robust_roi}", "robustfov")

        # BƯỚC 1: Conform 1mm (Dùng file đã cắt cổ làm đầu vào)
        run_cmd(f"mri_convert {robust_roi} {std_1mm} --conform", "conform")

        # BƯỚC 1.5: Bias field correction (N3) - tùy chọn, giúp watershed/flirt chính xác hơn
        skull_input = std_1mm
        if bias_correct:
            run_cmd(f"mri_nu_correct.mni --i {std_1mm} --o {biascorr} --n 3", "bias_correct")
            skull_input = biascorr

        # BƯỚC 2: Skull-strip (Tách sọ)
        run_cmd(f"mri_watershed {skull_input} {mask}", "watershed")
        run_cmd(f"mri_mask {skull_input} {mask} {stripped}", "mri_mask")

        # QC: the tich nao sau strip (mm3) - phat hien watershed fail/cat qua tay
        brain_vol_mm3 = float(run_cmd(f"fslstats {stripped} -V", "qc_volume").split()[1])

        # BƯỚC 3: Alignment (Đưa về không gian MNI)
        # Thêm các tham số search để tìm kiếm kỹ hơn nếu não bị nghiêng
        run_cmd(f"flirt -in {stripped} -ref {mni_ref} -out {affine} -dof {dof} "
                f"-searchrx -180 180 -searchry -180 180 -searchrz -180 180", "flirt")

        # BƯỚC 4: Resample (Nếu chọn 2mm)
        working_file = affine
        grid_lim = 256
        if res == 2:
            run_cmd(f"mri_convert {affine} {resampled} --voxsize 2 2 2", "resample_2mm")
            working_file = resampled
            grid_lim = 128

        # BƯỚC 5: Crop (Cắt theo chuẩn StyleGAN3D)
        dx, dy, dz = (160, 192, 224) if res == 1 else (80, 96, 112)
        bbox = run_cmd(f"fslstats {working_file} -w", "fslstats_bbox").split()
        mx, my, mz = int(bbox[0])+(int(bbox[1])/2), int(bbox[2])+(int(bbox[3])/2), int(bbox[4])+(int(bbox[5])/2)

        # Cảnh báo nếu bbox thực tế của não lớn hơn kích thước crop cố định -> có thể bị cắt mất mô
        xsize, ysize, zsize = int(bbox[1]), int(bbox[3]), int(bbox[5])
        if xsize > dx or ysize > dy or zsize > dz:
            print(f"    [!] CẢNH BÁO: bbox não ({xsize},{ysize},{zsize}) vượt kích thước crop "
                  f"({dx},{dy},{dz}) - có thể bị cắt mất mô não")

        cx = int(max(0, min(grid_lim - dx, mx - dx/2)))
        cy = int(max(0, min(grid_lim - dy, my - dy/2)))
        cz = int(max(0, min(grid_lim - dz, mz - dz/2)))

        run_cmd(f"fslroi {working_file} {crop} {cx} {dx} {cy} {dy} {cz} {dz}", "fslroi")

        # BƯỚC 6: CHUẨN HÓA CƯỜNG ĐỘ
        pct_clip1 = normalize_intensity_01(crop, output_p) # Ghi đè lên file output cuối cùng

        if not os.path.exists(output_p):
            return None
        return {"volume_mm3": brain_vol_mm3, "pct_clip1": pct_clip1}

    except Exception as e:
        print(f"    [x] LỖI xử lý {input_p}: {e}")
        return None

    finally:
        # Dọn dẹp TẤT CẢ file tạm
        temp_files = [reoriented, robust_roi, std_1mm, biascorr, mask, stripped, affine, resampled, crop]
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

def main():
    parser = argparse.ArgumentParser(description="BIDS Preprocessing for 3D-StyleGAN (External Output)")
    
    # Input & Output
    parser.add_argument("--bids_dir", required=True, help="Thư mục gốc BIDS (Read-only)")
    parser.add_argument("--out_dir", required=True, help="Thư mục lưu kết quả (Bạn có quyền ghi)")
    
    # Params
    parser.add_argument("--res", type=int, choices=[1, 2], default=1, help="Độ phân giải (1mm hoặc 2mm)")
    parser.add_argument("--dof", type=int, choices=[6, 12], default=6, help="DOF cho Alignment (12 hoặc 6)")
    parser.add_argument("--skip", action="store_true", help="Bỏ qua nếu file đã tồn tại")
    parser.add_argument("--bias_correct", action="store_true", help="Bật bias field correction (N3) trước khi skull-strip")

    # Software paths
    parser.add_argument("--fs_home", default="/home/dknguyen/software/freesurfer")
    parser.add_argument("--fsl_dir", default="/home/dknguyen/fsl")
    parser.add_argument("--license", default="/home/dknguyen/software/freesurfer/.license")

    args = parser.parse_args()

    # Setup
    mni_ref = "submodules/Wood_2022/Data/MNI152_T1_1mm_brain.nii"
    bids_root = Path(args.bids_dir).resolve()
    deriv_root = Path(args.out_dir).resolve()
    
    # Tìm file T1w
    t1w_list = list(bids_root.glob("sub-*/anat/*_T1w.nii.gz")) + \
               list(bids_root.glob("sub-*/ses-*/anat/*_T1w.nii.gz"))

    print(f"--- BIDS External Output Pipeline ---")
    print(f"Input: {bids_root}")
    print(f"Output: {deriv_root}")
    print(f"Found {len(t1w_list)} files.")

    qc_rows = []  # (subject, volume_mm3, pct_clip1)
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
        stats = process_file(str(t1w_path), str(final_out_path), mni_ref, args.res, args.dof, args.bias_correct)
        if stats:
            # print(f"    Saved to -> {final_out_path}")
            print(f"    [QC] volume_mm3={stats['volume_mm3']:.1f}  pct_clip1={stats['pct_clip1']:.2f}%")

            # File QC rieng cho tung subject
            subj_qc_path = Path(str(final_out_path).replace(".nii.gz", "_qc.txt"))
            subj_qc_path.write_text(
                f"subject: {t1w_path.name}\n"
                f"volume_mm3: {stats['volume_mm3']:.1f}\n"
                f"pct_clip1: {stats['pct_clip1']:.2f}\n"
            )

            qc_rows.append((t1w_path.name, stats["volume_mm3"], stats["pct_clip1"]))
        else:
            qc_rows.append((t1w_path.name, float("nan"), float("nan")))

    # File tổng toàn bộ dataset: bảng thống kê QC ra .txt + in ra bash
    qc_path = deriv_root / "qc_stats.txt"
    header = f"{'subject':<40}{'volume_mm3':>15}{'pct_clip1':>12}"
    lines = [header] + [f"{s:<40}{v:>15.1f}{p:>12.2f}" for s, v, p in qc_rows]
    print("\n--- QC STATS (toan bo dataset) ---")
    print("\n".join(lines))
    qc_path.write_text("\n".join(lines) + "\n")
    print(f"\nQC stats saved to -> {qc_path}")

if __name__ == "__main__":
    main()
    
    # python3 run_prepro_freesurfer.py --bids_dir /home/data/vub_ms/BIDS --out_dir /home/dknguyen/Documents/2_Works/1_FLAIR_MS/data/vub_ms/derivatives/dof6 --res 1 --dof 6

# export FREESURFER_HOME=$HOME/software/freesurfer
# source $FREESURFER_HOME/SetUpFreeSurfer.sh
# export FS_LICENSE=/home/dknguyen/software/freesurfer/.license
