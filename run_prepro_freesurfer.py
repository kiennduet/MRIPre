import os
import csv
import subprocess
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np

MNI_REF = "submodules/Wood_2022/Data/MNI152_T1_1mm_brain.nii"


def run_cmd(cmd, step=""):
    """Thực thi lệnh shell, raise lỗi ngay nếu thất bại (thay vì âm thầm bỏ qua)"""
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"[{step}] thất bại: {res.stderr.strip()[:300]}")
    return res.stdout.strip()


def normalize_intensity_01(input_p, output_p):
    """Chuẩn hóa cường độ về [0,1] theo percentile 99, trả về % voxel bị clip ở đỉnh (QC)"""
    img = nib.load(input_p)
    data = np.maximum(img.get_fdata(), 0)
    p0, p99 = np.min(data), np.percentile(data, 99)
    if p99 - p0 > 0:
        data = (data - p0) / (p99 - p0)
    data = np.clip(data, 0, 1)
    nib.save(nib.Nifti1Image(data.astype(np.float32), img.affine, img.header), output_p)
    return 100 * float(np.mean(data >= 1.0))


def process_file(input_p, output_p, res, dof, bias_correct=True):
    """Reorient -> RobustFOV -> Conform -> (Bias correct) -> Skull-strip -> Align -> Crop -> Normalize"""
    os.makedirs(os.path.dirname(output_p), exist_ok=True)

    tmp = output_p.replace(".nii.gz", "_TMP")
    reoriented, robust_roi, std_1mm = f"{tmp}_reorient.nii.gz", f"{tmp}_robust.nii.gz", f"{tmp}_1.nii.gz"
    biascorr, mask, stripped = f"{tmp}_n3.nii.gz", f"{tmp}_m.nii.gz", f"{tmp}_s.nii.gz"
    affine, resampled, crop = f"{tmp}_a.nii.gz", f"{tmp}_r.nii.gz", f"{tmp}_crop.nii.gz"
    nu_log = os.path.join(os.path.dirname(output_p), "mri_nu_correct.mni.log")  # log co dinh do mri_nu_correct.mni tao ra
    temp_files = [reoriented, robust_roi, std_1mm, biascorr, mask, stripped, affine, resampled, crop,
                  nu_log, f"{nu_log}.bak"]

    try:
        # Chuẩn hóa hướng + cắt cổ/vai thừa (tránh FLIRT nhầm cổ là não)
        run_cmd(f"fslreorient2std {input_p} {reoriented}", "reorient2std")
        run_cmd(f"robustfov -i {reoriented} -r {robust_roi}", "robustfov")

        # Conform 1mm isotropic
        run_cmd(f"mri_convert {robust_roi} {std_1mm} --conform", "conform")

        # Bias field correction (N3) - tùy chọn, giúp watershed/flirt chính xác hơn
        skull_input = std_1mm
        if bias_correct:
            run_cmd(f"mri_nu_correct.mni --i {std_1mm} --o {biascorr} --n 3", "bias_correct")
            skull_input = biascorr

        # Skull-strip
        run_cmd(f"mri_watershed {skull_input} {mask}", "watershed")
        run_cmd(f"mri_mask {skull_input} {mask} {stripped}", "mri_mask")
        brain_vol_mm3 = float(run_cmd(f"fslstats {stripped} -V", "qc_volume").split()[1])

        # Alignment về không gian MNI
        run_cmd(f"flirt -in {stripped} -ref {MNI_REF} -out {affine} -dof {dof} "
                f"-searchrx -180 180 -searchry -180 180 -searchrz -180 180", "flirt")

        # Resample nếu chọn 2mm
        working_file, grid_lim = affine, 256
        if res == 2:
            run_cmd(f"mri_convert {affine} {resampled} --voxsize 2 2 2", "resample_2mm")
            working_file, grid_lim = resampled, 128

        # Crop theo chuẩn StyleGAN3D, cảnh báo nếu bbox não vượt kích thước crop
        dx, dy, dz = (160, 192, 224) if res == 1 else (80, 96, 112)
        bbox = run_cmd(f"fslstats {working_file} -w", "fslstats_bbox").split()
        mx, my, mz = int(bbox[0]) + int(bbox[1]) / 2, int(bbox[2]) + int(bbox[3]) / 2, int(bbox[4]) + int(bbox[5]) / 2
        xsize, ysize, zsize = int(bbox[1]), int(bbox[3]), int(bbox[5])
        if xsize > dx or ysize > dy or zsize > dz:
            print(f"    [!] CẢNH BÁO: bbox não ({xsize},{ysize},{zsize}) vượt kích thước crop "
                  f"({dx},{dy},{dz}) - có thể bị cắt mất mô não")

        cx = int(max(0, min(grid_lim - dx, mx - dx / 2)))
        cy = int(max(0, min(grid_lim - dy, my - dy / 2)))
        cz = int(max(0, min(grid_lim - dz, mz - dz / 2)))
        run_cmd(f"fslroi {working_file} {crop} {cx} {dx} {cy} {dy} {cz} {dz}", "fslroi")

        # Chuẩn hóa cường độ về [0,1]
        pct_clip1 = normalize_intensity_01(crop, output_p)
        return {"volume_mm3": brain_vol_mm3, "pct_clip1": pct_clip1} if os.path.exists(output_p) else None

    except Exception as e:
        print(f"    [x] LỖI xử lý {input_p}: {e}")
        return None

    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)


def save_qc_snapshot(nii_path, out_png):
    """Xuat 1 anh PNG ghep 3 lat cat giua (sagittal/coronal/axial) de kiem tra bang mat"""
    base = out_png.replace(".png", "")
    sag, cor, axi = f"{base}_sag.png", f"{base}_cor.png", f"{base}_axi.png"
    run_cmd(f"slicer {nii_path} -x 0.5 {sag} -y 0.5 {cor} -z 0.5 {axi}", "slicer")
    run_cmd(f"pngappend {sag} + {cor} + {axi} {out_png}", "pngappend")
    for f in (sag, cor, axi):
        if os.path.exists(f):
            os.remove(f)


def main():
    parser = argparse.ArgumentParser(description="BIDS Preprocessing for 3D-StyleGAN (External Output)")
    parser.add_argument("--bids_dir", required=True, help="Thư mục gốc BIDS (Read-only)")
    parser.add_argument("--out_dir", required=True, help="Thư mục lưu kết quả")
    parser.add_argument("--res", type=int, choices=[1, 2], default=1, help="Độ phân giải (1mm hoặc 2mm)")
    parser.add_argument("--dof", type=int, choices=[6, 12], default=6, help="DOF cho Alignment")
    parser.add_argument("--skip", action="store_true", help="Bỏ qua nếu file đã tồn tại")
    parser.add_argument("--bias_correct", action="store_true", help="Bật bias field correction (N3)")
    args = parser.parse_args()

    bids_root = Path(args.bids_dir).resolve()
    deriv_root = Path(args.out_dir).resolve()
    qc_snap_dir = deriv_root / "qc_snapshots"
    qc_snap_dir.mkdir(parents=True, exist_ok=True)
    t1w_list = list(bids_root.glob("sub-*/anat/*_T1w.nii.gz")) + \
               list(bids_root.glob("sub-*/ses-*/anat/*_T1w.nii.gz")) + \
               list(bids_root.glob("*_T1w.nii.gz"))

    settings = f"res={args.res}mm dof={args.dof} bias_correct={args.bias_correct} skip={args.skip}"
    print(f"--- BIDS External Output Pipeline ---\nInput: {bids_root}\nOutput: {deriv_root}\n"
          f"Found {len(t1w_list)} files.\nSettings: {settings}\n")

    qc_rows = []  # (subject, volume_mm3, pct_clip1)
    for t1w_path in t1w_list:
        final_out_path = deriv_root / t1w_path.relative_to(bids_root).parent / t1w_path.name

        if args.skip and final_out_path.exists():
            print(f"[-] Skip: {t1w_path.name}")
            continue

        print(f"[*] Processing: {t1w_path.name}")
        stats = process_file(str(t1w_path), str(final_out_path), args.res, args.dof, args.bias_correct)
        if stats:
            print(f"    [QC] volume_mm3={stats['volume_mm3']:.1f}  pct_clip1={stats['pct_clip1']:.2f}%")
            Path(str(final_out_path).replace(".nii.gz", "_qc.txt")).write_text(
                f"subject: {t1w_path.name}\n"
                f"volume_mm3: {stats['volume_mm3']:.1f}\n"
                f"pct_clip1: {stats['pct_clip1']:.2f}\n"
            )
            qc_rows.append((t1w_path.name, stats["volume_mm3"], stats["pct_clip1"]))

            try:
                snap_path = qc_snap_dir / t1w_path.name.replace(".nii.gz", ".png")
                save_qc_snapshot(str(final_out_path), str(snap_path))
            except Exception as e:
                print(f"    [!] Khong xuat duoc QC snapshot: {e}")
        else:
            qc_rows.append((t1w_path.name, float("nan"), float("nan")))

    # File tổng toàn bộ dataset (.csv): settings dùng cho cả run + bảng thống kê QC
    print("\n--- QC STATS (toàn bộ dataset) ---")
    print(f"# settings: {settings}")
    print(f"{'subject':<40}{'volume_mm3':>15}{'pct_clip1':>12}")
    for s, v, p in qc_rows:
        print(f"{s:<40}{v:>15.1f}{p:>12.2f}")

    qc_path = deriv_root / "qc_stats.csv"
    with open(qc_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"# settings: {settings}"])
        writer.writerow(["subject", "volume_mm3", "pct_clip1"])
        writer.writerows(qc_rows)
    print(f"\nQC stats saved to -> {qc_path}")


if __name__ == "__main__":
    main()

    # python3 run_prepro_freesurfer.py --bids_dir /home/data/vub_ms/BIDS --out_dir /home/dknguyen/Documents/2_Works/1_FLAIR_MS/data/vub_ms/derivatives/dof6 --res 1 --dof 6

# export FREESURFER_HOME=$HOME/software/freesurfer
# source $FREESURFER_HOME/SetUpFreeSurfer.sh
# export FS_LICENSE=/home/dknguyen/software/freesurfer/.license
