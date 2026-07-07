import os
import subprocess
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np


def setup_environment(fs_home, fsl_dir, fs_license):
    """Thiet lap bien moi truong cho FreeSurfer va FSL"""
    os.environ["FREESURFER_HOME"] = fs_home
    os.environ["FSLDIR"] = fsl_dir
    os.environ["FS_LICENSE"] = fs_license
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    os.environ["PATH"] += f":{fs_home}/bin:{fsl_dir}/bin"
    return "submodules/Wood_2022/Data/MNI152_T1_1mm_brain.nii"


def run_cmd(cmd, step=""):
    """Thuc thi lenh shell, raise loi neu that bai thay vi im lang bo qua (FIX #3)"""
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"[{step}] that bai: {res.stderr.strip()[:300]}")
    return res.stdout.strip()


def normalize_intensity_01(input_p, output_p):
    """Chuan hoa cuong do ve [0,1] theo percentile 99 de tranh outlier"""
    img = nib.load(input_p)
    data = np.maximum(img.get_fdata(), 0)
    p0, p99 = np.min(data), np.percentile(data, 99)
    if p99 - p0 > 0:
        data = (data - p0) / (p99 - p0)
    data = np.clip(data, 0, 1)
    nib.save(nib.Nifti1Image(data.astype(np.float32), img.affine, img.header), output_p)


def process_file(input_p, output_p, mni_ref, res, dof, strip_border=3):
    """Reorient -> RobustFOV -> Conform -> N3 bias correct -> SynthStrip -> FLIRT -> Crop -> Normalize"""
    os.makedirs(os.path.dirname(output_p), exist_ok=True)

    tmp_base = output_p.replace(".nii.gz", "_TMP")
    reoriented = f"{tmp_base}_reorient.nii.gz"
    robust_roi = f"{tmp_base}_robust.nii.gz"
    std_1mm    = f"{tmp_base}_1.nii.gz"
    biascorr   = f"{tmp_base}_n3.nii.gz"
    stripped   = f"{tmp_base}_s.nii.gz"
    affine     = f"{tmp_base}_a.nii.gz"
    resampled  = f"{tmp_base}_r.nii.gz"
    crop       = f"{tmp_base}_crop.nii.gz"
    temp_files = [reoriented, robust_roi, std_1mm, biascorr, stripped, affine, resampled, crop]

    try:
        # 0. Chuan huong + cat co/vai thua
        run_cmd(f"fslreorient2std {input_p} {reoriented}", "reorient2std")
        run_cmd(f"robustfov -i {reoriented} -r {robust_roi}", "robustfov")

        # 1. Conform ve 1mm isotropic
        run_cmd(f"mri_convert {robust_roi} {std_1mm} --conform", "conform")

        # 2. Bias field correction (N3) - FIX #1: truoc day pipeline thieu buoc nay
        run_cmd(f"mri_nu_correct.mni --i {std_1mm} --o {biascorr} --n 3", "bias_correct")

        # 3. Skull-strip bang SynthStrip (deep-learning, robust hon watershed voi anh co lesion) - FIX #2
        run_cmd(f"mri_synthstrip -i {biascorr} -o {stripped} -b {strip_border}", "synthstrip")

        # 4. Alignment ve khong gian MNI
        run_cmd(
            f"flirt -in {stripped} -ref {mni_ref} -out {affine} -dof {dof} "
            f"-searchrx -180 180 -searchry -180 180 -searchrz -180 180",
            "flirt",
        )

        # 5. Resample neu chon 2mm
        working_file, grid_lim = affine, 256
        if res == 2:
            run_cmd(f"mri_convert {affine} {resampled} --voxsize 2 2 2", "resample_2mm")
            working_file, grid_lim = resampled, 128

        # 6. Crop theo chuan StyleGAN3D, canh bao neu bbox nao vuot box crop - FIX #4
        dx, dy, dz = (160, 192, 224) if res == 1 else (80, 96, 112)
        bbox = run_cmd(f"fslstats {working_file} -w", "fslstats_bbox").split()
        mx, my, mz = (int(bbox[0]) + int(bbox[1]) / 2,
                      int(bbox[2]) + int(bbox[3]) / 2,
                      int(bbox[4]) + int(bbox[5]) / 2)
        xsize, ysize, zsize = int(bbox[1]), int(bbox[3]), int(bbox[5])
        if xsize > dx or ysize > dy or zsize > dz:
            print(f"    [!] CANH BAO: bbox nao ({xsize},{ysize},{zsize}) vuot crop size "
                  f"({dx},{dy},{dz}) - co the bi cat mat mo nao")

        cx = int(max(0, min(grid_lim - dx, mx - dx / 2)))
        cy = int(max(0, min(grid_lim - dy, my - dy / 2)))
        cz = int(max(0, min(grid_lim - dz, mz - dz / 2)))
        run_cmd(f"fslroi {working_file} {crop} {cx} {dx} {cy} {dy} {cz} {dz}", "fslroi")

        # 7. Chuan hoa cuong do ve [0,1]
        normalize_intensity_01(crop, output_p)
        return os.path.exists(output_p)

    except Exception as e:
        print(f"    [x] LOI xu ly {input_p}: {e}")
        return False

    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)


def main():
    parser = argparse.ArgumentParser(description="BIDS Preprocessing for 3D-StyleGAN (FSL/FreeSurfer only)")
    parser.add_argument("--bids_dir", required=True, help="Thu muc goc BIDS (Read-only)")
    parser.add_argument("--out_dir", required=True, help="Thu muc luu ket qua")
    parser.add_argument("--res", type=int, choices=[1, 2], default=2, help="Do phan giai (1mm hoac 2mm)")
    parser.add_argument("--dof", type=int, choices=[6, 12], default=12, help="DOF cho Alignment")
    parser.add_argument("--skip", action="store_true", help="Bo qua neu file da ton tai")
    parser.add_argument("--strip_border", type=int, default=1, help="SynthStrip border (mm)")
    parser.add_argument("--fs_home", default="/home/dknguyen/software/freesurfer")
    parser.add_argument("--fsl_dir", default="/home/dknguyen/fsl")
    parser.add_argument("--license", default="/home/dknguyen/software/freesurfer/.license")
    args = parser.parse_args()

    mni_ref = setup_environment(args.fs_home, args.fsl_dir, args.license)
    bids_root = Path(args.bids_dir).resolve()
    deriv_root = Path(args.out_dir).resolve()

    t1w_list = list(bids_root.glob("sub-*/anat/*_T1w.nii.gz")) + \
               list(bids_root.glob("sub-*/ses-*/anat/*_T1w.nii.gz"))

    print(f"--- BIDS Pipeline (FSL/FreeSurfer) --- Input: {bids_root} | Output: {deriv_root} | Found {len(t1w_list)} files.")

    for t1w_path in t1w_list:
        relative_p = t1w_path.relative_to(bids_root)
        final_out_path = deriv_root / relative_p.parent / t1w_path.name

        if args.skip and final_out_path.exists():
            print(f"[-] Skip: {t1w_path.name}")
            continue

        print(f"[*] Processing: {t1w_path.name}")
        if process_file(str(t1w_path), str(final_out_path), mni_ref, args.res, args.dof, args.strip_border):
            print(f"    Saved -> {final_out_path}")


if __name__ == "__main__":
    main()

    # python3 run_prepro_freesurfer_v2.py --bids_dir /home/data/vub_ms/BIDS --out_dir /home/dknguyen/Documents/2_Works/1_FLAIR_MS/data/vub_ms/derivatives/dof6 --res 1 --dof 6
