import csv
import argparse
from pathlib import Path


def parse_qc_txt(qc_path):
    info = dict(line.split(":", 1) for line in qc_path.read_text().splitlines() if ":" in line)
    info = {k.strip(): v.strip() for k, v in info.items()}
    try:
        # bbox_warning duoc them sau, cac file qc.txt cu khong co -> de trong thay vi doan la False
        return (info["subject"], float(info["volume_mm3"]), float(info["pct_clip1"]),
                info.get("bbox_warning", ""))
    except (KeyError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Tong hop qc_stats.csv tu cac file *_qc.txt (dung khi run bi ngat giua chung)")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    deriv_root = Path(args.out_dir).resolve()
    qc_rows = sorted(filter(None, (parse_qc_txt(p) for p in deriv_root.glob("**/*_qc.txt"))))

    print(f"{'subject':<40}{'volume_mm3':>15}{'pct_clip1':>12}{'bbox_warning':>15}")
    for s, v, p, w in qc_rows:
        print(f"{s:<40}{v:>15.1f}{p:>12.2f}{w:>15}")

    qc_path_out = deriv_root / "qc_stats_recovered.csv"
    with open(qc_path_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "volume_mm3", "pct_clip1", "bbox_warning"])
        writer.writerows(qc_rows)
    print(f"\nDa luu -> {qc_path_out}")


if __name__ == "__main__":
    main()
