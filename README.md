# Brain Image Preprocessing Pipeline

This script provides an automated pipeline for batch preprocessing of T1w and FLAIR brain Magnetic Resonance Imaging (MRI) scans. The pipeline is inspired by the methodology published by [Wood et al. 2022](https://github.com/MIDIconsortium/BrainAge) and is designed to work with datasets compliant with the BIDS (Brain Imaging Data Structure) standard.

## Key Features

*   **Automated BIDS Traversal**: Automatically finds and processes all `_T1w.nii.gz` and `_FLAIR.nii.gz` files within a BIDS-structured dataset.
*   **Comprehensive Preprocessing Pipeline**: Applies a series of standard processing steps, including:
    *   Reorientation to standard space.
    *   Registration to the MNI152 template.
    *   Skull-stripping (optional).
    *   Intensity normalization.
*   **Organized Storage**: Saves the processed images in the `derivatives` directory, following the BIDS standard for easy management and traceability.
*   **Visual Quality Control**: Automatically generates PNG images of central slices (axial, coronal, sagittal) for quick quality checks of the preprocessing results.

## Setup and Installation

### 1. Project Directory Structure

This script requires the `Wood_2022` submodule to be located in the `submodules/` directory. Ensure your project structure is as follows:

```
your_project_folder/
├── preprocess_images.py                 # Main script
├── requirements_preprocessing.txt  # Requirements file
├── README.md                     # This file
└── submodules/
    └── Wood_2022/                # Wood et al.'s repository
```

### 2. Python Environment

It is recommended to use Python `3.9` and create a virtual environment to avoid library conflicts.

```bash
# Create a virtual environment
python3.9 -m venv venv

# Activate the environment (Linux/macOS)
source venv/bin/activate

# Activate the environment (Windows)
.\\venv\\Scripts\\activate
```

### 3. Install Required Libraries

All necessary libraries are listed in the `requirements_preprocessing.txt` file. Install them with a single command:

```bash
pip install -r requirements_preprocessing.txt
```

### 4. Data Preparation

Your input dataset **must** be compliant with the **BIDS (Brain Imaging Data Structure)** standard. The script will search for files ending with `_T1w.nii.gz` and `_FLAIR.nii.gz`.

Example of a valid data structure:
```
my_bids_dataset/
├── sub-001/
│   └── anat/
│       ├── sub-001_T1w.nii.gz
│       └── sub-001_FLAIR.nii.gz
├── sub-002/
│   └── anat/
│       ├── sub-002_T1w.nii.gz
│       └── sub-002_FLAIR.nii.gz
└── ...
```

## Usage

Use a terminal or command prompt to run the script with the required command-line arguments.

### Command-line Syntax
```bash
python preprocess.py --dataset_root_path <path_to_dataset> --preprocessing_name <name_for_output> [options]
```

### Arguments

| Argument               | Required? | Description                                                                                             |
| ---------------------- | :-------: | ------------------------------------------------------------------------------------------------------- |
| `--dataset_root_path`  | **Yes**   | Absolute path to the BIDS dataset root directory.                                                       |
| `--preprocessing_name` | **Yes**   | Name for the pipeline, used to create a subdirectory in `derivatives` for the output (e.g., `wood2022_preproc`). |
| `--skull_strip`        |    No     | Add this flag to perform skull-stripping.                                                               |
| `--use_gpu`            |    No     | Add this flag to use the GPU (if supported by the underlying libraries).                                |

### Examples

**1. Basic preprocessing (without skull-stripping):**
```bash
python preprocess.py \\
    --dataset_root_path /path/to/my_bids_dataset \\
    --preprocessing_name wood2022_default
```

**2. Full preprocessing with skull-stripping and GPU usage:**
```bash
python preprocess.py \\
    --dataset_root_path /path/to/my_bids_dataset \\
    --preprocessing_name wood2022_skullstripped_gpu \\
    --skull_strip \\
    --use_gpu
```

## Output

The results will be saved in the `derivatives/<preprocessing_name>` directory within your dataset's root folder.

The output directory structure will be as follows:
```
my_bids_dataset/
└── derivatives/
    └── <preprocessing_name>/
        ├── sub-001/
        │   └── anat/
        │       ├── sub-001_T1w.nii.gz            # Preprocessed T1w image
        │       ├── sub-001_T1w_slice_1.png       # Axial slice image
        │       ├── sub-001_T1w_slice_2.png       # Coronal slice image
        │       ├── sub-001_T1w_slice_3.png       # Sagittal slice image
        │       ├── sub-001_FLAIR.nii.gz          # Preprocessed FLAIR image
        │       └── ...
        └── sub-002/
            └── ...
```

-   The `.nii.gz` files are the fully preprocessed 3D images.
-   The `.png` files are 2D images of the central slices, allowing for a quick visual check without needing specialized software.

## Important Notes
*   **Execution Time**: Preprocessing a single MRI scan can take several minutes. This process will be significantly longer when the `--skull_strip` option is enabled. Specifically, an 87x87x87 image takes approximately 4 seconds to complete.
*   **Working Directory Change**: The script temporarily changes the current working directory (`os.chdir`) to the `Wood_2022` submodule directory. This is necessary to ensure that relative paths within the submodule work correctly.
*   **BIDS Validation**: The BIDS validation function (`dataset_is_bids()`) is commented out in the script to speed up execution. You can uncomment it if you wish to perform this check before running the pipeline.




# Brain Image Preprocessing Pipeline

Script này cung cấp một pipeline tự động để tiền xử lý hàng loạt các ảnh cộng hưởng từ (MRI) não T1w và FLAIR. Pipeline được lấy cảm hứng từ phương pháp được công bố bởi [Wood et al. 2022](https://github.com/MIDIconsortium/BrainAge) và được thiết kế để hoạt động với các bộ dữ liệu tuân thủ chuẩn BIDS.

## Chức năng chính

*   **Duyệt dữ liệu BIDS**: Tự động tìm và xử lý tất cả các file `_T1w.nii.gz` và `_FLAIR.nii.gz` trong một bộ dữ liệu có cấu trúc BIDS.
*   **Pipeline tiền xử lý toàn diện**: Áp dụng một chuỗi các bước xử lý tiêu chuẩn bao gồm:
    *   Định hướng lại (Reorientation) về không gian chuẩn.
    *   Đăng ký (Registration) ảnh vào template MNI152.
    *   Loại bỏ hộp sọ (Skull-stripping) (tùy chọn).
    *   Chuẩn hóa cường độ (Intensity normalization).
*   **Lưu trữ có tổ chức**: Lưu các ảnh đã xử lý vào thư mục `derivatives` theo đúng cấu trúc BIDS, giúp dễ dàng quản lý và theo dõi.
*   **Kiểm tra chất lượng trực quan**: Tự động tạo ra các ảnh PNG của các lát cắt trung tâm (axial, coronal, sagittal) để kiểm tra nhanh chất lượng tiền xử lý.

## Cài đặt và Chuẩn bị

### 1. Cấu trúc Thư mục Dự án

Script này yêu cầu submodule `Wood_2022` phải được đặt trong thư mục `submodules/`. Hãy đảm bảo cấu trúc dự án của bạn như sau:

```
your_project_folder/
├── preprocess.py                 # Script chính
├── requirements_preprocessing.txt  # File yêu cầu thư viện
├── README.md                     # File này
└── submodules/
    └── Wood_2022/                # Repo của Wood et al.
```



### 2. Môi trường Python

Khuyến nghị sử dụng Python `3.9` và tạo một môi trường ảo để tránh xung đột thư viện.

```bash
# Tạo môi trường ảo
python3.9 -m venv venv

# Kích hoạt môi trường (Linux/macOS)
source venv/bin/activate

# Kích hoạt môi trường (Windows)
.\\venv\\Scripts\\activate
```

### 3. Cài đặt các thư viện cần thiết

Tất cả các thư viện cần thiết được liệt kê trong file `requirements_preprocessing.txt`. Cài đặt chúng bằng một lệnh duy nhất:

```bash
pip install -r requirements_preprocessing.txt
```



### 4. Chuẩn bị Dữ liệu

Bộ dữ liệu đầu vào của bạn **phải** tuân thủ chuẩn **BIDS (Brain Imaging Data Structure)**. Script sẽ tìm kiếm các file kết thúc bằng `_T1w.nii.gz` và `_FLAIR.nii.gz`.

Ví dụ về cấu trúc dữ liệu hợp lệ:
```
my_bids_dataset/
├── sub-001/
│   └── anat/
│       ├── sub-001_T1w.nii.gz
│       └── sub-001_FLAIR.nii.gz
├── sub-002/
│   └── anat/
│       ├── sub-002_T1w.nii.gz
│       └── sub-002_FLAIR.nii.gz
└── ...
```

## Cách chạy Script

Sử dụng terminal hoặc command prompt để chạy script với các đối số dòng lệnh cần thiết.

### Cú pháp Lệnh
```bash
python preprocess.py --dataset_root_path <path_to_dataset> --preprocessing_name <name_for_output> [options]
```

### Các đối số

| Argument               | Required? | Description                                                                                             |
| ---------------------- | :-------: | ------------------------------------------------------------------------------------------------------- |
| `--dataset_root_path`  | **Yes**   | Đường dẫn tuyệt đối đến thư mục gốc của bộ dữ liệu BIDS.                                                  |
| `--preprocessing_name` | **Yes**   | Tên của pipeline, sẽ được dùng để tạo thư mục con trong `derivatives` để lưu kết quả (ví dụ: `wood2022_preproc`). |
| `--skull_strip`        |    No     | Thêm cờ này để thực hiện loại bỏ hộp sọ (skull-stripping).                                             |
| `--use_gpu`            |    No     | Thêm cờ này để sử dụng GPU (nếu được hỗ trợ bởi các thư viện nền).                                     |

### Ví dụ

**1. Tiền xử lý cơ bản (không loại bỏ hộp sọ):**
```bash
python preprocess.py \\
    --dataset_root_path /path/to/my_bids_dataset \\
    --preprocessing_name wood2022_default
```

**2. Tiền xử lý đầy đủ với loại bỏ hộp sọ và sử dụng GPU:**
```bash
python preprocess.py \\
    --dataset_root_path /path/to/my_bids_dataset \\
    --preprocessing_name wood2022_skullstripped_gpu \\
    --skull_strip \\
    --use_gpu
```

## Kết quả Đầu ra

Kết quả sẽ được lưu trong thư mục `derivatives/<preprocessing_name>` bên trong thư mục dữ liệu gốc của bạn.

Cấu trúc thư mục đầu ra sẽ như sau:
```
my_bids_dataset/
└── derivatives/
    └── <preprocessing_name>/
        ├── sub-001/
        │   └── anat/
        │       ├── sub-001_T1w.nii.gz            # Ảnh T1w đã tiền xử lý
        │       ├── sub-001_T1w_slice_1.png       # Ảnh lát cắt axial
        │       ├── sub-001_T1w_slice_2.png       # Ảnh lát cắt coronal
        │       ├── sub-001_T1w_slice_3.png       # Ảnh lát cắt sagittal
        │       ├── sub-001_FLAIR.nii.gz          # Ảnh FLAIR đã tiền xử lý
        │       └── ...
        └── sub-002/
            └── ...
```

-   Các file `.nii.gz` là ảnh 3D đã được tiền xử lý đầy đủ.
-   Các file `.png` là ảnh 2D của các lát cắt trung tâm, giúp bạn kiểm tra nhanh kết quả mà không cần phần mềm chuyên dụng.

## Lưu ý Quan trọng
*   **Thời gian thực thi**: Tiền xử lý một ảnh MRI có thể mất vài phút. Quá trình này sẽ tốn nhiều thời gian hơn đáng kể khi bật tùy chọn `--skull_strip`. Cụ thể với ảnh 87x87x87 sẽ tốn 4 giây để hoàn thành.
*   **Thay đổi Thư mục làm việc**: Script sẽ tạm thời thay đổi thư mục làm việc hiện tại (`os.chdir`) sang thư mục của submodule `Wood_2022`. Điều này là cần thiết để đảm bảo các đường dẫn tương đối trong submodule hoạt động chính xác.
*   **Kiểm tra BIDS**: Chức năng kiểm tra tính hợp lệ của BIDS (`dataset_is_bids()`) đã được comment lại trong script để tăng tốc độ. Bạn có thể bỏ comment nếu muốn thực hiện kiểm tra này trước khi chạy pipeline.


