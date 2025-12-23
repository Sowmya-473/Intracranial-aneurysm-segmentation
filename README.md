# RSNA Intracranial Aneurysm Segmentation (3D UNet)

## Overview
This project implements a **3D deep learning pipeline for intracranial aneurysm segmentation** using CT angiography scans from the RSNA Intracranial Aneurysm Detection dataset. The workflow converts raw **DICOM slices into 3D volumes**, applies medical imaging–specific preprocessing, and trains a **3D U-Net** model using **MONAI** and **PyTorch**.

The project is designed to be compatible with **Kaggle’s RSNA inference server**, making it suitable for both offline experimentation and competition-style evaluation.

---

## Objectives
- Convert DICOM CT slices into normalized 3D volumes
- Leverage provided NIfTI segmentation masks
- Train a robust 3D UNet for aneurysm segmentation
- Evaluate performance using Dice-based metrics
- Visualize predicted vs ground-truth segmentation masks

---

## Dataset
- **RSNA Intracranial Aneurysm Detection**
- Input: DICOM CT angiography series
- Labels: NIfTI (`.nii.gz`) vessel / aneurysm segmentations
- Structure:
  - Each `SeriesInstanceUID` corresponds to a 3D scan
  - Segmentation files are matched using normalized UIDs

---

## Pipeline

### 1. Data Exploration
- Traverse series folders
- Load and visualize individual DICOM slices using `pydicom`

### 2. Preprocessing
- Convert DICOM slices → 3D volumes
- CT windowing and intensity normalization
- Orientation standardization (RAS)
- Padding/cropping to uniform volume size `(128 × 128 × 128)`
- Ensure spatial dimensions are divisible by 16 (UNet safety)

### 3. Data Augmentation
- Random flips along all spatial axes
- Random 90° rotations
- Random contrast adjustment

### 4. Model
- Architecture: **3D U-Net**
- Framework: MONAI
- Configuration:
  - 5-level encoder–decoder
  - Residual units
  - Sigmoid output for binary segmentation

### 5. Training
- Loss: Dice + Cross Entropy (DiceCELoss)
- Optimizer: Adam
- Validation: Sliding window inference
- Metric: Dice coefficient
- Best model checkpointing based on validation Dice

### 6. Visualization
- Overlay predicted segmentation masks
- Compare against ground truth mid-slice

---

## Project Structure
```text
├── fda-review-1.ipynb             # (Replace with actual notebook name if needed)
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── data/
│   ├── series/                    # DICOM CT scan series
│   └── segmentations/             # NIfTI ground-truth masks
├── models/
│   └── best_patch_unet.pth        # Best saved 3D UNet model
└── results/
    ├── confusion_matrices/
    └── metrics.csv
```

Tech Stack:
```
- Python
- PyTorch
- MONAI
- NumPy, Pandas
- pydicom, nibabel
- Matplotlib
- Kaggle RSNA inference API
```

How to Run:
1. Install dependencies
```
pip install -r requirements.txt
```

2. Run training notebook
```
jupyter notebook fda-review-1.ipynb
```
## Results
- The 3D U-Net successfully learns volumetric vascular structures  
- Dice score improves consistently across training epochs  
- Sliding-window inference enables memory-efficient 3D prediction  
- Visual inspection shows strong overlap with ground-truth segmentation masks  

---

## Key Challenges Addressed
- Handling large 3D medical volumes under GPU memory constraints  
- Accurate alignment between DICOM volumes and NIfTI segmentation masks  
- Managing irregular scan sizes across different series  
- Patch-safe training for 3D U-Net architectures  
- Compatibility with competition-style inference constraints  

---

## Future Improvements
- Increase training epochs and dataset coverage  
- Extend to multi-class segmentation (vessels vs aneurysms)  
- Apply post-processing using connected component analysis  
- Explore transformer-based 3D segmentation models  
- Optimize inference for real-time or near–real-time deployment  
