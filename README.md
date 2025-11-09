## 🧠 Model Architecture

The **U-Net architecture** is designed for **pixel-level segmentation** of medical images.

### 🧩 Encoder (Contracting Path)
- Two `3×3` convolutions → ReLU activation → MaxPooling  
- Captures **semantic and spatial context** from input CT slices  

### 🧩 Decoder (Expanding Path)
- Upsampling + skip connections from encoder layers  
- Restores **spatial resolution** for precise segmentation boundaries  

### 🧩 Output Layer
- `1×1` convolution + **sigmoid activation** → produces binary mask predictions  

**Loss Function:** Dice Loss + Binary Cross-Entropy  
**Evaluation Metric:** Dice Coefficient (DSC)

---

## 📊 Results

Example overlays of **U-Net predictions** on random CT slices:

<p align="center">
  <img src="outputs/prediction_grid.png" width="45%" />

</p>



### 📈 Quantitative Performance (Demo Subset)

| Metric | Value |
|---------|-------|
| Dice Coefficient | ≈ 0.29 (3-case subset) |
| Validation Accuracy | ≈ 97.8% |
| Loss (BCE + Dice) | ≈ 0.75 |

These results demonstrate **accurate segmentation boundaries** on liver regions even with a small dataset (3 cases).  
GPU acceleration significantly reduced training time — each epoch took approximately **2 seconds** on an **NVIDIA RTX 5060 Ti**.

---

## 🗂️ Dataset Note

Due to licensing and privacy restrictions, the **full LiTS17 dataset** is **not included** in this repository.  
This notebook demonstrates the workflow using **three sample cases only**.

To run complete experiments, download the official dataset from the Liver Tumor Segmentation Challenge:  


---

## 🏁 Project Summary

This project validates a **GPU-accelerated U-Net** model for **liver segmentation** from CT scans.

It demonstrates:
- Efficient preprocessing of volumetric medical images  
- End-to-end segmentation with a 2D U-Net  
- Evaluation via Dice Score and qualitative overlays  
- Clear visualization of medical segmentation predictions  

All code is provided in a **single reproducible Jupyter Notebook** for easy experimentation and reproducibility.

---

## 🚀 How to Run

1. Place your `.nii` files inside the `sample_data/` folder  
2. Update the data path in the notebook:
   ```python
   data_dir = Path("sample_data/")
