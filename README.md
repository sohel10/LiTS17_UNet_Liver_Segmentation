# 🧠 Liver Tumor Segmentation with U-Net (LiTS17, GPU-Accelerated)

### Author: Sohel Ahmed, PhD  
**Goal:** Demonstrate end-to-end medical image segmentation using a 2D U-Net trained on a subset of the **LiTS17** (Liver Tumor Segmentation) dataset.

---

## ⚙️ Overview
This project showcases a deep-learning workflow for liver segmentation from abdominal CT scans:

- Loads **NIfTI** volumes (`.nii`) using **NiBabel**  
- Extracts 2D axial slices for training  
- Trains a **2D U-Net** in TensorFlow/Keras (GPU-accelerated)  
- Combines **Dice + Binary Cross-Entropy loss**  
- Produces clear overlay visualizations of predicted masks  

---

## 🧩 Environment
| Library | Version |
|----------|----------|
| Python | 3.10 |
| TensorFlow | 2.21 |
| nibabel | ≥ 5.0 |
| NumPy | ≥ 1.26 |
| scikit-learn | ≥ 1.3 |
| Matplotlib | ≥ 3.8 |

Install all dependencies:
```bash
pip install -r requirements.txt
📊 Results

Example overlays of U-Net predictions on random CT slices:🗂️ Dataset Note

Due to data-sharing restrictions, the full LiTS17 dataset is not included.
This notebook demonstrates the pipeline using three sample cases only.
For full experiments, download LiTS17 from the official source:
👉 https://competitions.codalab.org/competitions/17094🏁 Project Summary

This demo validates a GPU-accelerated 2D U-Net architecture for clinical-grade liver segmentation.
It highlights medical imaging preprocessing, model training, and visualization—all in a single reproducible Jupyter Notebook.
