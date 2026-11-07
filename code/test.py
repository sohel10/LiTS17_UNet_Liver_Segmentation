"""
Preprocess all 20 LiTS17 volumes (extracted)
Run from terminal: python preprocess_lits17.py
"""

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# CONFIG
work_dir = r"C:\Users\sohel\Downloads\Ai_health\LITS17"
output_dir = r"C:\Users\sohel\Downloads\Ai_health\preprocessed"
num_volumes = 20  # ← CHANGED FROM 131 to 20

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("PREPROCESSING ALL 20 LITS17 VOLUMES")
print("="*60)

# STEP 1: FIND MEDIAN DEPTH
print("\nSTEP 1: Finding median depth...")
depths = []

for i in range(num_volumes):
    seg = nib.load(os.path.join(work_dir, f'segmentation-{i}.nii')).get_fdata()
    depths.append(seg.shape[2])
    
    if (i + 1) % 5 == 0:
        print(f"  Processed {i+1}/{num_volumes}")

median_depth = int(np.median(depths))
print(f"Median depth: {median_depth}")

# STEP 2: STANDARDIZE, NORMALIZE, RESIZE
print("\nSTEP 2: Standardizing all volumes...")

X_all = []
y_all = []

for i in range(num_volumes):
    ct = nib.load(os.path.join(work_dir, f'volume-{i}.nii')).get_fdata()
    seg = nib.load(os.path.join(work_dir, f'segmentation-{i}.nii')).get_fdata()
    
    # Standardize (pad/crop)
    if ct.shape[2] < median_depth:
        add_slices = median_depth - ct.shape[2]
        ct = np.pad(ct, ((0, 0), (0, 0), (0, add_slices)))
        seg = np.pad(seg, ((0, 0), (0, 0), (0, add_slices)))
    else:
        ct = ct[:, :, :median_depth]
        seg = seg[:, :, :median_depth]
    
    # Normalize
    ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-5)
    
    # Resize (0.25, 0.25, 0.5)
    ct = zoom(ct, (0.25, 0.25, 0.5))
    seg = zoom(seg, (0.25, 0.25, 0.5), order=0)
    
    X_all.append(ct)
    y_all.append(seg)
    
    if (i + 1) % 5 == 0:
        print(f"  Processed {i+1}/{num_volumes}")

# Convert to arrays and add channel dimension
X = np.array(X_all)
y = np.array(y_all)
X = X[:, :, :, :, np.newaxis]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Crop to match
X = X[:, :, :, :268, :]
y = y[:, :, :, :268]

# STEP 3: SPLIT DATA
print("\nSTEP 3: Splitting data (60/20/20)...")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

# One-hot encode
y_train = to_categorical(y_train, num_classes=3)
y_val = to_categorical(y_val, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Val:   {X_val.shape}, {y_val.shape}")
print(f"Test:  {X_test.shape}, {y_test.shape}")

# STEP 4: SAVE PREPROCESSED DATA
print("\nSTEP 4: Saving preprocessed data...")

np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

print(f"✓ Saved to {output_dir}")

# STEP 5: ANALYZE CLASS BALANCE
print("\nSTEP 5: Class balance analysis...")

y_train_labels = np.argmax(y_train, axis=-1).flatten()

background_count = np.sum(y_train_labels == 0)
liver_count = np.sum(y_train_labels == 1)
tumor_count = np.sum(y_train_labels == 2)
total = background_count + liver_count + tumor_count

print(f"Background: {background_count:,} ({100*background_count/total:.1f}%)")
print(f"Liver:      {liver_count:,} ({100*liver_count/total:.1f}%)")
print(f"Tumor:      {tumor_count:,} ({100*tumor_count/total:.1f}%)")

# Calculate weights
weight_bg = total / (3 * background_count)
weight_liver = total / (3 * liver_count)
weight_tumor = total / (3 * tumor_count)

print(f"\nClass weights:")
print(f"Background: {weight_bg:.4f}")
print(f"Liver:      {weight_liver:.4f}")
print(f"Tumor:      {weight_tumor:.4f}")

print("\n" + "="*60)
print("✓ PREPROCESSING COMPLETE!")
print("="*60)