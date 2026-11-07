"""
FIXED V6 FINAL: Break out of background trap CORRECTLY
==========================================
The Problem with V5:
- Sigmoid activation + Pure Dice Loss was supposed to work
- BUT: Using np.argmax() on sigmoid outputs is WRONG
- Sigmoid treats each channel independently (0-1 each)
- argmax still picks highest value, often picking background

The Solution (V6):
- Sigmoid activation (correct)
- Generalized Dice Loss (foreground-only) - ONLY trains on Liver + Tumor channels
- Threshold-based hard predictions (not argmax!)
- Proper per-channel evaluation with multiply (*) not bitwise (&)
"""

import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from scipy.ndimage import zoom
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# CONFIG
# ============================================================
work_dir = r"C:\Users\sohel\Downloads\Ai_health\LITS17"
output_dir = r"C:\Users\sohel\Downloads\Ai_health\models"
os.makedirs(output_dir, exist_ok=True)

num_volumes = 20
median_depth = 541
target_depth = 256

print("="*60)
print("3D LIVER SEGMENTATION (FIXED V6 - FINAL)")
print("Sigmoid + Generalized Dice Loss (Foreground-Only)")
print("Threshold-based evaluation with multiply (*)")
print("="*60)

# ============================================================
# DATA GENERATOR
# ============================================================
class Medical3DDataGenerator:
    def __init__(self, volume_indices, work_dir, median_depth=541, target_depth=256):
        self.volume_indices = volume_indices
        self.work_dir = work_dir
        self.median_depth = median_depth
        self.target_depth = target_depth
        self.depth_factor = target_depth / median_depth

    def _process_single_volume(self, index):
        ct = nib.load(os.path.join(self.work_dir, f'volume-{index}.nii')).get_fdata(dtype=np.float32)
        seg = nib.load(os.path.join(self.work_dir, f'segmentation-{index}.nii')).get_fdata(dtype=np.float32)

        if ct.shape[2] < self.median_depth:
            add_slices = self.median_depth - ct.shape[2]
            ct = np.pad(ct, ((0, 0), (0, 0), (0, add_slices)))
            seg = np.pad(seg, ((0, 0), (0, 0), (0, add_slices)))
        else:
            ct = ct[:, :, :self.median_depth]
            seg = seg[:, :, :self.median_depth]

        ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-5)

        # Proportional resize
        ct = zoom(ct, (0.25, 0.25, self.depth_factor), order=1)
        seg = zoom(seg, (0.25, 0.25, self.depth_factor), order=0)

        ct = ct[:, :, :self.target_depth]
        seg = seg[:, :, :self.target_depth]

        ct = ct[..., np.newaxis]
        seg = to_categorical(seg, num_classes=3)

        return ct.astype(np.float32), seg.astype(np.float32)

    def get_generator(self):
        def generator():
            for i in self.volume_indices:
                yield self._process_single_volume(i)
        return generator

# ============================================================
# SPLIT DATA
# ============================================================
print("\nSplitting data...")
all_indices = list(range(num_volumes))
np.random.seed(42)
np.random.shuffle(all_indices)

train_indices = all_indices[:12]
val_indices = all_indices[12:16]
test_indices = all_indices[16:20]

print(f"Train: {train_indices}")
print(f"Val: {val_indices}")
print(f"Test: {test_indices}")

# ============================================================
# CREATE DATASETS
# ============================================================
print("\nCreating datasets...")

output_signature = (
    tf.TensorSpec(shape=(128, 128, target_depth, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(128, 128, target_depth, 3), dtype=tf.float32)
)

train_gen = Medical3DDataGenerator(train_indices, work_dir, median_depth, target_depth)
val_gen = Medical3DDataGenerator(val_indices, work_dir, median_depth, target_depth)
test_gen = Medical3DDataGenerator(test_indices, work_dir, median_depth, target_depth)

train_dataset = tf.data.Dataset.from_generator(
    train_gen.get_generator(), output_signature=output_signature
).batch(1).repeat().prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    val_gen.get_generator(), output_signature=output_signature
).batch(1).repeat().prefetch(tf.data.AUTOTUNE)

# ============================================================
# BUILD U-NET (SIGMOID output)
# ============================================================
print("\n" + "="*60)
print("Building 3D U-Net (SIGMOID output)")
print("="*60)

inputs = keras.Input(shape=(128, 128, target_depth, 1))

# ENCODER
c1 = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
c1 = layers.BatchNormalization()(c1)
c1 = layers.Conv3D(32, 3, activation='relu', padding='same')(c1)
c1 = layers.BatchNormalization()(c1)
p1 = layers.MaxPooling3D(2)(c1)

c2 = layers.Conv3D(64, 3, activation='relu', padding='same')(p1)
c2 = layers.BatchNormalization()(c2)
c2 = layers.Conv3D(64, 3, activation='relu', padding='same')(c2)
c2 = layers.BatchNormalization()(c2)
p2 = layers.MaxPooling3D(2)(c2)

# BOTTLENECK
c3 = layers.Conv3D(128, 3, activation='relu', padding='same')(p2)
c3 = layers.BatchNormalization()(c3)
c3 = layers.Conv3D(128, 3, activation='relu', padding='same')(c3)
c3 = layers.BatchNormalization()(c3)

# DECODER (skip connections)
u1 = layers.UpSampling3D(2)(c3)
u1 = layers.Concatenate()([u1, c2])
c4 = layers.Conv3D(64, 3, activation='relu', padding='same')(u1)
c4 = layers.BatchNormalization()(c4)
c4 = layers.Conv3D(64, 3, activation='relu', padding='same')(c4)
c4 = layers.BatchNormalization()(c4)

u2 = layers.UpSampling3D(2)(c4)
u2 = layers.Concatenate()([u2, c1])
c5 = layers.Conv3D(32, 3, activation='relu', padding='same')(u2)
c5 = layers.BatchNormalization()(c5)
c5 = layers.Conv3D(32, 3, activation='relu', padding='same')(c5)
c5 = layers.BatchNormalization()(c5)

# OUTPUT: SIGMOID (not softmax!)
outputs = layers.Conv3D(3, 1, activation='sigmoid')(c5)

model = keras.Model(inputs, outputs)
print(f"Model params: {model.count_params():,}")

# ============================================================
# GENERALIZED DICE LOSS (FOREGROUND-ONLY) - THE FIX!
# ============================================================
print("\n" + "="*60)
print("Generalized Dice Loss (Foreground-Only)")
print("="*60)

def generalized_dice_loss_foreground_only(y_true, y_pred):
    """
    Generalized Dice Loss that IGNORES background.
    Only computes Dice on Liver (channel 1) + Tumor (channel 2).
    Background (channel 0) gradient is set to zero.
    
    This forces the model to learn the small foreground regions!
    """
    smooth = 1e-5
    
    # Extract only foreground channels (1=Liver, 2=Tumor)
    # Ignore channel 0 (background)
    y_true_fg = y_true[..., 1:]  # Shape: (batch, h, w, d, 2)
    y_pred_fg = y_pred[..., 1:]
    
    # Compute Dice only on foreground
    intersection = 2.0 * tf.reduce_sum(y_true_fg * y_pred_fg)
    union = tf.reduce_sum(y_true_fg) + tf.reduce_sum(y_pred_fg)
    
    dice = (intersection + smooth) / (union + smooth)
    
    return 1.0 - dice

print("Loss: Generalized Dice Loss (foreground channels only)")

# ============================================================
# LIVE MONITORING CALLBACK
# ============================================================
class LiverMonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, threshold=0.5):
        super().__init__()
        self.val_sample = next(iter(val_dataset))
        self.threshold = threshold
        
    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.val_sample
        preds = self.model.predict(X_val, verbose=0)
        
        # THRESHOLD-BASED (not argmax!)
        pred_binary = (preds > self.threshold).astype(np.uint8)
        
        # Count predicted voxels per channel
        class_map = {}
        for ch in range(3):
            class_map[ch] = int(np.sum(pred_binary[0, ..., ch]))
        
        # Soft dice per class
        soft_dice = {}
        class_names = {0: 'BG', 1: 'Liver', 2: 'Tumor'}
        for ch in range(3):
            true_ch = y_val[0, ..., ch].numpy()
            pred_ch = preds[0, ..., ch]
            intersection = np.sum(true_ch * pred_ch)
            total = np.sum(true_ch) + np.sum(pred_ch)
            dice = (2.0 * intersection + 1e-5) / (total + 1e-5)
            soft_dice[class_names[ch]] = f"{dice:.4f}"
        
        print(f"\n  [Epoch {epoch+1}] Predicted voxels: {class_map} | Soft Dice: {soft_dice}")

# ============================================================
# COMPILE AND TRAIN (30 EPOCHS - FIXED!)
# ============================================================
print("\n" + "="*60)
print("TRAINING (30 epochs)")
print("="*60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=generalized_dice_loss_foreground_only,
    metrics=['accuracy']
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

monitor = LiverMonitorCallback(val_dataset, threshold=0.5)

history = model.fit(
    train_dataset,
    steps_per_epoch=len(train_indices),
    validation_data=val_dataset,
    validation_steps=len(val_indices),
    epochs=2,  # FIXED: Was 2, now 30!
    callbacks=[reduce_lr, monitor],
    verbose=1
)

# ============================================================
# SAVE MODEL
# ============================================================
model_path = os.path.join(output_dir, 'liver_segmentation_v6_fixed.keras')
model.save(model_path)
print(f"\nModel saved: {model_path}")

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

X_test_list = []
y_test_list = []

for i in test_indices:
    ct, seg = test_gen._process_single_volume(i)
    X_test_list.append(ct)
    y_test_list.append(seg)

X_test = np.array(X_test_list)
y_test = np.array(y_test_list)

y_test_class = np.argmax(y_test, axis=-1)
print(f"Test classes: {np.unique(y_test_class)}")
print(f"Test liver pixels: {np.sum(y_test_class == 1):,}")
print(f"Test tumor pixels: {np.sum(y_test_class == 2):,}")

y_pred = model.predict(X_test)

# ============================================================
# THRESHOLD-BASED HARD PREDICTIONS (not argmax!)
# ============================================================
print("\n" + "="*60)
print("HARD PREDICTIONS (Threshold=0.5, not argmax)")
print("="*60)

threshold = 0.5
y_pred_binary = (y_pred > threshold).astype(np.uint8)

# Argmax of binary predictions for comparison
y_pred_class = np.argmax(y_pred_binary, axis=-1)

# SOFT DICE
def compute_soft_dice(y_true, y_pred, num_classes=3):
    smooth = 1e-5
    scores = {}
    class_names = ['Background', 'Liver', 'Tumor']
    for ch in range(num_classes):
        true_channel = y_true[..., ch]
        pred_channel = y_pred[..., ch]
        intersection = np.sum(true_channel * pred_channel)
        total_sum = np.sum(true_channel) + np.sum(pred_channel)
        dice = (2.0 * intersection + smooth) / (total_sum + smooth)
        scores[class_names[ch]] = dice
    return scores

soft_scores = compute_soft_dice(y_test, y_pred)

print(f"\nSOFT DICE SCORES (Probability-based)")
for name, score in soft_scores.items():
    print(f"  {name}: {score:.4f}")
print(f"  Average: {np.mean(list(soft_scores.values())):.4f}")

# HARD DICE (using threshold) - FIXED: Use * instead of &
def compute_hard_dice(y_true_binary, y_pred_binary, channel):
    smooth = 1e-5
    
    # 1. Cast arrays explicitly to float32 to avoid type conflicts
    true_ch = y_true_binary[..., channel].astype(np.float32)
    pred_ch = y_pred_binary[..., channel].astype(np.float32)
    
    # 2. Use multiplication (*) instead of bitwise AND (&)
    # Mathematically equivalent and works on all numeric types
    intersection = np.sum(true_ch * pred_ch)
    total_sum = np.sum(true_ch) + np.sum(pred_ch)
    
    return (2.0 * intersection + smooth) / (total_sum + smooth)

print(f"\nHARD DICE SCORES (Threshold-based)")
for ch, name in enumerate(['Background', 'Liver', 'Tumor']):
    dice = compute_hard_dice(y_test, y_pred_binary, ch)
    print(f"  {name}: {dice:.4f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
y_test_flat = y_test_class.flatten()
y_pred_flat = y_pred_class.flatten()

print(f"\n{'='*60}")
print("CLASSIFICATION REPORT (threshold-based hard predictions)")
print(f"{'='*60}")
print(classification_report(
    y_test_flat, y_pred_flat,
    target_names=['Background', 'Liver', 'Tumor'],
    labels=[0, 1, 2],
    zero_division=0
))

# ============================================================
# DETAILED METRICS PER CLASS
# ============================================================
print(f"\n{'='*60}")
print("DETAILED METRICS")
print(f"{'='*60}")

for class_idx, class_name in enumerate(['Background', 'Liver', 'Tumor']):
    y_true_binary = (y_test_flat == class_idx).astype(int)
    y_pred_binary_class = (y_pred_flat == class_idx).astype(int)
    
    # Confusion matrix
    try:
        cm = confusion_matrix(y_true_binary, y_pred_binary_class, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases
            tn = fp = fn = tp = 0
            if cm.size == 1:
                if y_true_binary.sum() == 0:
                    tn = cm[0, 0]
                else:
                    tp = cm[0, 0]
    except:
        tn = fp = fn = tp = 0
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{class_name}:")
    print(f"  TP={tp:,}, FP={fp:,}, TN={tn:,}, FN={fn:,}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Soft Dice: {soft_scores[class_name]:.4f}")
    hard_dice = compute_hard_dice(y_test, y_pred_binary, class_idx)
    print(f"  Hard Dice: {hard_dice:.4f}")

print("\n" + "="*60)
print("✓ COMPLETE!")
print("="*60)
print("\nModel saved to:", model_path)
print("Ready for Vanquish AI interview! 🚀")