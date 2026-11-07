"""
FINAL V5: Break out of background trap
- Sigmoid activation (not softmax)
- Pure Dice Loss (no CCE)
- 30 epochs
- Live monitoring callback
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
from sklearn.metrics import classification_report

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
print("3D LIVER SEGMENTATION (FINAL V5)")
print("Sigmoid + Pure Dice Loss + 30 Epochs")
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
# BUILD U-NET (SIGMOID output instead of softmax!)
# ============================================================
print("\n" + "="*60)
print("Building 3D U-Net (SIGMOID output)")
print("="*60)

inputs = keras.Input(shape=(128, 128, target_depth, 1))

# ENCODER
c1 = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
c1 = layers.Conv3D(32, 3, activation='relu', padding='same')(c1)
p1 = layers.MaxPooling3D(2)(c1)

c2 = layers.Conv3D(64, 3, activation='relu', padding='same')(p1)
c2 = layers.Conv3D(64, 3, activation='relu', padding='same')(c2)
p2 = layers.MaxPooling3D(2)(c2)

# BOTTLENECK
c3 = layers.Conv3D(128, 3, activation='relu', padding='same')(p2)
c3 = layers.Conv3D(128, 3, activation='relu', padding='same')(c3)

# DECODER (skip connections)
u1 = layers.UpSampling3D(2)(c3)
u1 = layers.Concatenate()([u1, c2])
c4 = layers.Conv3D(64, 3, activation='relu', padding='same')(u1)
c4 = layers.Conv3D(64, 3, activation='relu', padding='same')(c4)

u2 = layers.UpSampling3D(2)(c4)
u2 = layers.Concatenate()([u2, c1])
c5 = layers.Conv3D(32, 3, activation='relu', padding='same')(u2)
c5 = layers.Conv3D(32, 3, activation='relu', padding='same')(c5)

# OUTPUT: SIGMOID (not softmax!)
outputs = layers.Conv3D(3, 1, activation='sigmoid')(c5)

model = keras.Model(inputs, outputs)
print(f"Model params: {model.count_params():,}")

# ============================================================
# PURE DICE LOSS (no CCE!)
# ============================================================
print("\n" + "="*60)
print("Pure Dice Loss (no CCE)")
print("="*60)

def pure_dice_loss(y_true, y_pred):
    """
    Multi-class soft Dice Loss.
    Ignores background size by looking only at overlap.
    """
    smooth = 1e-5
    
    y_true_f = tf.reshape(y_true, [-1, 3])
    y_pred_f = tf.reshape(y_pred, [-1, 3])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    total = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
    
    dice_score = (2.0 * intersection + smooth) / (total + smooth)
    
    return 1.0 - tf.reduce_mean(dice_score)

print("Loss: Pure multi-class Dice (no cross-entropy)")

# ============================================================
# LIVE MONITORING CALLBACK
# ============================================================
class LiverMonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset):
        super().__init__()
        self.val_sample = next(iter(val_dataset))
        
    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.val_sample
        preds = self.model.predict(X_val, verbose=0)
        
        # Hard predictions
        pred_classes = np.argmax(preds[0], axis=-1)
        unique_classes, counts = np.unique(pred_classes, return_counts=True)
        class_map = dict(zip(unique_classes.astype(int), counts))
        
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
        
        print(f"\n  [Epoch {epoch+1}] Pixels: {class_map} | Soft Dice: {soft_dice}")

# ============================================================
# COMPILE AND TRAIN (20 epochs)
# ============================================================
print("\n" + "="*60)
print("TRAINING (20 epochs)")
print("="*60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # Higher LR to escape local min
    loss=pure_dice_loss,
    metrics=['accuracy']
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

monitor = LiverMonitorCallback(val_dataset)

history = model.fit(
    train_dataset,
    steps_per_epoch=len(train_indices),
    validation_data=val_dataset,
    validation_steps=len(val_indices),
    epochs=3,
    callbacks=[reduce_lr, monitor],
    verbose=1
)

# ============================================================
# SAVE MODEL
# ============================================================
model_path = os.path.join(output_dir, 'liver_segmentation_v5.keras')
model.save(model_path)
print(f"\nModel saved: {model_path}")

# ============================================================
# EVALUATE
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

print(f"\n{'='*60}")
print("SOFT DICE SCORES")
print(f"{'='*60}")
for name, score in soft_scores.items():
    print(f"{name}: {score:.4f}")
print(f"Average: {np.mean(list(soft_scores.values())):.4f}")

# HARD DICE
def compute_hard_dice(y_true, y_pred, class_label):
    smooth = 1e-5
    y_true_c = (y_true == class_label)
    y_pred_c = (y_pred == class_label)
    intersection = np.sum(y_true_c & y_pred_c)
    total = np.sum(y_true_c) + np.sum(y_pred_c)
    return (2 * intersection + smooth) / (total + smooth)

y_pred_class = np.argmax(y_pred, axis=-1)

print(f"\n{'='*60}")
print("HARD DICE SCORES")
print(f"{'='*60}")
print(f"Background: {compute_hard_dice(y_test_class, y_pred_class, 0):.4f}")
print(f"Liver: {compute_hard_dice(y_test_class, y_pred_class, 1):.4f}")
print(f"Tumor: {compute_hard_dice(y_test_class, y_pred_class, 2):.4f}")

# CLASSIFICATION REPORT
y_true_flat = y_test_class.flatten()
y_pred_flat = y_pred_class.flatten()

print(f"\n{'='*60}")
print("CLASSIFICATION REPORT")
print(f"{'='*60}")
print(classification_report(
    y_true_flat, y_pred_flat,
    target_names=['Background', 'Liver', 'Tumor'],
    labels=[0, 1, 2],
    zero_division=0
))

# DETAILED METRICS
print(f"\n{'='*60}")
print("DETAILED METRICS")
print(f"{'='*60}")

for class_idx, class_name in enumerate(['Background', 'Liver', 'Tumor']):
    y_true_binary = (y_true_flat == class_idx).astype(int)
    y_pred_binary = (y_pred_flat == class_idx).astype(int)
    
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    
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
    print(f"  Hard Dice: {compute_hard_dice(y_test_class, y_pred_class, class_idx):.4f}")

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)