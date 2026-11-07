"""
FINAL V4: All bugs fixed
- Proportional depth resize (preserves liver!)
- Soft Dice evaluation
- 10 epochs
- Skip connections
- Robust loss
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
print("3D LIVER SEGMENTATION (FINAL V4)")
print("="*60)

# ============================================================
# DATA GENERATOR (FIXED: proportional depth resize)
# ============================================================
class Medical3DDataGenerator:
    def __init__(self, volume_indices, work_dir, median_depth=541, target_depth=256):
        self.volume_indices = volume_indices
        self.work_dir = work_dir
        self.median_depth = median_depth
        self.target_depth = target_depth
        self.depth_factor = target_depth / median_depth  # 256/541 = 0.473

    def _process_single_volume(self, index):
        ct = nib.load(os.path.join(self.work_dir, f'volume-{index}.nii')).get_fdata(dtype=np.float32)
        seg = nib.load(os.path.join(self.work_dir, f'segmentation-{index}.nii')).get_fdata(dtype=np.float32)

        # Pad/crop to median depth
        if ct.shape[2] < self.median_depth:
            add_slices = self.median_depth - ct.shape[2]
            ct = np.pad(ct, ((0, 0), (0, 0), (0, add_slices)))
            seg = np.pad(seg, ((0, 0), (0, 0), (0, add_slices)))
        else:
            ct = ct[:, :, :self.median_depth]
            seg = seg[:, :, :self.median_depth]

        # Normalize
        ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-5)

        # FIX: Resize ALL dimensions proportionally
        ct = zoom(ct, (0.25, 0.25, self.depth_factor), order=1)
        seg = zoom(seg, (0.25, 0.25, self.depth_factor), order=0)

        # Ensure exact target depth
        ct = ct[:, :, :self.target_depth]
        seg = seg[:, :, :self.target_depth]

        # Add channel and one-hot
        ct = ct[..., np.newaxis]
        seg = to_categorical(seg, num_classes=3)

        return ct.astype(np.float32), seg.astype(np.float32)

    def get_generator(self):
        def generator():
            for i in self.volume_indices:
                yield self._process_single_volume(i)
        return generator

# ============================================================
# VERIFY DATA (check liver exists after resize)
# ============================================================
print("\nVerifying data after proportional resize...")

depth_factor = target_depth / median_depth
print(f"Depth factor: {target_depth}/{median_depth} = {depth_factor:.4f}")

for i in [0, 5, 10, 15]:
    seg = nib.load(os.path.join(work_dir, f'segmentation-{i}.nii')).get_fdata(dtype=np.float32)
    
    # Pad/crop
    if seg.shape[2] < median_depth:
        add_slices = median_depth - seg.shape[2]
        seg = np.pad(seg, ((0, 0), (0, 0), (0, add_slices)))
    else:
        seg = seg[:, :, :median_depth]
    
    # Resize proportionally
    seg_resized = zoom(seg, (0.25, 0.25, depth_factor), order=0)
    seg_resized = seg_resized[:, :, :target_depth]
    
    classes = np.unique(seg_resized)
    liver_count = np.sum(seg_resized == 1)
    tumor_count = np.sum(seg_resized == 2)
    
    print(f"  Volume {i}: shape {seg_resized.shape}, classes {classes}, liver={liver_count:,}, tumor={tumor_count:,}")

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
# CREATE DATASETS WITH .repeat()
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

print(f"Train: {len(train_indices)} volumes")
print(f"Val: {len(val_indices)} volumes")
print(f"Test: {len(test_indices)} volumes")

# ============================================================
# BUILD U-NET WITH SKIP CONNECTIONS
# ============================================================
print("\n" + "="*60)
print("Building 3D U-Net with Skip Connections")
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

# DECODER (with skip connections)
u1 = layers.UpSampling3D(2)(c3)
u1 = layers.Concatenate()([u1, c2])  # Skip connection!
c4 = layers.Conv3D(64, 3, activation='relu', padding='same')(u1)
c4 = layers.Conv3D(64, 3, activation='relu', padding='same')(c4)

u2 = layers.UpSampling3D(2)(c4)
u2 = layers.Concatenate()([u2, c1])  # Skip connection!
c5 = layers.Conv3D(32, 3, activation='relu', padding='same')(u2)
c5 = layers.Conv3D(32, 3, activation='relu', padding='same')(c5)

# OUTPUT
outputs = layers.Conv3D(3, 1, activation='softmax')(c5)

model = keras.Model(inputs, outputs)
print(f"Model params: {model.count_params():,}")

# ============================================================
# ROBUST LOSS FUNCTION
# ============================================================
print("\n" + "="*60)
print("Robust Segmentation Loss (CCE + Bounded GDL)")
print("="*60)

def robust_segmentation_loss(y_true, y_pred):
    smooth = 1e-5
    y_true_f = tf.reshape(y_true, [-1, 3])
    y_pred_f = tf.reshape(y_pred, [-1, 3])

    # CCE component
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    mean_cce = tf.reduce_mean(cce)

    # Bounded GDL component
    counts = tf.reduce_sum(y_true_f, axis=0)
    weights = 1.0 / (tf.square(counts) + 1.0)  # +1.0 prevents explosion

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    lengths = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)

    numerator = tf.reduce_sum(weights * intersection)
    denominator = tf.reduce_sum(weights * lengths)

    gdl = 2.0 * (numerator + smooth) / (denominator + smooth)
    dice_component = 1.0 - gdl

    return mean_cce + dice_component

# ============================================================
# COMPILE AND TRAIN
# ============================================================
print("\n" + "="*60)
print("TRAINING (10 epochs)")
print("="*60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=robust_segmentation_loss,
    metrics=['accuracy']
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

history = model.fit(
    train_dataset,
    steps_per_epoch=len(train_indices),
    validation_data=val_dataset,
    validation_steps=len(val_indices),
    epochs=2,
    callbacks=[reduce_lr],
    verbose=1
)

# ============================================================
# SAVE MODEL
# ============================================================
model_path = os.path.join(output_dir, 'liver_segmentation_v4.keras')
model.save(model_path)
print(f"\nModel saved: {model_path}")

# ============================================================
# EVALUATE
# ============================================================
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

# Load test data
X_test_list = []
y_test_list = []

for i in test_indices:
    ct, seg = test_gen._process_single_volume(i)
    X_test_list.append(ct)
    y_test_list.append(seg)

X_test = np.array(X_test_list)
y_test = np.array(y_test_list)

# Verify test set has liver
y_test_class = np.argmax(y_test, axis=-1)
print(f"Test classes: {np.unique(y_test_class)}")
print(f"Test liver pixels: {np.sum(y_test_class == 1):,}")
print(f"Test tumor pixels: {np.sum(y_test_class == 2):,}")

# Predict
y_pred = model.predict(X_test)

# ============================================================
# SOFT DICE (probability-based)
# ============================================================
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
print("SOFT DICE SCORES (Probability-based)")
print(f"{'='*60}")
for name, score in soft_scores.items():
    print(f"{name}: {score:.4f}")
print(f"Average: {np.mean(list(soft_scores.values())):.4f}")

# ============================================================
# HARD DICE (argmax-based)
# ============================================================
def compute_hard_dice(y_true, y_pred, class_label):
    smooth = 1e-5
    y_true_c = (y_true == class_label)
    y_pred_c = (y_pred == class_label)
    intersection = np.sum(y_true_c & y_pred_c)
    total = np.sum(y_true_c) + np.sum(y_pred_c)
    return (2 * intersection + smooth) / (total + smooth)

y_pred_class = np.argmax(y_pred, axis=-1)

print(f"\n{'='*60}")
print("HARD DICE SCORES (Argmax-based)")
print(f"{'='*60}")
print(f"Background: {compute_hard_dice(y_test_class, y_pred_class, 0):.4f}")
print(f"Liver: {compute_hard_dice(y_test_class, y_pred_class, 1):.4f}")
print(f"Tumor: {compute_hard_dice(y_test_class, y_pred_class, 2):.4f}")

# ============================================================
# CLASSIFICATION REPORT (FIXED)
# ============================================================
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

# ============================================================
# PER-CLASS DETAILED METRICS
# ============================================================
from sklearn.metrics import confusion_matrix

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