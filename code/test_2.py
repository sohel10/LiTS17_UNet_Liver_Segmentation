"""
FINAL FIXED: 3D Liver Segmentation
Fixes: Power-of-2, Skip Connections, GDL Loss, Data Generator
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
target_depth = 256  # Power of 2!

print("="*60)
print("3D LIVER SEGMENTATION (FINAL FIXED)")
print("="*60)

# ============================================================
# STEP 1: DATA GENERATOR (No RAM crash)
# ============================================================
print("\nSTEP 1: Setting up Data Generator...")

class Medical3DDataGenerator:
    def __init__(self, volume_indices, work_dir, median_depth=541, target_depth=256):
        self.volume_indices = volume_indices
        self.work_dir = work_dir
        self.median_depth = median_depth
        self.target_depth = target_depth

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

        # Resize (0.25, 0.25 for width/height, keep depth)
        ct = zoom(ct, (0.25, 0.25, 1.0), order=1)
        seg = zoom(seg, (0.25, 0.25, 1.0), order=0)

        # Crop to power-of-2 depth
        ct = ct[:, :, :self.target_depth]
        seg = seg[:, :, :self.target_depth]

        # Add channel and one-hot encode
        ct = ct[..., np.newaxis]
        seg = to_categorical(seg, num_classes=3)

        return ct.astype(np.float32), seg.astype(np.float32)

    def get_generator(self):
        def generator():
            for i in self.volume_indices:
                yield self._process_single_volume(i)
        return generator

# Split indices: 12 train, 4 val, 4 test
all_indices = list(range(num_volumes))
np.random.seed(42)
np.random.shuffle(all_indices)

train_indices = all_indices[:12]
val_indices = all_indices[12:16]
test_indices = all_indices[16:20]

print(f"Train: {len(train_indices)} volumes")
print(f"Val: {len(val_indices)} volumes")
print(f"Test: {len(test_indices)} volumes")

# Create TF datasets
output_signature = (
    tf.TensorSpec(shape=(128, 128, target_depth, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(128, 128, target_depth, 3), dtype=tf.float32)
)

train_gen = Medical3DDataGenerator(train_indices, work_dir, median_depth, target_depth)
val_gen = Medical3DDataGenerator(val_indices, work_dir, median_depth, target_depth)
test_gen = Medical3DDataGenerator(test_indices, work_dir, median_depth, target_depth)

train_dataset = tf.data.Dataset.from_generator(
    train_gen.get_generator(), output_signature=output_signature
).batch(1).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    val_gen.get_generator(), output_signature=output_signature
).batch(1).prefetch(tf.data.AUTOTUNE)

print("Data generators ready!")

# ============================================================
# STEP 2: BUILD PROPER U-NET (Skip Connections + Power-of-2)
# ============================================================
print("\n" + "="*60)
print("STEP 2: Building Proper 3D U-Net")
print("="*60)

inputs = keras.Input(shape=(128, 128, target_depth, 1))

# --- ENCODER ---
c1 = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
c1 = layers.Conv3D(32, 3, activation='relu', padding='same')(c1)
p1 = layers.MaxPooling3D(2)(c1)  # (64, 64, 128)

c2 = layers.Conv3D(64, 3, activation='relu', padding='same')(p1)
c2 = layers.Conv3D(64, 3, activation='relu', padding='same')(c2)
p2 = layers.MaxPooling3D(2)(c2)  # (32, 32, 64)

# --- BOTTLENECK ---
c3 = layers.Conv3D(128, 3, activation='relu', padding='same')(p2)
c3 = layers.Conv3D(128, 3, activation='relu', padding='same')(c3)

# --- DECODER (With Skip Connections!) ---
u1 = layers.UpSampling3D(2)(c3)  # (64, 64, 128)
u1 = layers.Concatenate()([u1, c2])  # SKIP CONNECTION!
c4 = layers.Conv3D(64, 3, activation='relu', padding='same')(u1)
c4 = layers.Conv3D(64, 3, activation='relu', padding='same')(c4)

u2 = layers.UpSampling3D(2)(c4)  # (128, 128, 256)
u2 = layers.Concatenate()([u2, c1])  # SKIP CONNECTION!
c5 = layers.Conv3D(32, 3, activation='relu', padding='same')(u2)
c5 = layers.Conv3D(32, 3, activation='relu', padding='same')(c5)

# Output
outputs = layers.Conv3D(3, 1, activation='softmax')(c5)

model = keras.Model(inputs, outputs)
print(f"Model params: {model.count_params():,}")

# ============================================================
# STEP 3: GENERALIZED DICE LOSS (Stable)
# ============================================================
print("\n" + "="*60)
print("STEP 3: Generalized Dice Loss")
print("="*60)

def generalized_dice_loss(y_true, y_pred):
    smooth = 1e-5
    y_true_f = tf.reshape(y_true, [-1, 3])
    y_pred_f = tf.reshape(y_pred, [-1, 3])

    counts = tf.reduce_sum(y_true_f, axis=0)
    weights = 1.0 / (tf.square(counts) + smooth)

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    lengths = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)

    numerator = tf.reduce_sum(weights * intersection)
    denominator = tf.reduce_sum(weights * lengths)

    gdl = 2.0 * (numerator + smooth) / (denominator + smooth)
    return 1.0 - gdl

print("Generalized Dice Loss defined")

# ============================================================
# STEP 4: COMPILE AND TRAIN
# ============================================================
print("\n" + "="*60)
print("STEP 4: Compiling and Training")
print("="*60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=generalized_dice_loss,
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
    epochs=2,
    validation_data=val_dataset,
    callbacks=[reduce_lr],
    verbose=1
)

# ============================================================
# STEP 5: SAVE MODEL
# ============================================================
print("\n" + "="*60)
print("STEP 5: Saving Model")
print("="*60)

model_path = os.path.join(output_dir, 'liver_segmentation_final_fixed.keras')
model.save(model_path)
print(f"Model saved: {model_path}")

# ============================================================
# STEP 6: EVALUATE
# ============================================================
print("\n" + "="*60)
print("STEP 6: Evaluating on Test Set")
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

y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=-1)
y_test_class = np.argmax(y_test, axis=-1)

def dice_score(y_true, y_pred, class_label):
    smooth = 1e-5
    y_true_c = (y_true == class_label)
    y_pred_c = (y_pred == class_label)
    intersection = np.sum(y_true_c & y_pred_c)
    total = np.sum(y_true_c) + np.sum(y_pred_c)
    return (2 * intersection + smooth) / (total + smooth)

print(f"\n{'='*60}")
print("DICE SCORES")
print(f"{'='*60}")
print(f"Background Dice: {dice_score(y_test_class, y_pred_class, 0):.4f}")
print(f"Liver Dice: {dice_score(y_test_class, y_pred_class, 1):.4f}")
print(f"Tumor Dice: {dice_score(y_test_class, y_pred_class, 2):.4f}")

y_true_flat = y_test_class.flatten()
y_pred_flat = y_pred_class.flatten()

print(f"\n{'='*60}")
print("CLASSIFICATION REPORT")
print(f"{'='*60}")
print(classification_report(
    y_true_flat, y_pred_flat,
    target_names=['Background', 'Liver', 'Tumor'],
    zero_division=0
))

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)