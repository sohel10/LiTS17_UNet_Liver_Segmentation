import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from scipy.ndimage import zoom
from sklearn.metrics import classification_report, confusion_matrix

# CONFIG
work_dir = r"C:\Users\sohel\Downloads\Ai_health\LITS17"
output_dir = r"C:\Users\sohel\Downloads\Ai_health\models"
os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("3D LIVER SEGMENTATION - V6 WEIGHTED (CLEAN)")
print("Loss: Weighted Foreground-Only Generalized Dice")
print("Liver weight: 1.0 | Tumor weight: 125.0")
print("Epochs: 30")
print("="*60)

# ============================================================
# DATA GENERATOR
# ============================================================
class Medical3DDataGenerator:
    def __init__(self, volume_indices, work_dir):
        self.volume_indices = volume_indices
        self.work_dir = work_dir

    def _process_single_volume(self, index):
        ct = nib.load(os.path.join(self.work_dir, f'volume-{index}.nii')).get_fdata(dtype=np.float32)
        seg = nib.load(os.path.join(self.work_dir, f'segmentation-{index}.nii')).get_fdata(dtype=np.float32)

        # Pad/crop to median depth 541
        if ct.shape[2] < 541:
            ct = np.pad(ct, ((0, 0), (0, 0), (0, 541 - ct.shape[2])))
            seg = np.pad(seg, ((0, 0), (0, 0), (0, 541 - seg.shape[2])))
        else:
            ct = ct[:, :, :541]
            seg = seg[:, :, :541]

        # Normalize CT
        ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-5)

        # Resize: (512, 512, 541) -> (128, 128, 256)
        ct = zoom(ct, (0.25, 0.25, 0.473), order=1)
        seg = zoom(seg, (0.25, 0.25, 0.473), order=0)
        
        ct = ct[:, :, :256]
        seg = seg[:, :, :256]

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
all_indices = list(range(20))
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
    tf.TensorSpec(shape=(128, 128, 256, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(128, 128, 256, 3), dtype=tf.float32)
)

train_gen = Medical3DDataGenerator(train_indices, work_dir)
val_gen = Medical3DDataGenerator(val_indices, work_dir)
test_gen = Medical3DDataGenerator(test_indices, work_dir)

train_dataset = tf.data.Dataset.from_generator(
    train_gen.get_generator(), output_signature=output_signature
).batch(1).repeat().prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    val_gen.get_generator(), output_signature=output_signature
).batch(1).repeat().prefetch(tf.data.AUTOTUNE)

# ============================================================
# BUILD 3D U-NET
# ============================================================
print("\nBuilding 3D U-Net...")

inputs = keras.Input(shape=(128, 128, 256, 1))

# Encoder
c1 = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
c1 = layers.BatchNormalization()(c1)
c1 = layers.Conv3D(32, 3, activation='relu', padding='same')(c1)
p1 = layers.MaxPooling3D(2)(c1)

c2 = layers.Conv3D(64, 3, activation='relu', padding='same')(p1)
c2 = layers.BatchNormalization()(c2)
c2 = layers.Conv3D(64, 3, activation='relu', padding='same')(c2)
p2 = layers.MaxPooling3D(2)(c2)

# Bottleneck
c3 = layers.Conv3D(128, 3, activation='relu', padding='same')(p2)
c3 = layers.BatchNormalization()(c3)
c3 = layers.Conv3D(128, 3, activation='relu', padding='same')(c3)

# Decoder with skip connections
u1 = layers.UpSampling3D(2)(c3)
u1 = layers.Concatenate()([u1, c2])
c4 = layers.Conv3D(64, 3, activation='relu', padding='same')(u1)
c4 = layers.Conv3D(64, 3, activation='relu', padding='same')(c4)

u2 = layers.UpSampling3D(2)(c4)
u2 = layers.Concatenate()([u2, c1])
c5 = layers.Conv3D(32, 3, activation='relu', padding='same')(u2)
c5 = layers.Conv3D(32, 3, activation='relu', padding='same')(c5)

# Output: sigmoid (not softmax!)
outputs = layers.Conv3D(3, 1, activation='sigmoid')(c5)

model = keras.Model(inputs, outputs)
print(f"Model parameters: {model.count_params():,}")

# ============================================================
# WEIGHTED GENERALIZED DICE LOSS (FOREGROUND-ONLY)
# ============================================================
def weighted_generalized_dice_loss(y_true, y_pred, liver_weight=1.0, tumor_weight=125.0):
    """
    Weighted Generalized Dice Loss - FIXED:
    1. Compute loss ONLY on foreground (channels 1-2)
    2. Completely ignore channel 0 (background)
    3. Weight tumor 125x heavier than liver
    """
    smooth = 1e-5
    
    # Extract ONLY foreground channels (1=liver, 2=tumor)
    # IGNORE channel 0 (background) completely
    y_true_fg = y_true[..., 1:]  # Shape: (batch, h, w, d, 2)
    y_pred_fg = y_pred[..., 1:]  # IGNORE BACKGROUND PREDICTIONS
    
    # Separate liver (channel 0 of foreground) and tumor (channel 1 of foreground)
    y_true_liver = y_true_fg[..., 0]
    y_pred_liver = y_pred_fg[..., 0]
    
    y_true_tumor = y_true_fg[..., 1]
    y_pred_tumor = y_pred_fg[..., 1]
    
    # Compute Dice for each class (ONLY on foreground)
    intersection_liver = 2.0 * tf.reduce_sum(y_true_liver * y_pred_liver)
    union_liver = tf.reduce_sum(y_true_liver) + tf.reduce_sum(y_pred_liver)
    dice_liver = (intersection_liver + smooth) / (union_liver + smooth)
    
    intersection_tumor = 2.0 * tf.reduce_sum(y_true_tumor * y_pred_tumor)
    union_tumor = tf.reduce_sum(y_true_tumor) + tf.reduce_sum(y_pred_tumor)
    dice_tumor = (intersection_tumor + smooth) / (union_tumor + smooth)
    
    # Weighted loss: tumor gets 125x weight
    loss = (liver_weight * (1.0 - dice_liver)) + (tumor_weight * (1.0 - dice_tumor))
    
    return loss

print("\nCompiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=weighted_generalized_dice_loss,
    metrics=['accuracy']
)

# ============================================================
# TRAIN (30 EPOCHS)
# ============================================================
print("\nTraining (30 epochs)...\n")

history = model.fit(
    train_dataset,
    steps_per_epoch=len(train_indices),
    validation_data=val_dataset,
    validation_steps=len(val_indices),
    epochs=5,  # Start with 10 for quick test
    verbose=1
)

# ============================================================
# SAVE MODEL
# ============================================================
model_path = os.path.join(output_dir, 'liver_segmentation_v6_weighted.keras')
model.save(model_path)
print(f"\nModel saved: {model_path}")

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

X_test_list, y_test_list = [], []
for i in test_indices:
    ct, seg = test_gen._process_single_volume(i)
    X_test_list.append(ct)
    y_test_list.append(seg)

X_test = np.array(X_test_list)
y_test = np.array(y_test_list)

y_test_class = np.argmax(y_test, axis=-1)
print(f"\nTest set info:")
print(f"  Liver voxels: {np.sum(y_test_class == 1):,}")
print(f"  Tumor voxels: {np.sum(y_test_class == 2):,}")

# Predict
y_pred = model.predict(X_test)

# ============================================================
# EVALUATE
# ============================================================
threshold = 0.5
y_pred_binary = (y_pred > threshold).astype(np.uint8)
y_pred_class = np.argmax(y_pred_binary, axis=-1)

# Soft Dice
def compute_soft_dice(y_true, y_pred):
    smooth = 1e-5
    scores = {}
    names = ['Background', 'Liver', 'Tumor']
    for ch in range(3):
        true_ch = y_true[..., ch]
        pred_ch = y_pred[..., ch]
        intersection = np.sum(true_ch * pred_ch)
        total = np.sum(true_ch) + np.sum(pred_ch)
        dice = (2.0 * intersection + smooth) / (total + smooth)
        scores[names[ch]] = dice
    return scores

soft_scores = compute_soft_dice(y_test, y_pred)

print("\nSOFT DICE SCORES:")
for name, score in soft_scores.items():
    print(f"  {name}: {score:.4f}")

# Hard Dice
def compute_hard_dice(y_true_binary, y_pred_binary, channel):
    smooth = 1e-5
    true_ch = y_true_binary[..., channel].astype(np.float32)
    pred_ch = y_pred_binary[..., channel].astype(np.float32)
    intersection = np.sum(true_ch * pred_ch)
    total = np.sum(true_ch) + np.sum(pred_ch)
    return (2.0 * intersection + smooth) / (total + smooth)

print("\nHARD DICE SCORES:")
for ch, name in enumerate(['Background', 'Liver', 'Tumor']):
    dice = compute_hard_dice(y_test, y_pred_binary, ch)
    print(f"  {name}: {dice:.4f}")

# Classification report
y_test_flat = y_test_class.flatten()
y_pred_flat = y_pred_class.flatten()

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(
    y_test_flat, y_pred_flat,
    target_names=['Background', 'Liver', 'Tumor'],
    labels=[0, 1, 2],
    zero_division=0
))

# Detailed metrics
print("\n" + "="*60)
print("DETAILED METRICS PER CLASS")
print("="*60)

for class_idx, class_name in enumerate(['Background', 'Liver', 'Tumor']):
    y_true_binary = (y_test_flat == class_idx).astype(int)
    y_pred_binary_class = (y_pred_flat == class_idx).astype(int)
    
    cm = confusion_matrix(y_true_binary, y_pred_binary_class, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    print(f"\n{class_name}:")
    print(f"  TP={tp:,}, FP={fp:,}, TN={tn:,}, FN={fn:,}")
    print(f"  Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    print(f"  Precision: {precision:.4f}, Recall: {sensitivity:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Soft Dice: {soft_scores[class_name]:.4f}")
    print(f"  Hard Dice: {compute_hard_dice(y_test, y_pred_binary, class_idx):.4f}")

print("\n" + "="*60)
print("✓ COMPLETE! Ready for interview! 🚀")
print("="*60)