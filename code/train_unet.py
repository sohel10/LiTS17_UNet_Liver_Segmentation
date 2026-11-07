"""
Train 3D U-Net with Weighted Dice Loss (FIXED - Capped weights)
Run from terminal: python train_unet_fixed.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# CONFIG
preprocessed_dir = r"C:\Users\sohel\Downloads\Ai_health\preprocessed"
output_dir = r"C:\Users\sohel\Downloads\Ai_health\models"

os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("LOADING PREPROCESSED DATA")
print("="*60)

# Load data
X_train = np.load(os.path.join(preprocessed_dir, 'X_train.npy'))
y_train = np.load(os.path.join(preprocessed_dir, 'y_train.npy'))
X_val = np.load(os.path.join(preprocessed_dir, 'X_val.npy'))
y_val = np.load(os.path.join(preprocessed_dir, 'y_val.npy'))
X_test = np.load(os.path.join(preprocessed_dir, 'X_test.npy'))
y_test = np.load(os.path.join(preprocessed_dir, 'y_test.npy'))

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

# ============================================================================
# STEP 1: BUILD 3D U-NET MODEL
# ============================================================================
print("\n" + "="*60)
print("BUILDING 3D U-NET MODEL")
print("="*60)

inputs = keras.Input(shape=(128, 128, 268, 1))

# Encoder
c1 = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
p1 = layers.MaxPooling3D(2)(c1)

c2 = layers.Conv3D(64, 3, activation='relu', padding='same')(p1)
p2 = layers.MaxPooling3D(2)(c2)

# Bottleneck
c3 = layers.Conv3D(128, 3, activation='relu', padding='same')(p2)

# Decoder
u1 = layers.UpSampling3D(2)(c3)
c4 = layers.Conv3D(64, 3, activation='relu', padding='same')(u1)

u2 = layers.UpSampling3D(2)(c4)
c5 = layers.Conv3D(32, 3, activation='relu', padding='same')(u2)

# Output: 3 classes
outputs = layers.Conv3D(3, 1, activation='softmax')(c5)

model = keras.Model(inputs, outputs)
print(f"Model params: {model.count_params():,}")

# ============================================================================
# STEP 2: WEIGHTED DICE LOSS (FIXED - Capped weights)
# ============================================================================
print("\n" + "="*60)
print("DEFINING WEIGHTED DICE LOSS (Capped Weights)")
print("="*60)

def weighted_dice_loss(y_true, y_pred):
    """
    Weighted Dice Loss with CAPPED weights
    Prevents numerical instability from extreme values
    """
    smooth = 1.0
    
    # CAPPED WEIGHTS (not 365!)
    class_weights = tf.constant([1.0, 10.0, 50.0])
    
    total_loss = 0.0
    
    for i in range(3):  # 3 classes
        y_true_class = y_true[..., i]
        y_pred_class = y_pred[..., i]
        
        intersection = tf.reduce_sum(y_true_class * y_pred_class)
        union = tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        # Weighted loss
        class_loss = (1.0 - dice) * class_weights[i]
        
        total_loss += class_loss
    
    return total_loss / 3.0

print("Loss function: Weighted Dice Loss")
print("Class weights (CAPPED): [1.0, 10.0, 50.0]")

# ============================================================================
# STEP 3: COMPILE MODEL
# ============================================================================
print("\n" + "="*60)
print("COMPILING MODEL")
print("="*60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=weighted_dice_loss,
    metrics=['accuracy']
)

print("Optimizer: Adam (lr=0.001)")
print("Loss: Weighted Dice Loss (capped weights)")
print("Metrics: accuracy")

# ============================================================================
# STEP 4: TRAIN MODEL
# ============================================================================
print("\n" + "="*60)
print("TRAINING MODEL (15 epochs)")
print("="*60)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=1,
    validation_data=(X_val, y_val),
    callbacks=[reduce_lr],
    verbose=1
)

# ============================================================================
# STEP 5: SAVE MODEL
# ============================================================================
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

model_path = os.path.join(output_dir, 'liver_segmentation_weighted_dice.keras')
model.save(model_path)
print(f"✓ Model saved: {model_path}")

# ============================================================================
# STEP 6: EVALUATE ON TEST SET
# ============================================================================
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=-1)
y_test_class = np.argmax(y_test, axis=-1)

# Dice score function
def dice_score(y_true, y_pred, class_label):
    smooth = 1e-5
    y_true_class = (y_true == class_label)
    y_pred_class = (y_pred == class_label)
    
    intersection = np.sum(y_true_class & y_pred_class)
    total = np.sum(y_true_class) + np.sum(y_pred_class)
    
    dice = (2 * intersection + smooth) / (total + smooth)
    return dice

# Calculate Dice for each class
dice_bg = dice_score(y_test_class, y_pred_class, 0)
dice_liver = dice_score(y_test_class, y_pred_class, 1)
dice_tumor = dice_score(y_test_class, y_pred_class, 2)

print(f"\n{'='*60}")
print("DICE SCORES")
print(f"{'='*60}")
print(f"Background Dice: {dice_bg:.4f}")
print(f"Liver Dice: {dice_liver:.4f}")
print(f"Tumor Dice: {dice_tumor:.4f}")
print(f"Average Dice: {(dice_bg + dice_liver + dice_tumor) / 3:.4f}")

# Classification report
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

# Per-class detailed metrics
print(f"\n{'='*60}")
print("DETAILED METRICS")
print(f"{'='*60}")

for class_idx, class_name in enumerate(['Background', 'Liver', 'Tumor']):
    y_true_binary = (y_true_flat == class_idx).astype(int)
    y_pred_binary = (y_pred_flat == class_idx).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{class_name}:")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Dice: {dice_score(y_test_class, y_pred_class, class_idx):.4f}")

print("\n" + "="*60)
print("✓ TRAINING AND EVALUATION COMPLETE!")
print("="*60)