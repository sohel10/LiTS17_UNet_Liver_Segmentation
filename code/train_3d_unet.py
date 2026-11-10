"""
3D U-Net Training Script — LITS17 Liver + Tumor Segmentation
==============================================================
Standalone retraining script — does NOT modify your main pipeline.
Uses ALL available NIfTI volumes to train a new 3D U-Net.

Architecture: Input (128, 128, 268, 1) → Output (128, 128, 268, 3)
Classes: 0=Background, 1=Liver, 2=Tumor

Usage:
    conda activate tf-gpu
    cd ~/LITS17/code
    python train_3d_unet.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# ─── Configuration ───────────────────────────────────────────────
LITS_ROOT = Path("/home/sohel/LITS17")
OUTPUT_DIR = LITS_ROOT / "models" / "retrained"

# All data directories to scan for volume/segmentation pairs
DATA_DIRS = [
    # (volume_dir, seg_dir, volume_prefix, seg_prefix)
    (LITS_ROOT / "data" / "volumes", LITS_ROOT / "data" / "segmentations", "volume-", "segmentation-"),
    (LITS_ROOT / "training_data", LITS_ROOT / "training_data", "volume-", "segmentation-"),
    (LITS_ROOT / "data" / "training_data", LITS_ROOT / "data" / "training_data", "volume-", "segmentation-"),
    (LITS_ROOT / "data" / "training_data_v2", LITS_ROOT / "data" / "training_data_v2", "volume-", "segmentation-"),
]

TARGET_SHAPE = (128, 128, 268)
NUM_CLASSES = 3
BATCH_SIZE = 1        # 3D volumes are large; 1 per batch with 16GB GPU
EPOCHS = 50
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15      # 15% for validation
EARLY_STOP_PATIENCE = 10

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Data Discovery ──────────────────────────────────────────────

def discover_pairs():
    """Find all volume/segmentation NIfTI pairs across all data dirs."""
    pairs = {}  # keyed by volume_id string to deduplicate

    for vol_dir, seg_dir, vol_prefix, seg_prefix in DATA_DIRS:
        if not vol_dir.exists():
            continue

        for f in sorted(vol_dir.glob("*.nii*")):
            name = f.name
            # Extract ID from filename like "volume-17.nii" or "volume-17.nii.gz"
            stem = f.stem
            if stem.endswith('.nii'):  # handle .nii.gz
                stem = stem[:-4]

            if not stem.startswith(vol_prefix.rstrip('-')):
                continue

            vid = stem.replace(vol_prefix.rstrip('-'), '').lstrip('-')

            # Find matching segmentation
            seg_patterns = [
                seg_dir / f"{seg_prefix}{vid}.nii",
                seg_dir / f"{seg_prefix}{vid}.nii.gz",
            ]
            seg_path = None
            for sp in seg_patterns:
                if sp.exists():
                    seg_path = sp
                    break

            if seg_path and vid not in pairs:
                pairs[vid] = (f, seg_path)

    return pairs


# ─── Preprocessing ───────────────────────────────────────────────

def load_and_preprocess(vol_path, seg_path):
    """Load NIfTI pair, preprocess to model input shape."""
    import nibabel as nib
    from scipy.ndimage import zoom

    # Load
    vol_data = nib.load(str(vol_path)).get_fdata().astype(np.float32)
    seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)

    # Clip CT window and normalize volume
    vol_data = np.clip(vol_data, -200, 250)
    vol_data = (vol_data - vol_data.min()) / (vol_data.max() - vol_data.min() + 1e-7)

    # Compute zoom factors
    zoom_factors = (
        TARGET_SHAPE[0] / vol_data.shape[0],
        TARGET_SHAPE[1] / vol_data.shape[1],
        TARGET_SHAPE[2] / vol_data.shape[2],
    )

    # Resize
    vol_resized = zoom(vol_data, zoom_factors, order=1)   # bilinear
    seg_resized = zoom(seg_data, zoom_factors, order=0)    # nearest-neighbor

    # Clamp segmentation labels to valid range
    seg_resized = np.clip(seg_resized, 0, NUM_CLASSES - 1).astype(np.int32)

    return vol_resized, seg_resized


# ─── Data Augmentation ───────────────────────────────────────────

def augment_volume(vol, seg):
    """Simple 3D augmentations: random flips along each axis."""
    for axis in range(3):
        if np.random.random() > 0.5:
            vol = np.flip(vol, axis=axis)
            seg = np.flip(seg, axis=axis)
    return vol.copy(), seg.copy()


# ─── Data Generator ──────────────────────────────────────────────

class VolumeGenerator(tf.keras.utils.Sequence):
    """Generator that loads, preprocesses, and augments volumes on the fly."""

    def __init__(self, pairs_list, batch_size=1, augment=False):
        self.pairs = pairs_list
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(self.pairs))

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []

        for i in batch_indices:
            vid, vol_path, seg_path = self.pairs[i]
            vol, seg = load_and_preprocess(vol_path, seg_path)

            if self.augment:
                vol, seg = augment_volume(vol, seg)

            # Add channel dim for input: (128, 128, 268) → (128, 128, 268, 1)
            batch_x.append(vol[..., np.newaxis])

            # One-hot encode segmentation: (128, 128, 268) → (128, 128, 268, 3)
            seg_onehot = tf.keras.utils.to_categorical(seg, num_classes=NUM_CLASSES)
            batch_y.append(seg_onehot)

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.indices)


# ─── Model Architecture ─────────────────────────────────────────

def build_3d_unet(input_shape=(128, 128, 268, 1), num_classes=3):
    """
    3D U-Net matching the original architecture:
    Encoder: Conv3D(32) → MaxPool → Conv3D(64) → MaxPool → Conv3D(128)
    Decoder: UpSample → Conv3D(64) → UpSample → Conv3D(32) → Conv3D(num_classes)

    Same structure as liver_segmentation_model.keras (554,243 params)
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    c1 = tf.keras.layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
    p1 = tf.keras.layers.MaxPooling3D(pool_size=2)(c1)

    c2 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(p1)
    p2 = tf.keras.layers.MaxPooling3D(pool_size=2)(c2)

    # Bottleneck
    c3 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same')(p2)

    # Decoder
    u1 = tf.keras.layers.UpSampling3D(size=2)(c3)
    c4 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same')(u1)

    u2 = tf.keras.layers.UpSampling3D(size=2)(c4)
    c5 = tf.keras.layers.Conv3D(32, 3, activation='relu', padding='same')(u2)

    # Output — softmax over 3 classes
    outputs = tf.keras.layers.Conv3D(num_classes, 1, activation='softmax', padding='same')(c5)

    model = tf.keras.Model(inputs, outputs)
    return model


# ─── Loss Functions ──────────────────────────────────────────────

def dice_loss(y_true, y_pred):
    """Dice loss — better for imbalanced segmentation."""
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    denom = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
    dice_per_class = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - tf.reduce_mean(dice_per_class)


def weighted_categorical_crossentropy(y_true, y_pred):
    """
    Class-weighted cross-entropy to handle extreme imbalance.
    Background is ~95% of voxels, liver ~4.5%, tumor ~0.5%.
    """
    # Weights: heavily upweight tumor
    weights = tf.constant([0.1, 1.0, 5.0], dtype=tf.float32)  # bg, liver, tumor

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    loss = -y_true * tf.math.log(y_pred) * weights
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


def combined_loss(y_true, y_pred):
    """Combined: weighted CE + Dice loss."""
    return weighted_categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# ─── Metrics ─────────────────────────────────────────────────────

def dice_coef_liver(y_true, y_pred):
    """Dice for liver class (index 1)."""
    smooth = 1e-6
    y_true_f = y_true[..., 1]
    y_pred_f = y_pred[..., 1]
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_tumor(y_true, y_pred):
    """Dice for tumor class (index 2)."""
    smooth = 1e-6
    y_true_f = y_true[..., 2]
    y_pred_f = y_pred[..., 2]
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


# ─── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  3D U-Net Retraining — LITS17")
    print("  Standalone experiment (does NOT touch main pipeline)")
    print("=" * 60)

    # Discover all data
    print("\n📂 Scanning for volume/segmentation pairs...")
    pairs = discover_pairs()

    if not pairs:
        print("❌ No data found! Check DATA_DIRS paths in the script.")
        return

    # Convert to list
    pairs_list = [(vid, vp, sp) for vid, (vp, sp) in sorted(pairs.items(), key=lambda x: int(x[0]))]
    print(f"  Found {len(pairs_list)} volume/segmentation pairs:")
    for vid, vp, sp in pairs_list:
        print(f"    Volume {vid}: {vp.parent.name}/{vp.name}")

    # Train/validation split
    np.random.seed(42)
    indices = np.arange(len(pairs_list))
    np.random.shuffle(indices)
    val_count = max(2, int(len(pairs_list) * VAL_SPLIT))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_pairs = [pairs_list[i] for i in train_indices]
    val_pairs = [pairs_list[i] for i in val_indices]

    print(f"\n  Training: {len(train_pairs)} volumes")
    print(f"  Validation: {len(val_pairs)} volumes")
    print(f"    Val IDs: {[p[0] for p in val_pairs]}")

    # Create generators
    train_gen = VolumeGenerator(train_pairs, batch_size=BATCH_SIZE, augment=True)
    val_gen = VolumeGenerator(val_pairs, batch_size=BATCH_SIZE, augment=False)

    # Build model
    print("\n🏗️  Building 3D U-Net...")
    model = build_3d_unet(
        input_shape=(*TARGET_SHAPE, 1),
        num_classes=NUM_CLASSES
    )
    print(f"  Parameters: {model.count_params():,}")

    # Compile with combined loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss,
        metrics=['accuracy', dice_coef_liver, dice_coef_tumor]
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / "best_retrained_3d_unet.keras"),
            monitor='val_dice_coef_liver',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coef_liver',
            mode='max',
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(OUTPUT_DIR / "training_log.csv")
        ),
    ]

    # Train
    print(f"\n🚀 Starting training — {EPOCHS} epochs, batch_size={BATCH_SIZE}")
    print(f"  Loss: Combined (weighted CE + Dice)")
    print(f"  Class weights: Background=0.1, Liver=1.0, Tumor=5.0")
    print(f"  Augmentation: random 3D flips")
    print(f"  Saving to: {OUTPUT_DIR}")
    start = datetime.now()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    elapsed = datetime.now() - start
    print(f"\n⏱️  Training completed in {elapsed}")

    # Save final model
    final_path = OUTPUT_DIR / "final_retrained_3d_unet.keras"
    model.save(str(final_path))
    print(f"💾 Final model: {final_path}")

    # Save training history
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_data['training_time'] = str(elapsed)
    hist_data['total_volumes'] = len(pairs_list)
    hist_data['train_volumes'] = len(train_pairs)
    hist_data['val_volumes'] = len(val_pairs)
    hist_data['val_ids'] = [p[0] for p in val_pairs]

    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(hist_data, f, indent=2)
    print(f"💾 History: {OUTPUT_DIR / 'training_history.json'}")

    # Quick summary
    best_liver = max(history.history.get('val_dice_coef_liver', [0]))
    best_tumor = max(history.history.get('val_dice_coef_tumor', [0]))
    print(f"\n📊 Best validation scores:")
    print(f"  Liver Dice:  {best_liver:.4f}")
    print(f"  Tumor Dice:  {best_tumor:.4f}")

    # Plot training curves
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('3D U-Net Retraining — Training Curves', fontsize=14, fontweight='bold')

        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train', color='#e76f51')
        axes[0, 0].plot(history.history['val_loss'], label='Val', color='#00b4d8')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Train', color='#e76f51')
        axes[0, 1].plot(history.history['val_accuracy'], label='Val', color='#00b4d8')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Liver Dice
        axes[1, 0].plot(history.history['dice_coef_liver'], label='Train', color='#e76f51')
        axes[1, 0].plot(history.history['val_dice_coef_liver'], label='Val', color='#00b4d8')
        axes[1, 0].set_title('Liver Dice')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Tumor Dice
        axes[1, 1].plot(history.history['dice_coef_tumor'], label='Train', color='#e76f51')
        axes[1, 1].plot(history.history['val_dice_coef_tumor'], label='Val', color='#00b4d8')
        axes[1, 1].set_title('Tumor Dice')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
        plt.close()
        print(f"📊 Curves: {OUTPUT_DIR / 'training_curves.png'}")
    except ImportError:
        print("  (matplotlib not available — skipping plots)")

    print("\n✅ Done! Run evaluate_segmentation.py with the new model to compare.")
    print(f"   Change MODEL_PATH to: {OUTPUT_DIR / 'best_retrained_3d_unet.keras'}")


if __name__ == "__main__":
    main()
