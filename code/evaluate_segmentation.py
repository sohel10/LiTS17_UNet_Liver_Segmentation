"""
Segmentation Evaluation Script — LITS17 3D U-Net Liver Segmentation
====================================================================
3D U-Net: Input (128, 128, 268, 1) → Output (128, 128, 268, 3)
3 classes: 0=Background, 1=Liver, 2=Tumor

Usage:
    conda activate tf-gpu
    cd ~/LITS17/code
    python evaluate_segmentation.py
"""

import os
import json
import csv
import numpy as np
import tensorflow as tf
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────
LITS_ROOT = Path("/home/sohel/LITS17")
MODEL_PATH = LITS_ROOT / "models" / "retrained" / "best_retrained_3d_unet.keras"
VOLUME_DIR = LITS_ROOT / "data" / "volumes"
SEG_DIR = LITS_ROOT / "data" / "segmentations"
OUTPUT_DIR = LITS_ROOT / "code" / "evaluation_output"

# Model expects this shape
TARGET_SHAPE = (128, 128, 268)  # (H, W, D)
NUM_CLASSES = 3  # background, liver, tumor
THRESHOLD = 0.5

# Test volumes — adjust to match your train/test split
TEST_VOLUMES = [17, 18, 19]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── NIfTI Loading ───────────────────────────────────────────────

def load_nifti(filepath):
    """Load NIfTI file → numpy array."""
    import nibabel as nib
    img = nib.load(str(filepath))
    return img.get_fdata()


def preprocess_volume(vol_data):
    """
    Preprocess a full 3D volume for the model.
    vol_data shape from nibabel: (H, W, num_slices)
    Model expects: (128, 128, 268)
    """
    from scipy.ndimage import zoom

    # Clip to CT liver window
    vol = np.clip(vol_data, -200, 250).astype(np.float32)

    # Normalize to [0, 1]
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-7)

    # Resize to target shape
    zoom_factors = (
        TARGET_SHAPE[0] / vol.shape[0],
        TARGET_SHAPE[1] / vol.shape[1],
        TARGET_SHAPE[2] / vol.shape[2]
    )
    vol = zoom(vol, zoom_factors, order=1)  # bilinear

    return vol


def preprocess_seg(seg_data):
    """
    Preprocess segmentation mask to match model input shape.
    Keeps integer labels (0=bg, 1=liver, 2=tumor).
    """
    from scipy.ndimage import zoom

    zoom_factors = (
        TARGET_SHAPE[0] / seg_data.shape[0],
        TARGET_SHAPE[1] / seg_data.shape[1],
        TARGET_SHAPE[2] / seg_data.shape[2]
    )
    seg = zoom(seg_data, zoom_factors, order=0)  # nearest-neighbor
    return seg.astype(np.int32)


# ─── Metric Functions ────────────────────────────────────────────

def dice_score(y_true, y_pred):
    smooth = 1e-7
    intersection = np.sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def iou_score(y_true, y_pred):
    smooth = 1e-7
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_score(y_true, y_pred):
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    return float(tp / (tp + fp + 1e-7))


def recall_score(y_true, y_pred):
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    return float(tp / (tp + fn + 1e-7))


def f1_from(prec, rec):
    return 2 * (prec * rec) / (prec + rec + 1e-7)


def confusion_values(y_true, y_pred):
    tp = int(np.sum(y_true * y_pred))
    fp = int(np.sum((1 - y_true) * y_pred))
    fn = int(np.sum(y_true * (1 - y_pred)))
    tn = int(np.sum((1 - y_true) * (1 - y_pred)))
    return tp, fp, fn, tn


# ─── Main Evaluation ──────────────────────────────────────────────

def evaluate():
    print("=" * 60)
    print("  LITS17 3D U-Net Segmentation Evaluation")
    print("  Model input:  (128, 128, 268, 1)")
    print("  Model output: (128, 128, 268, 3)")
    print("  Classes: Background | Liver | Tumor")
    print("=" * 60)

    # Load model
    print(f"\n📦 Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    print(f"  ✅ Model loaded! Params: {model.count_params():,}")

    # Find available test volumes
    available = []
    for vid in TEST_VOLUMES:
        vf = VOLUME_DIR / f"volume-{vid}.nii"
        sf = SEG_DIR / f"segmentation-{vid}.nii"
        if vf.exists() and sf.exists():
            available.append(vid)
        else:
            print(f"  ⚠️  Volume {vid}: vol={vf.exists()}, seg={sf.exists()} — skipping")

    if not available:
        print("\n❌ No test volumes found! Available:")
        for f in sorted(VOLUME_DIR.glob("volume-*.nii")):
            vid = f.stem.split("-")[1]
            has_seg = (SEG_DIR / f"segmentation-{vid}.nii").exists()
            print(f"    volume-{vid}  seg: {'✅' if has_seg else '❌'}")
        print("\n  Edit TEST_VOLUMES in the script.")
        return

    print(f"\n  Evaluating volumes: {available}")

    # Per-class storage
    # class_names[0] = Background, [1] = Liver, [2] = Tumor
    class_names = ['Background', 'Liver', 'Tumor']
    volume_results = []

    # Global accumulators per class
    global_tp = [0, 0, 0]
    global_fp = [0, 0, 0]
    global_fn = [0, 0, 0]
    global_tn = [0, 0, 0]
    all_slice_dice_liver = []  # per-slice Dice for liver class

    for vol_id in available:
        print(f"\n🔬 Volume {vol_id}...")

        # Load
        vol_raw = load_nifti(VOLUME_DIR / f"volume-{vol_id}.nii")
        seg_raw = load_nifti(SEG_DIR / f"segmentation-{vol_id}.nii")
        print(f"  Raw shape: {vol_raw.shape}")

        # Preprocess
        print("  Preprocessing...")
        vol = preprocess_volume(vol_raw)
        seg = preprocess_seg(seg_raw)
        print(f"  Resized to: {vol.shape}")

        # Predict — input shape (1, 128, 128, 268, 1)
        print("  Running inference...")
        vol_input = vol[np.newaxis, ..., np.newaxis]
        pred_raw = model.predict(vol_input, verbose=0)[0]  # (128, 128, 268, 3)

        # Convert to class predictions
        pred_classes = np.argmax(pred_raw, axis=-1)  # (128, 128, 268)

        # Per-class metrics for this volume
        vol_metrics = {}
        for c in range(NUM_CLASSES):
            gt_binary = (seg == c).astype(np.float32)
            pred_binary = (pred_classes == c).astype(np.float32)

            d = dice_score(gt_binary, pred_binary)
            iou = iou_score(gt_binary, pred_binary)
            prec = precision_score(gt_binary, pred_binary)
            rec = recall_score(gt_binary, pred_binary)
            f1 = f1_from(prec, rec)
            tp, fp, fn, tn = confusion_values(gt_binary, pred_binary)

            vol_metrics[class_names[c]] = {
                'dice': float(d), 'iou': float(iou),
                'precision': float(prec), 'recall': float(rec),
                'f1': float(f1),
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'voxel_count_gt': int(np.sum(gt_binary)),
                'voxel_count_pred': int(np.sum(pred_binary))
            }

            global_tp[c] += tp
            global_fp[c] += fp
            global_fn[c] += fn
            global_tn[c] += tn

        # Per-slice Dice for liver (for histogram)
        num_slices = vol.shape[2]
        for s in range(num_slices):
            gt_slice = (seg[:, :, s] == 1).astype(np.float32)
            pred_slice = (pred_classes[:, :, s] == 1).astype(np.float32)
            sd = dice_score(gt_slice, pred_slice)
            all_slice_dice_liver.append({
                'volume': vol_id, 'slice': s,
                'dice': float(sd),
                'has_liver': bool(np.sum(gt_slice) > 0)
            })

        # Also compute liver+tumor combined (all foreground)
        gt_fg = (seg > 0).astype(np.float32)
        pred_fg = (pred_classes > 0).astype(np.float32)
        fg_dice = dice_score(gt_fg, pred_fg)

        vol_summary = {
            'volume': vol_id,
            'original_shape': list(vol_raw.shape),
            'num_slices': num_slices,
            'foreground_dice': float(fg_dice),
            **{f'{cn}_dice': vol_metrics[cn]['dice'] for cn in class_names},
            **{f'{cn}_precision': vol_metrics[cn]['precision'] for cn in class_names},
            **{f'{cn}_recall': vol_metrics[cn]['recall'] for cn in class_names},
        }
        volume_results.append(vol_summary)

        print(f"  ✅ Background Dice: {vol_metrics['Background']['dice']:.4f}")
        print(f"     Liver Dice:      {vol_metrics['Liver']['dice']:.4f}")
        print(f"     Tumor Dice:      {vol_metrics['Tumor']['dice']:.4f}")
        print(f"     Foreground Dice: {fg_dice:.4f}")

    # ─── Global Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GLOBAL RESULTS")
    print("=" * 60)

    global_metrics = {}
    for c in range(NUM_CLASSES):
        cn = class_names[c]
        tp, fp, fn, tn = global_tp[c], global_fp[c], global_fn[c], global_tn[c]
        d = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)
        prec = tp / (tp + fp + 1e-7)
        rec = tp / (tp + fn + 1e-7)
        f1 = f1_from(prec, rec)
        acc = (tp + tn) / (tp + fp + fn + tn + 1e-7)

        global_metrics[cn] = {
            'dice': float(d), 'iou': float(iou),
            'precision': float(prec), 'recall': float(rec),
            'f1': float(f1), 'pixel_accuracy': float(acc),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }

        print(f"\n  {cn}:")
        print(f"    Dice:      {d:.4f}")
        print(f"    IoU:       {iou:.4f}")
        print(f"    Precision: {prec:.4f}  ({prec*100:.1f}%)")
        print(f"    Recall:    {rec:.4f}  ({rec*100:.1f}%)")
        print(f"    F1:        {f1:.4f}")

    # Per-slice stats
    all_dice_vals = [r['dice'] for r in all_slice_dice_liver]
    liver_dice_vals = [r['dice'] for r in all_slice_dice_liver if r['has_liver']]

    print(f"\n  Mean Slice Dice — Liver (all slices):  {np.mean(all_dice_vals):.4f} ± {np.std(all_dice_vals):.4f}")
    if liver_dice_vals:
        print(f"  Mean Slice Dice — Liver (liver only):  {np.mean(liver_dice_vals):.4f} ± {np.std(liver_dice_vals):.4f}")
    print(f"\n  Total slices evaluated: {len(all_dice_vals)}")
    print(f"  Slices with liver: {len(liver_dice_vals)}")

    # Confusion matrix display
    print(f"\n  Confusion Matrix (Liver class):")
    tp, fp, fn, tn = global_metrics['Liver']['tp'], global_metrics['Liver']['fp'], global_metrics['Liver']['fn'], global_metrics['Liver']['tn']
    print(f"    TP: {tp:>12,}    FP: {fp:>10,}")
    print(f"    FN: {fn:>12,}    TN: {tn:>10,}")

    # ─── Save Results ─────────────────────────────────────────────

    report = {
        'model': str(MODEL_PATH),
        'model_type': '3D U-Net',
        'input_shape': [128, 128, 268, 1],
        'output_shape': [128, 128, 268, 3],
        'num_classes': NUM_CLASSES,
        'class_names': class_names,
        'test_volumes': available,
        'global_metrics': global_metrics,
        'per_volume': volume_results,
        'slice_stats': {
            'total_slices': len(all_dice_vals),
            'liver_slices': len(liver_dice_vals),
            'mean_slice_dice_all': float(np.mean(all_dice_vals)),
            'std_slice_dice_all': float(np.std(all_dice_vals)),
            'mean_slice_dice_liver': float(np.mean(liver_dice_vals)) if liver_dice_vals else 0.0,
            'std_slice_dice_liver': float(np.std(liver_dice_vals)) if liver_dice_vals else 0.0,
        }
    }

    # 1. JSON report
    with open(OUTPUT_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n💾 Report: {OUTPUT_DIR / 'evaluation_report.json'}")

    # 2. CSV
    csv_path = OUTPUT_DIR / "per_volume_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=volume_results[0].keys())
        writer.writeheader()
        writer.writerows(volume_results)
    print(f"💾 CSV: {csv_path}")

    # 3. Summary text
    summary_path = OUTPUT_DIR / "evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("LITS17 3D U-Net Segmentation — Evaluation Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Type: 3D U-Net  |  Input: (128,128,268,1)  |  Output: (128,128,268,3)\n")
        f.write(f"Classes: Background (0), Liver (1), Tumor (2)\n")
        f.write(f"Test Volumes: {available}\n\n")
        for cn in class_names:
            m = global_metrics[cn]
            f.write(f"{cn}:\n")
            f.write(f"  Dice: {m['dice']:.4f}  |  IoU: {m['iou']:.4f}  |  ")
            f.write(f"Prec: {m['precision']:.4f}  |  Rec: {m['recall']:.4f}  |  F1: {m['f1']:.4f}\n")
        f.write(f"\nMean Slice Dice (liver, all): {np.mean(all_dice_vals):.4f}\n")
        if liver_dice_vals:
            f.write(f"Mean Slice Dice (liver only): {np.mean(liver_dice_vals):.4f}\n")
        f.write(f"\nConfusion Matrix (Liver):\n")
        f.write(f"  TP: {global_metrics['Liver']['tp']:>12,}   FP: {global_metrics['Liver']['fp']:>10,}\n")
        f.write(f"  FN: {global_metrics['Liver']['fn']:>12,}   TN: {global_metrics['Liver']['tn']:>10,}\n")
        for v in volume_results:
            f.write(f"\nVolume {v['volume']} (shape {v['original_shape']}):\n")
            f.write(f"  Liver Dice: {v['Liver_dice']:.4f}  |  Tumor Dice: {v['Tumor_dice']:.4f}  |  FG Dice: {v['foreground_dice']:.4f}\n")
    print(f"💾 Summary: {summary_path}")

    # 4. Dice histogram
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('3D U-Net Evaluation — Per-Slice Liver Dice Distribution',
                     fontsize=14, fontweight='bold')

        axes[0].hist(all_dice_vals, bins=50, color='#00b4d8', alpha=0.7, edgecolor='#0077b6')
        axes[0].axvline(x=np.mean(all_dice_vals), color='#e76f51', linestyle='--',
                        label=f'Mean: {np.mean(all_dice_vals):.4f}')
        axes[0].set_title('All Slices')
        axes[0].set_xlabel('Dice Score')
        axes[0].set_ylabel('Count')
        axes[0].legend()

        if liver_dice_vals:
            axes[1].hist(liver_dice_vals, bins=50, color='#00d4aa', alpha=0.7, edgecolor='#008060')
            axes[1].axvline(x=np.mean(liver_dice_vals), color='#e76f51', linestyle='--',
                            label=f'Mean: {np.mean(liver_dice_vals):.4f}')
            axes[1].set_title('Liver Slices Only')
            axes[1].set_xlabel('Dice Score')
            axes[1].set_ylabel('Count')
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "dice_histogram.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Histogram: {OUTPUT_DIR / 'dice_histogram.png'}")

        # Per-volume bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(volume_results))
        w = 0.25
        ax.bar(x - w, [v['Liver_dice'] for v in volume_results], w, label='Liver', color='#00b4d8')
        ax.bar(x, [v['Tumor_dice'] for v in volume_results], w, label='Tumor', color='#e76f51')
        ax.bar(x + w, [v['foreground_dice'] for v in volume_results], w, label='Foreground', color='#00d4aa')
        ax.set_xlabel('Volume')
        ax.set_ylabel('Dice Score')
        ax.set_title('Per-Volume Dice Scores')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Vol {v['volume']}" for v in volume_results])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "per_volume_dice.png", dpi=150)
        plt.close()
        print(f"📊 Per-volume chart: {OUTPUT_DIR / 'per_volume_dice.png'}")
    except ImportError:
        print("  (matplotlib not available — skipping plots)")

    print("\n✅ Evaluation complete!")
    return report


if __name__ == "__main__":
    evaluate()