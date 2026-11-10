"""
Standalone v1 3D U-Net inference for LiTS17.
Input: a volume-*.nii CT file. Output: a liver/tumor mask saved as NIfTI.
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom

LITS_ROOT = Path.home() / "LITS17"
MODEL_PATH = LITS_ROOT / "models" / "retrained" / "best_retrained_3d_unet.keras"
TARGET_SHAPE = (128, 128, 268)

_model = None

def load_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    return _model

def preprocess(ct):
    # Same as train_3d_unet.py: clip window, then per-volume min-max
    ct = np.clip(ct, -200, 250).astype(np.float32)
    ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-7)
    factors = [t / s for t, s in zip(TARGET_SHAPE, ct.shape)]
    return zoom(ct, factors, order=1)

def predict(nii_path, out_path=None):
    img = nib.load(str(nii_path))
    ct = img.get_fdata()
    original_shape = ct.shape

    x = preprocess(ct)[..., np.newaxis][np.newaxis]   # (1,128,128,268,1)
    probs = load_model().predict(x, verbose=0)[0]     # (128,128,268,3)
    mask_small = np.argmax(probs, axis=-1).astype(np.uint8)

    factors = [o / t for o, t in zip(original_shape, TARGET_SHAPE)]
    mask = zoom(mask_small, factors, order=0).astype(np.uint8)

    if out_path is None:
        out_path = Path(nii_path).with_name("predicted_mask.nii.gz")
    nib.save(nib.Nifti1Image(mask, img.affine), str(out_path))

    counts = {"background": int((mask == 0).sum()),
              "liver": int((mask == 1).sum()),
              "tumor": int((mask == 2).sum())}
    return out_path, counts

def predict_array(ct):
    """Take a CT array (x, y, z) and return a 3D mask on the same grid."""
    original_shape = ct.shape
    x = preprocess(ct)[..., np.newaxis][np.newaxis]
    probs = load_model().predict(x, verbose=0)[0]
    mask_small = np.argmax(probs, axis=-1).astype(np.uint8)
    factors = [o / t for o, t in zip(original_shape, TARGET_SHAPE)]
    return zoom(mask_small, factors, order=0).astype(np.uint8)

if __name__ == "__main__":
    out, counts = predict(sys.argv[1])
    print(f"Saved mask: {out}")
    print(f"Voxel counts: {counts}")