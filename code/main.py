"""
Updated FastAPI Server for Liver Segmentation with Weighted U-Net
Now uses: best_unet_lits17_balanced.keras (Dice 0.8131, Recall 0.71+)
"""

import os
import numpy as np
import nibabel as nib
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import tensorflow as tf
from pathlib import Path
import base64
from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile

app = FastAPI(title="LiTS17 U-Net Liver Segmentation - Weighted")
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

MODEL = None
MODEL_STATUS = "Not loaded"
MODEL_VERSION = "best_unet_lits17_balanced.keras"

MODEL_METRICS = {
    "dice": 0.8131,
    "recall": 0.71,
    "sensitivity": 0.71,
    "specificity": 0.99,
    "training_cases": 17,
    "test_cases": 3
}

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))

@app.on_event("startup")
async def load_model():
    global MODEL, MODEL_STATUS
    try:
        model_path = BASE_DIR / MODEL_VERSION
        
        if not model_path.exists():
            MODEL_STATUS = f"Error: {MODEL_VERSION} not found"
            print(f"❌ {MODEL_STATUS}")
            return
        
        print(f"\n{'='*70}")
        print(f"🚀 Loading Weighted Model")
        print(f"{'='*70}")
        
        tf.keras.config.enable_unsafe_deserialization()
        
        MODEL = tf.keras.models.load_model(
            str(model_path),
            custom_objects={
                "bce_dice_loss": bce_dice_loss,
                "dice_coef": dice_coef
            }
        )
        
        print(f"✅ Model loaded!")
        print(f"   Dice: {MODEL_METRICS['dice']}")
        print(f"   Recall: {MODEL_METRICS['recall']}")
        print(f"{'='*70}\n")
        
        MODEL_STATUS = f"✅ Loaded - Weighted (Dice: {MODEL_METRICS['dice']})"
        
    except Exception as e:
        MODEL_STATUS = f"❌ Failed: {str(e)}"
        print(MODEL_STATUS)

def preprocess_slice(slice_2d):
    from PIL import Image as PILImage
    
    slice_2d = slice_2d.astype(np.float32)
    arr_min, arr_max = slice_2d.min(), slice_2d.max()
    if arr_max > arr_min:
        slice_2d = (slice_2d - arr_min) / (arr_max - arr_min)
    
    img = PILImage.fromarray(slice_2d)
    img = img.resize((256, 256), PILImage.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    return arr

def find_best_slice(volume):
    best_slice = None
    best_coverage = 0
    best_idx = 0
    
    total = volume.shape[2]
    search_range = [int(total * i / 10) for i in range(2, 7)]
    
    for idx in search_range:
        try:
            slice_2d = volume[:, :, idx]
            preprocessed = preprocess_slice(slice_2d)
            input_tensor = preprocessed[np.newaxis, :, :, np.newaxis]
            pred = MODEL.predict(input_tensor, verbose=0)[0, :, :, 0]
            coverage = int((pred > 0.5).sum())
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_slice = slice_2d
                best_idx = idx
        except:
            continue
    
    return best_slice, best_idx, best_coverage

def predict_and_overlay(slice_2d):
    preprocessed = preprocess_slice(slice_2d)
    input_tensor = preprocessed[np.newaxis, :, :, np.newaxis]
    
    pred = MODEL.predict(input_tensor, verbose=0)[0, :, :, 0]
    pred_mask = (pred > 0.5).astype(np.uint8)
    coverage = float(pred_mask.sum()) / (256 * 256) * 100
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0d1117')
    
    axes[0].imshow(preprocessed, cmap='gray')
    axes[0].set_title('CT Input', color='white', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(pred_mask, cmap='Reds')
    axes[1].set_title('Liver Mask', color='white', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(preprocessed, cmap='gray')
    masked = np.ma.masked_where(pred_mask == 0, pred_mask)
    axes[2].imshow(masked, cmap='Reds', alpha=0.6)
    axes[2].set_title(f'Overlay ({coverage:.1f}%)', color='white', fontsize=13, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout(pad=1.5)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64, coverage

def get_overlay_images():
    outputs_dir = BASE_DIR / "outputs"
    if not outputs_dir.exists():
        return []
    return [f"/outputs/{f.name}" for f in sorted(outputs_dir.glob("prediction_overlay_*.png"))]

def get_prediction_grid():
    grid = BASE_DIR / "outputs" / "prediction_grid.png"
    return "/outputs/prediction_grid.png" if grid.exists() else None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    overlays = get_overlay_images()
    grid = get_prediction_grid()
    
    if MODEL:
        params_str = f"{MODEL.count_params():,}"
        input_shape = str(MODEL.input_shape)
        output_shape = str(MODEL.output_shape)
    else:
        params_str = "N/A"
        input_shape = "N/A"
        output_shape = "N/A"
    
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_status": MODEL_STATUS,
            "model_version": MODEL_VERSION,
            "params": params_str,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "overlays": overlays,
            "grid": grid,
            "overlay_count": len(overlays),
            "tf_version": tf.__version__,
            "dice": MODEL_METRICS["dice"],
            "recall": MODEL_METRICS["recall"],
            "sensitivity": MODEL_METRICS["sensitivity"],
            "specificity": MODEL_METRICS["specificity"],
        }
    )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        return JSONResponse(
            {"error": "Model not loaded"},
            status_code=503
        )
    
    try:
        suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        nii = nib.load(tmp_path)
        volume = nii.get_fdata()
        os.unlink(tmp_path)
        
        print(f"\n📊 Processing: {file.filename}")
        print(f"   Shape: {volume.shape}")
        
        best_slice, best_idx, best_coverage = find_best_slice(volume)
        
        if best_slice is None:
            return JSONResponse({"error": "No suitable slice"}, status_code=400)
        
        img_b64, coverage = predict_and_overlay(best_slice)
        
        print(f"   ✅ Slice {best_idx}: {coverage:.1f}%")
        
        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "volume_shape": list(volume.shape),
            "slice_used": best_idx,
            "coverage_percent": round(coverage, 2),
            "overlay_image": img_b64,
            "model_version": MODEL_VERSION,
            "dice_score": MODEL_METRICS["dice"],
            "recall": MODEL_METRICS["recall"],
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
async def health():
    return {
        "status": "running",
        "model_loaded": MODEL is not None,
        "model_version": MODEL_VERSION,
        "dice": MODEL_METRICS["dice"],
        "recall": MODEL_METRICS["recall"],
    }

@app.get("/metrics")
async def get_metrics():
    return {
        "model": MODEL_VERSION,
        "metrics": MODEL_METRICS,
        "loaded": MODEL is not None
    }

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(BASE_DIR / "outputs")), name="outputs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
