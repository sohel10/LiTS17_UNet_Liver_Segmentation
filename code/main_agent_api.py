"""
FastAPI backend for Autonomous Liver Lesion Intelligence System
Integrates U-Net segmentation with multi-agent reasoning
Updated for LangGraph 1.2.20 / LangChain 1.4.9
"""

import asyncio
import uuid
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import numpy as np
import nibabel as nib
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
from starlette.requests import Request
import tensorflow as tf

from agent_orchestrator import run_liver_intelligence_pipeline

# ============================================================================
# INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Autonomous Liver Lesion Intelligence System",
    description="Deep Learning + Multi-Agent Reasoning for Liver Imaging Analysis",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# templates removed — using React frontend

JOBS: Dict[str, Dict[str, Any]] = {}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_unet_lits17_balanced.keras"

print("=" * 80)
print("AUTONOMOUS LIVER LESION INTELLIGENCE SYSTEM")
print("Model: U-Net Balanced (LiTS17)")
print("Agents: Segmentation -> Classification -> Evidence -> Diagnosis -> Report")
print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {MODEL_PATH.exists()}")
print("=" * 80)

# ============================================================================
# UTILITIES
# ============================================================================

def load_nifti_volume(nifti_path: str) -> tuple:
    img = nib.load(nifti_path)
    data = img.get_fdata()
    return data, {
        "shape": data.shape,
        "affine": img.affine.tolist(),
        "voxel_mm3": float(img.header.get_zooms()[0]) * float(img.header.get_zooms()[1]) * float(img.header.get_zooms()[2]),
        "units": dict(zip(["spatial", "temporal"], img.header.get_xyzt_units())),
    }

# ============================================================================
# REST ENDPOINTS
# ============================================================================

# React frontend serves / via serve_app.py

@app.get("/api/status")
async def status():
    return {
        "status": "online",
        "system": "Autonomous Liver Lesion Intelligence",
        "model": "3D U-Net v1 (LiTS17)",
        "model_loaded": MODEL_PATH.exists(),
        "agents_active": 2,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/analyze")
async def analyze_liver(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not file.filename.endswith(('.nii', '.nii.gz')):
        raise HTTPException(status_code=400, detail="File must be NIfTI (.nii or .nii.gz)")

    job_id = str(uuid.uuid4())[:8]
    temp_path = f"/tmp/{job_id}_{file.filename}"
    with open(temp_path, 'wb') as f:
        contents = await file.read()
        f.write(contents)

    JOBS[job_id] = {
        "status": "queued",
        "file": file.filename,
        "created": datetime.now().isoformat(),
        "progress": 0,
        "current_agent": None,
        "results": None,
        "error": None
    }

    if background_tasks:
        background_tasks.add_task(process_liver_analysis, job_id, temp_path)
    else:
        asyncio.create_task(process_liver_analysis(job_id, temp_path))

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Analysis queued. Use /api/jobs/{job_id} to poll status"
    }

async def process_liver_analysis(job_id: str, file_path: str):
    try:
        JOBS[job_id]["status"] = "loading"
        JOBS[job_id]["progress"] = 10
        volume, metadata = load_nifti_volume(file_path)

        JOBS[job_id]["progress"] = 20
        JOBS[job_id]["status"] = "analyzing"
        JOBS[job_id]["current_agent"] = "SegmentationAgent"

        results = await run_liver_intelligence_pipeline(volume, patient_id=job_id, voxel_mm3=metadata["voxel_mm3"])

        JOBS[job_id]["results"] = results
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["status"] = "completed"

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)

    finally:
        Path(file_path).unlink(missing_ok=True)

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "current_agent": job["current_agent"],
        "created": job["created"],
    }
    if job["status"] == "completed" and job["results"]:
        response["results"] = job["results"]
    elif job["status"] == "error":
        response["error"] = job["error"]
    return response

@app.get("/api/jobs/{job_id}/report")
async def get_report(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {job['status']}")
    if not job["results"] or not job["results"].get("report"):
        raise HTTPException(status_code=400, detail="No report available")
    return job["results"]["report"]

@app.get("/api/jobs/{job_id}/execution-log")
async def get_execution_log(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    return {
        "job_id": job_id,
        "log": job["results"].get("execution_log", []) if job["results"] else [],
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

# --- Slice Visualization Endpoint ---
from fastapi.responses import FileResponse as SliceFileResponse

@app.get("/api/jobs/{job_id}/slices")
async def list_slices(job_id: str):
    slice_dir = Path(__file__).resolve().parent / "slice_cache" / job_id
    if not slice_dir.exists():
        return {"slices": [], "count": 0}
    files = sorted(slice_dir.glob("slice_*.png"))
    indices = [int(f.stem.split("_")[1]) for f in files]
    return {"slices": indices, "count": len(indices)}

@app.get("/api/jobs/{job_id}/slices/{slice_idx}")
async def get_slice(job_id: str, slice_idx: int):
    from fastapi import HTTPException
    slice_path = Path(__file__).resolve().parent / "slice_cache" / job_id / f"slice_{slice_idx:04d}.png"
    if not slice_path.exists():
        raise HTTPException(status_code=404, detail="Slice not found")
    return SliceFileResponse(str(slice_path), media_type="image/png")

@app.get("/api/slice-jobs")
async def list_slice_jobs():
    """List all jobs that have saved slice images"""
    slice_dir = Path(__file__).resolve().parent / "slice_cache"
    if not slice_dir.exists():
        return {"jobs": []}
    dirs = sorted([d for d in slice_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    return {"jobs": [d.name for d in dirs]}

# --- PDF Report Generation ---
@app.get("/api/jobs/{job_id}/report/pdf")
async def generate_pdf_report(job_id: str):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from xml.sax.saxutils import escape
    from io import BytesIO
    from fastapi.responses import StreamingResponse

    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not complete")

    result = job.get("results") or {}
    seg = result.get("segmentation") or {}
    report = result.get("report") or {}

    def fmt(v, suffix=""):
        if isinstance(v, (int, float)):
            return f"{v:.2f}{suffix}" if isinstance(v, float) else f"{v}{suffix}"
        return "n/a"

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=18, textColor=colors.HexColor('#1e3a5f'), spaceAfter=12)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#1e3a5f'), spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle('CustomBody', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=4)
    small_style = ParagraphStyle('Small', parent=styles['Normal'], fontSize=8, textColor=colors.gray)

    story = []
    story.append(Paragraph("Liver Lesion Intelligence: Research Report", title_style))
    story.append(Paragraph(f"Job ID: {escape(job_id)} | Generated: {escape(str(job.get('created_at', 'N/A')))}", small_style))
    story.append(Spacer(1, 12))

    summary_data = [
        ["Metric", "Value"],
        ["Liver volume", f"{fmt(report.get('liver_volume_ml'))} mL"],
        ["Predicted tumor burden", f"{fmt(report.get('tumor_burden_pct'))}%"],
        ["Tumor detection", str(report.get("tumor_detection", "n/a"))],
        ["Validation Dice (liver)", fmt(report.get("validation_liver_dice"))],
        ["Validation Dice (tumor)", fmt(report.get("validation_tumor_dice"))],
    ]
    t = Table(summary_data, colWidths=[2.8*inch, 2.7*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f0f4f8')]),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)

    sections = [
        ("Findings", report.get("findings") or []),
        ("Model performance", report.get("model_performance") or []),
        ("Limitations", report.get("limitations") or []),
    ]
    for title, items in sections:
        if items:
            story.append(Paragraph(title, heading_style))
            for item in items:
                story.append(Paragraph("&bull; " + escape(str(item)), body_style))

    story.append(Paragraph("Recommendation", heading_style))
    story.append(Paragraph(escape(str(report.get("recommendation") or report.get("statement", ""))), body_style))

    story.append(Spacer(1, 24))
    story.append(Paragraph("Built by Sohel Ahmed, PhD. Research prototype, not for clinical use.", small_style))
    story.append(Paragraph("This report is for research purposes only. Not for clinical diagnosis.", small_style))

    doc.build(story)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=liver_report_{job_id}.pdf"})

if __name__ == "__main__":
    import uvicorn
    print("\nStarting Autonomous Liver Lesion Intelligence API on port 8000\n")
    from starlette.responses import FileResponse
    from starlette.staticfiles import StaticFiles
    from starlette.responses import FileResponse
    import os
    # Serve React frontend from build folder
    import os as _os
    _build = _os.path.join(_os.path.dirname(__file__), 'frontend', 'build')
    if _os.path.isdir(_build):
        from starlette.staticfiles import StaticFiles
        from starlette.responses import FileResponse
        app.mount('/static', StaticFiles(directory=_os.path.join(_build, 'static')), name='static')
        @app.get('/{full_path:path}')
        async def serve_react(full_path: str):
            if full_path.startswith("api/"):
                from fastapi.responses import JSONResponse
                return JSONResponse({"error": "not found"}, status_code=404)
            fp = _os.path.join(_build, full_path)
            if _os.path.isfile(fp):
                return FileResponse(fp)
            return FileResponse(_os.path.join(_build, 'index.html'))

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

