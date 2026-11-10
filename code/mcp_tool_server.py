"""
MCP Tool Server for Liver Lesion Intelligence System
Exposes segmentation pipeline tools via Model Context Protocol (MCP)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
SLICE_CACHE = Path(__file__).resolve().parent / "slice_cache"
DATA_DIRS = [BASE_DIR / "training_data", BASE_DIR / "training_data_v2"]
EVAL_METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

class ToolResult(BaseModel):
    tool_name: str
    status: str
    result: Any
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

def lookup_volume_metadata(volume_id=None):
    cached_jobs = []
    if SLICE_CACHE.exists():
        for d in sorted(SLICE_CACHE.iterdir()):
            if d.is_dir():
                slices = list(d.glob("slice_*.png"))
                cached_jobs.append({
                    "job_id": d.name,
                    "slice_count": len(slices),
                    "slice_indices": sorted([int(s.stem.split("_")[1]) for s in slices]),
                })
    if volume_id:
        match = [j for j in cached_jobs if j["job_id"] == volume_id]
        if not match:
            return {"error": f"Volume '{volume_id}' not found", "available": [j["job_id"] for j in cached_jobs]}
        return {"volume": match[0]}
    raw_volumes = []
    for data_dir in DATA_DIRS:
        vol_dir = data_dir / "volumes"
        if vol_dir.exists():
            raw_volumes.extend([f.name for f in sorted(vol_dir.iterdir()) if f.suffix in (".nii", ".nii.gz")])
    models = []
    if MODELS_DIR.exists():
        models = [f.name for f in sorted(MODELS_DIR.iterdir()) if f.suffix in (".keras", ".h5")]
    return {"cached_analyses": cached_jobs, "cached_count": len(cached_jobs), "raw_volumes_available": len(raw_volumes), "models": models, "active_model": "best_retrained_3d_unet.keras"}

def retrieve_slice(job_id, slice_index=None):
    job_dir = SLICE_CACHE / job_id
    if not job_dir.exists():
        return {"error": f"No cached slices for job '{job_id}'", "available_jobs": [d.name for d in SLICE_CACHE.iterdir() if d.is_dir()] if SLICE_CACHE.exists() else []}
    slices = sorted(job_dir.glob("slice_*.png"))
    available = [int(s.stem.split("_")[1]) for s in slices]
    if slice_index is None:
        return {"job_id": job_id, "total_slices": len(slices), "available_indices": available, "slice_dir": str(job_dir)}
    target_file = job_dir / f"slice_{slice_index:04d}.png"
    if target_file.exists():
        return {"job_id": job_id, "slice_index": slice_index, "file_path": str(target_file), "url": f"/api/slice-image/{job_id}/{slice_index}"}
    if available:
        nearest = min(available, key=lambda x: abs(x - slice_index))
        return {"job_id": job_id, "requested_index": slice_index, "nearest_index": nearest, "file_path": str(job_dir / f"slice_{nearest:04d}.png"), "url": f"/api/slice-image/{job_id}/{nearest}"}
    return {"error": "No slices found"}

def generate_report(job_id):
    results_file = SLICE_CACHE / job_id / "pipeline_result.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    job_dir = SLICE_CACHE / job_id
    if not job_dir.exists():
        return {"error": f"No analysis found for job '{job_id}'", "hint": "Submit a volume via POST /api/analyze first", "available_jobs": [d.name for d in SLICE_CACHE.iterdir() if d.is_dir()] if SLICE_CACHE.exists() else []}
    slices = list(job_dir.glob("slice_*.png"))
    return {"job_id": job_id, "status": "partial", "segmentation_slices": len(slices), "note": "Segmentation slices exist but full report not cached."}

def get_model_metrics():
    EVAL_METRICS_PATH = Path.home() / "LITS17" / "models" / "retrained" / "v1_metrics.json"
    if not EVAL_METRICS_PATH.exists():
        return {"error": "No evaluation_metrics.json found", "model_dir": str(MODELS_DIR)}
    with open(EVAL_METRICS_PATH) as f:
        metrics = json.load(f)
    return {"model": "best_retrained_3d_unet.keras", "architecture": "3D U-Net (128x128x268, background/liver/tumor)", "dataset": "LiTS17 (131 volumes)", "metrics": metrics}

def get_segmentation_stats(job_id):
    result_file = SLICE_CACHE / job_id / "pipeline_result.json"
    if not result_file.exists():
        return {"error": f"No pipeline result for job '{job_id}'"}
    with open(result_file) as f:
        data = json.load(f)
    seg = data.get("segmentation") or {}
    return {"job_id": job_id, "liver_volume_ml": seg.get("liver_volume"),
            "tumor_burden_pct": seg.get("total_tumor_burden"),
            "tumor_detection": (data.get("report") or {}).get("tumor_detection")}

TOOL_REGISTRY = {
    "lookup_volume_metadata": {
        "function": lookup_volume_metadata,
        "definition": ToolDefinition(name="lookup_volume_metadata", description="Look up metadata about available CT volumes, cached analyses, and models.", input_schema={"type": "object", "properties": {"volume_id": {"type": "string", "description": "Optional job/volume ID"}}, "required": []}),
    },
    "retrieve_slice": {
        "function": retrieve_slice,
        "definition": ToolDefinition(name="retrieve_slice", description="Retrieve a segmentation overlay slice image from the cache.", input_schema={"type": "object", "properties": {"job_id": {"type": "string", "description": "The job/volume ID"}, "slice_index": {"type": "integer", "description": "The slice number to retrieve"}}, "required": ["job_id"]}),
    },
    "generate_report": {
        "function": generate_report,
        "definition": ToolDefinition(name="generate_report", description="Retrieve the full clinical analysis report for a completed pipeline run.", input_schema={"type": "object", "properties": {"job_id": {"type": "string", "description": "The job/volume ID"}}, "required": ["job_id"]}),
    },
    "get_model_metrics": {
        "function": get_model_metrics,
        "definition": ToolDefinition(name="get_model_metrics", description="Return evaluation metrics (Dice, precision, recall, F1, confusion matrix) from the most recent training run.", input_schema={"type": "object", "properties": {}, "required": []}),
    },
    "get_segmentation_stats": {
        "function": get_segmentation_stats,
        "definition": ToolDefinition(name="get_segmentation_stats", description="Return the liver volume and predicted tumor burden for a completed analysis job.", input_schema={"type": "object", "properties": {"job_id": {"type": "string", "description": "The job/volume ID"}}, "required": ["job_id"]}),
    },
}

mcp_app = FastAPI(title="Liver Intelligence MCP Tool Server", version="1.0.0")

@mcp_app.get("/mcp/tools")
async def list_tools():
    return {"tools": [entry["definition"].model_dump() for entry in TOOL_REGISTRY.values()]}

@mcp_app.post("/mcp/call")
async def call_tool(tool_call: ToolCall):
    if tool_call.name not in TOOL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_call.name}' not found. Available: {list(TOOL_REGISTRY.keys())}")
    entry = TOOL_REGISTRY[tool_call.name]
    try:
        result = entry["function"](**tool_call.arguments)
        return ToolResult(tool_name=tool_call.name, status="success", result=result)
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid arguments: {e}")
    except Exception as e:
        return ToolResult(tool_name=tool_call.name, status="error", result={"error": str(e)})

@mcp_app.get("/mcp/health")
async def mcp_health():
    return {"status": "healthy", "tools_available": len(TOOL_REGISTRY), "tool_names": list(TOOL_REGISTRY.keys()), "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    print("Starting MCP Tool Server on port 8001...")
    print(f"Tools: {list(TOOL_REGISTRY.keys())}")
    uvicorn.run(mcp_app, host="0.0.0.0", port=8001)
