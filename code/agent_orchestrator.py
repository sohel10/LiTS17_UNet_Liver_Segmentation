"""
Autonomous Liver Lesion Intelligence System
Multi-agent orchestration for liver segmentation, classification, and reporting
Updated for LangGraph 1.2.20 / LangChain 1.4.9

MCP Tool Integration: Agents 2-5 call the MCP Tool Server (port 8001) for
metadata lookup, slice retrieval, report caching, and model metrics --
demonstrating genuine tool-use capability in an agentic pipeline.
"""

import json
import re
import operator
from enum import Enum
from typing import Any, Dict, List, Optional, Annotated
from typing_extensions import TypedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import requests
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from predict_v1 import predict_array

# Initialize Claude as the reasoning engine
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# MCP Tool Server endpoint
MCP_SERVER_URL = "http://localhost:8001"

# ============================================================================
# REPORT AGENT (factual summary, no clinical grading)
# ============================================================================

def report_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    seg = state.get("segmentation_result") or {}
    try:
        r = requests.post(f"{MCP_SERVER_URL}/mcp/call",
                          json={"name": "get_model_metrics", "arguments": {}}, timeout=10)
        metrics = r.json().get("result", {}).get("metrics", {})
    except Exception as e:
        metrics = {"error": str(e)}
    val = metrics.get("validation_15pct_held_out", {})
    liver_ml = seg.get("liver_volume")
    burden = seg.get("total_tumor_burden")
    liver_txt = f"{liver_ml:.1f} mL" if isinstance(liver_ml, (int, float)) else "not available"
    burden_txt = f"{burden:.2f}%" if isinstance(burden, (int, float)) else "not available"
    liver_dice = val.get("liver_dice")
    tumor_dice = val.get("tumor_dice")
    report = {
        "liver_volume_ml": round(liver_ml, 1) if isinstance(liver_ml, (int, float)) else None,
        "tumor_burden_pct": round(burden, 2) if isinstance(burden, (int, float)) else None,
        "tumor_detection": "Weak",
        "model": "3D U-Net (v1)",
        "validation_liver_dice": liver_dice,
        "validation_tumor_dice": tumor_dice,
        "findings": [
            f"Estimated liver volume: {liver_txt}, computed from the predicted liver mask and CT voxel spacing.",
            f"Predicted tumor burden: {burden_txt} of the liver volume. This is a model prediction, not a verified measurement.",
            "Tumor detection is weak. Check the Imaging page overlays before trusting any tumor region.",
        ],
        "model_performance": [
            f"Liver validation Dice: {liver_dice if liver_dice is not None else 'n/a'} (held-out 15% of training data).",
            f"Tumor validation Dice: {tumor_dice if tumor_dice is not None else 'n/a'}. A low tumor overlap means the model often misses or misplaces tumors.",
        ],
        "limitations": [
            "Trained on the public LiTS17 dataset only; not tested on hospital scans.",
            "Tumor predictions can include false positives and miss real tumors.",
            "Metrics come from one training run and one validation split.",
        ],
        "recommendation": "Research and demonstration use only. Review the overlays. Do not use this output for diagnosis or treatment decisions.",
        "statement": "Research output from a segmentation model. Not a diagnosis. Tumor detection is weak.",
    }
    log_entry = {"timestamp": datetime.now().isoformat(), "agent": "ReportAgent",
                 "status": "COMPLETED", "details": "Report built from segmentation and MCP model metrics"}
    mcp_ok = "error" not in metrics
    call = {"tool_name": "get_model_metrics", "status": "success" if mcp_ok else "error"}
    return {"final_report": report, "execution_log": [log_entry], "mcp_tool_calls": [call], "mcp_server_available": mcp_ok}

# ============================================================================
# MCP TOOL CLIENT
# ============================================================================

def call_mcp_tool(tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
    """Call a tool on the MCP Tool Server via HTTP POST."""
    try:
        response = requests.post(
            f"{MCP_SERVER_URL}/mcp/call",
            json={"name": tool_name, "arguments": arguments or {}},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return {
            "tool_name": result.get("tool_name", tool_name),
            "status": result.get("status", "unknown"),
            "result": result.get("result", {}),
            "timestamp": result.get("timestamp", datetime.now().isoformat())
        }
    except requests.ConnectionError:
        return {
            "tool_name": tool_name,
            "status": "error",
            "result": {"error": f"MCP server not reachable at {MCP_SERVER_URL}"},
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "tool_name": tool_name,
            "status": "error",
            "result": {"error": str(e)},
            "timestamp": datetime.now().isoformat()
        }


def check_mcp_health() -> bool:
    """Check if the MCP Tool Server is healthy."""
    try:
        r = requests.get(f"{MCP_SERVER_URL}/mcp/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

# ============================================================================
# STATE (TypedDict for LangGraph 1.2.20)
# ============================================================================

class AgentState(TypedDict, total=False):
    ct_volume: np.ndarray
    voxel_mm3: float
    report: Any
    mcp_server_available: bool
    patient_id: str
    imaging_modality: str
    segmentation_result: Optional[Dict[str, Any]]
    classification_result: Optional[Dict[str, Any]]
    guidance_result: Optional[Dict[str, Any]]
    differential_result: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]
    execution_log: Annotated[List[Dict[str, Any]], operator.add]
    mcp_tool_calls: Annotated[List[Dict[str, Any]], operator.add]

# ============================================================================
# ENUMS
# ============================================================================

class LIRADSCategory(Enum):
    LR1 = "LR-1: Definitely benign"
    LR2 = "LR-2: Probably benign"
    LR3 = "LR-3: Intermediate probability HCC"
    LR4 = "LR-4: Probably HCC"
    LR5 = "LR-5: Definitely HCC"
    LR_NC = "LR-NC: Not categorizable"

# ============================================================================
# AGENT 1: SEGMENTATION (direct model inference -- no MCP needed)
# ============================================================================

def segmentation_agent(state: AgentState) -> Dict[str, Any]:
    import tensorflow as tf
    from PIL import Image
    from scipy import ndimage
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": "SegmentationAgent",
        "status": "STARTED",
        "details": "Loading U-Net model and processing volume"
    }

    try:
        ct_volume = state["ct_volume"]
        from predict_v1 import predict_array
        mask3d = predict_array(ct_volume)
        predictions = [mask3d[:, :, i] for i in range(mask3d.shape[2])]

        # Save overlay slices for visualization
        slice_dir = Path(__file__).resolve().parent / "slice_cache"
        job_id = state.get("patient_id", "unknown")
        job_slice_dir = slice_dir / job_id
        job_slice_dir.mkdir(parents=True, exist_ok=True)
        step = max(1, len(predictions) // 20)
        saved_slices = []
        for i in range(0, len(predictions), step):
            fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=80)
            ct_s = ct_volume[:, :, i]
            ct_s_norm = (ct_s - ct_s.min()) / max(1, ct_s.max() - ct_s.min())
            ax.imshow(ct_s_norm, cmap='gray', aspect='equal')
            mask = predictions[i]
            if (mask == 1).any():
                ax.contour((mask == 1).astype(float), levels=[0.5], colors=['#22c55e'], linewidths=1.5)
            if (mask == 2).any():
                ax.contour((mask == 2).astype(float), levels=[0.5], colors=['#ef4444'], linewidths=1.5)
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color='#22c55e', label='Liver'), Patch(color='#ef4444', label='Tumor (weak)')],
                      loc='lower left', fontsize=8, facecolor='#0f172a', labelcolor='white', framealpha=0.8)
            ax.set_axis_off()
            ax.set_title(f'Slice {i}', color='white', fontsize=10)
            fig.patch.set_facecolor('#0f172a')
            fig.tight_layout(pad=0.5)
            out_path = job_slice_dir / f"slice_{i:04d}.png"
            fig.savefig(str(out_path), facecolor='#0f172a', bbox_inches='tight')
            plt.close(fig)
            saved_slices.append(i)

        # Count tumors in 3D (label 2 only), not liver pieces
        mask3d_all = np.stack(predictions, axis=2)
        liver_volume_voxels = int(np.sum(mask3d_all > 0))
        labeled_3d, num_features = ndimage.label(mask3d_all == 2)
        lesion_count = 0
        lesion_sizes = []
        total_tumor_voxels = 0
        for lesion_id in range(1, num_features + 1):
            lesion_volume = int(np.sum(labeled_3d == lesion_id))
            if lesion_volume > 50:
                lesion_count += 1
                lesion_sizes.append(float(np.sqrt(lesion_volume)))
                total_tumor_voxels += lesion_volume
        liver_volume_ml = liver_volume_voxels * state.get("voxel_mm3", 1.0) / 1000.0
        tumor_burden_pct = (total_tumor_voxels / max(1, liver_volume_voxels)) * 100

        seg_result = {
            "lesion_count": None,
            "lesion_sizes": [],
            "lesion_locations": [],
            "tumor_warning": "Tumor detection is weak (validation Dice 0.216). Shown for reference only, not a diagnosis.",
            "liver_volume": liver_volume_ml,
            "total_tumor_burden": tumor_burden_pct,
            "imaging_modality": "CT",
            "timestamp": datetime.now().isoformat()
        }

        log_entry["status"] = "COMPLETED"
        log_entry["details"] = f"Liver volume {liver_volume_ml:.1f} mL; {lesion_count} tumor region(s) above 50 voxels (weak detection)"

        return {
            "segmentation_result": seg_result,
            "execution_log": [log_entry],
            "mcp_tool_calls": []
        }

    except Exception as e:
        log_entry["status"] = "ERROR"
        log_entry["details"] = str(e)
        return {"execution_log": [log_entry], "mcp_tool_calls": []}

# ============================================================================
# AGENT 2: LI-RADS CLASSIFICATION (uses MCP: get_model_metrics, lookup_volume_metadata)
# ============================================================================

def lirads_classification_agent(state: AgentState) -> Dict[str, Any]:
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": "LIRADSClassificationAgent",
        "status": "STARTED",
        "details": "Applying LI-RADS criteria with MCP tool augmentation"
    }
    tool_calls = []

    seg = state.get("segmentation_result")
    if not seg:
        log_entry["status"] = "ERROR"
        log_entry["details"] = "No segmentation result available"
        return {"execution_log": [log_entry], "mcp_tool_calls": []}

    try:
        # -- MCP Tool Call 1: Get model metrics for confidence calibration --
        metrics_result = call_mcp_tool("get_model_metrics")
        tool_calls.append(metrics_result)

        model_context = ""
        if metrics_result["status"] == "success":
            metrics = metrics_result["result"].get("metrics", {})
            dice = metrics.get("global_dice_score", "N/A")
            precision = metrics.get("precision", "N/A")
            recall = metrics.get("recall", "N/A")
            model_context = f"""
Model Performance Context (from MCP get_model_metrics):
- Global Dice Score: {dice}
- Precision: {precision}
- Recall: {recall}
- Architecture: {metrics_result['result'].get('architecture', '2D U-Net')}
Calibrate your confidence scores relative to these model metrics.
"""

        # -- MCP Tool Call 2: Lookup volume metadata for context --
        job_id = state.get("patient_id", "unknown")
        metadata_result = call_mcp_tool("lookup_volume_metadata", {"volume_id": job_id})
        tool_calls.append(metadata_result)

        volume_context = ""
        if metadata_result["status"] == "success":
            vol = metadata_result["result"]
            if "volume" in vol:
                slice_count = vol["volume"].get("slice_count", "N/A")
                volume_context = f"\nVolume Metadata (from MCP lookup_volume_metadata): {slice_count} slices cached for this analysis.\n"

        prompt = f"""You are a radiology expert assessing liver lesions for HCC risk (LI-RADS classification).

Patient: {state.get('patient_id', 'UNKNOWN')}
Imaging Modality: {state.get('imaging_modality', 'CT')}
Liver Volume: {seg['liver_volume']:.1f} mL
Tumor detection: weak (validation Dice 0.216); counts not reported
Total Tumor Burden: {seg['total_tumor_burden']:.2f}%
{model_context}{volume_context}
Based on typical imaging features (APHE, washout pattern, capsule enhancement):
1. Classify each lesion into LI-RADS category (LR-1 through LR-5, or LR-NC)
2. Provide reasoning for each classification
3. Determine the overall patient LI-RADS category

Return a JSON object with:
{{
    "lesions": [
        {{"id": 1, "size_mm": X, "lirads_category": "LR-X", "features": ["APHE", "washout", "capsule"]}}
    ],
    "primary_category": "LR-X",
    "confidence": 0.85,
    "clinical_reasoning": "..."
}}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        if '```json' in content:
            content = content.split('```json')[-1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)

        classification_data = None
        if json_match:
            try:
                classification_data = json.loads(json_match.group())
            except json.JSONDecodeError:
                classification_data = None

        if not classification_data:
            classification_data = {
                "lesions": [
                    {"id": i+1, "size_mm": size, "lirads_category": "LR-3", "features": []}
                    for i, size in enumerate(seg["lesion_sizes"])
                ],
                "primary_category": "LR-3",
                "confidence": 0.70,
                "clinical_reasoning": "Default classification due to parsing"
            }

        cls_result = {
            "lesions": classification_data["lesions"],
            "primary_category": classification_data["primary_category"],
            "imaging_criteria": {"APHE": True, "washout": True, "capsule": False},
            "confidence": classification_data.get("confidence", 0.7),
            "reasoning": classification_data.get("clinical_reasoning", "")
        }

        log_entry["status"] = "COMPLETED"
        log_entry["details"] = f"Primary category: {cls_result['primary_category']} (MCP: {len(tool_calls)} tool calls)"

        return {
            "classification_result": cls_result,
            "execution_log": [log_entry],
            "mcp_tool_calls": tool_calls
        }

    except Exception as e:
        log_entry["status"] = "ERROR"
        log_entry["details"] = str(e)
        return {"execution_log": [log_entry], "mcp_tool_calls": tool_calls}

# ============================================================================
# AGENT 3: CLINICAL EVIDENCE (uses MCP: get_model_metrics for evidence context)
# ============================================================================

def clinical_evidence_agent(state: AgentState) -> Dict[str, Any]:
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": "ClinicalEvidenceAgent",
        "status": "STARTED",
        "details": "Retrieving clinical guidelines with MCP tool augmentation"
    }
    tool_calls = []

    cls = state.get("classification_result")
    seg = state.get("segmentation_result")
    if not cls:
        log_entry["status"] = "ERROR"
        log_entry["details"] = "No classification result available"
        return {"execution_log": [log_entry], "mcp_tool_calls": []}

    try:
        # -- MCP Tool Call: Get model metrics for evidence-based confidence --
        metrics_result = call_mcp_tool("get_model_metrics")
        tool_calls.append(metrics_result)

        model_note = ""
        if metrics_result["status"] == "success":
            metrics = metrics_result["result"].get("metrics", {})
            model_note = f"""
Note from MCP get_model_metrics: The segmentation model achieves a global Dice score of
{metrics.get('global_dice_score', 'N/A')} with precision {metrics.get('precision', 'N/A')}
and recall {metrics.get('recall', 'N/A')}. Factor this into your confidence when
recommending management pathways.
"""

        prompt = f"""You are a hepatology expert. Given the LI-RADS assessment below, provide clinical guidance.

Patient: {state.get('patient_id', 'UNKNOWN')}
Primary LI-RADS Category: {cls['primary_category']}
Number of Lesions: {len(cls['lesions'])}
Total Tumor Burden: {seg['total_tumor_burden']:.2f}%
{model_note}
Based on current ACR LI-RADS and AASLD guidelines:
1. What is the appropriate management for this LI-RADS category?
2. What is the recommended follow-up imaging?
3. Should biopsy be considered?
4. Are there any contraindications to standard therapy?

Return a JSON object with:
{{
    "management_pathways": ["Surveillance with imaging", "Biopsy consideration"],
    "follow_up_recommendation": "...",
    "applicable_guidelines": ["ACR LI-RADS 2018", "AASLD 2023"],
    "supporting_evidence": [
        {{"source": "AASLD 2023", "recommendation": "..."}}
    ]
}}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        if '```json' in content:
            content = content.split('```json')[-1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)

        guidance_data = None
        if json_match:
            try:
                guidance_data = json.loads(json_match.group())
            except json.JSONDecodeError:
                guidance_data = None

        if not guidance_data:
            guidance_data = {
                "management_pathways": ["Imaging surveillance"],
                "follow_up_recommendation": "Follow ACR LI-RADS protocol",
                "applicable_guidelines": ["ACR LI-RADS 2018"],
                "supporting_evidence": []
            }

        guid_result = {
            "applicable_guidelines": guidance_data.get("applicable_guidelines", []),
            "management_pathways": guidance_data.get("management_pathways", []),
            "follow_up_recommendations": guidance_data.get("follow_up_recommendation", ""),
            "supporting_evidence": guidance_data.get("supporting_evidence", [])
        }

        log_entry["status"] = "COMPLETED"
        log_entry["details"] = f"Retrieved {len(guid_result['applicable_guidelines'])} guideline sources (MCP: {len(tool_calls)} tool call)"

        return {
            "guidance_result": guid_result,
            "execution_log": [log_entry],
            "mcp_tool_calls": tool_calls
        }

    except Exception as e:
        log_entry["status"] = "ERROR"
        log_entry["details"] = str(e)
        return {"execution_log": [log_entry], "mcp_tool_calls": tool_calls}

# ============================================================================
# AGENT 4: DIFFERENTIAL DIAGNOSIS (uses MCP: retrieve_slice for imaging context)
# ============================================================================

def differential_diagnosis_agent(state: AgentState) -> Dict[str, Any]:
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": "DifferentialDiagnosisAgent",
        "status": "STARTED",
        "details": "Generating differential diagnosis with MCP slice retrieval"
    }
    tool_calls = []

    cls = state.get("classification_result", {})
    seg = state.get("segmentation_result", {})

    try:
        # -- MCP Tool Call: Retrieve slice info to ground the diagnosis --
        job_id = state.get("patient_id", "unknown")
        slice_result = call_mcp_tool("retrieve_slice", {"job_id": job_id})
        tool_calls.append(slice_result)

        slice_context = ""
        if slice_result["status"] == "success":
            sr = slice_result["result"]
            total = sr.get("total_slices", "N/A")
            available = sr.get("available_indices", [])
            slice_context = f"""
Imaging Data (from MCP retrieve_slice):
- Total segmentation slices available: {total}
- Slice indices with segmentation overlay: {available[:10]}{'...' if len(available) > 10 else ''}
This confirms the segmentation was performed across the full volume.
"""

        prompt = f"""You are a liver imaging specialist. Generate a differential diagnosis for the following case:

LI-RADS Category: {cls.get('primary_category', 'Unknown')}
Tumor detection: weak; counts not reported
Liver Volume: {seg.get('liver_volume', 'Unknown')} mL
Imaging Features: APHE, washout, capsule enhancement
{slice_context}
Generate a ranked differential diagnosis including:
1. Hepatocellular carcinoma (HCC)
2. Cholangiocarcinoma
3. Hepatic adenoma
4. Focal nodular hyperplasia
5. Hemangioma
6. Metastatic disease

Return a JSON object:
{{
    "diagnoses": [
        {{"rank": 1, "diagnosis": "...", "probability": 0.75, "reasoning": "..."}}
    ],
    "most_likely": "HCC",
    "confidence": 0.80,
    "alternative_considerations": ["History of cirrhosis", "Hepatitis B/C status"]
}}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        if '```json' in content:
            content = content.split('```json')[-1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)

        diff_data = None
        if json_match:
            try:
                diff_data = json.loads(json_match.group())
            except json.JSONDecodeError:
                diff_data = None

        if not diff_data:
            diff_data = {
                "diagnoses": [{"rank": 1, "diagnosis": "HCC", "probability": 0.65, "reasoning": "LI-RADS category suggests HCC"}],
                "most_likely": "HCC",
                "confidence": 0.65,
                "alternative_considerations": []
            }

        diff_result = {
            "ranked_diagnoses": diff_data.get("diagnoses", []),
            "most_likely": diff_data.get("most_likely", "Unknown"),
            "confidence": diff_data.get("confidence", 0.5),
            "alternative_considerations": diff_data.get("alternative_considerations", [])
        }

        log_entry["status"] = "COMPLETED"
        log_entry["details"] = f"Most likely: {diff_result['most_likely']} (confidence {diff_result['confidence']:.2f}, MCP: {len(tool_calls)} tool call)"

        return {
            "differential_result": diff_result,
            "execution_log": [log_entry],
            "mcp_tool_calls": tool_calls
        }

    except Exception as e:
        log_entry["status"] = "ERROR"
        log_entry["details"] = str(e)
        return {"execution_log": [log_entry], "mcp_tool_calls": tool_calls}

# ============================================================================
# AGENT 5: REPORT GENERATOR (uses MCP: generate_report to cache, lookup_volume_metadata)
# ============================================================================

def report_generator_agent(state: AgentState) -> Dict[str, Any]:
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": "ReportGeneratorAgent",
        "status": "STARTED",
        "details": "Generating clinical report with MCP tool integration"
    }
    tool_calls = []

    seg = state.get("segmentation_result", {})
    cls = state.get("classification_result", {})
    diff = state.get("differential_result", {})
    guid = state.get("guidance_result", {})

    try:
        # -- MCP Tool Call 1: Check if a report already exists (cache check) --
        job_id = state.get("patient_id", "unknown")
        cached = call_mcp_tool("generate_report", {"job_id": job_id})
        tool_calls.append(cached)

        # If we got a full cached report with a radiologist_note, use it
        if cached["status"] == "success" and "radiologist_note" in cached["result"]:
            log_entry["status"] = "COMPLETED"
            log_entry["details"] = "Retrieved cached report from MCP generate_report"
            return {
                "final_report": cached["result"],
                "execution_log": [log_entry],
                "mcp_tool_calls": tool_calls
            }

        # -- MCP Tool Call 2: Get model metrics to include in report --
        metrics_result = call_mcp_tool("get_model_metrics")
        tool_calls.append(metrics_result)

        model_section = ""
        if metrics_result["status"] == "success":
            m = metrics_result["result"]
            metrics = m.get("metrics", {})
            model_section = f"""

MODEL PERFORMANCE (from MCP get_model_metrics):
Architecture: {m.get('architecture', 'N/A')}
Dataset: {m.get('dataset', 'N/A')}
Global Dice Score: {metrics.get('global_dice_score', 'N/A')}
Precision: {metrics.get('precision', 'N/A')}
Recall: {metrics.get('recall', 'N/A')}
"""

        prompt = f"""You are a radiology attending generating a formal clinical report. Create a structured radiology report based on:

Segmentation: liver volume {seg.get('liver_volume', 0):.1f} mL
LI-RADS Assessment: {cls.get('primary_category', 'Unknown')}
Most Likely Diagnosis: {diff.get('most_likely', 'Unknown')}
Management: {', '.join(guid.get('management_pathways', ['Undefined']))}
{model_section}
Format the report with sections:
1. FINDINGS
2. IMPRESSION/ASSESSMENT
3. RECOMMENDATIONS
4. FOLLOW-UP PLAN
5. AI MODEL DISCLOSURE (include the model performance metrics above)

Return plain text report without markdown."""

        response = llm.invoke([HumanMessage(content=prompt)])

        report = {
            "patient_id": state.get("patient_id", "UNKNOWN"),
            "exam_date": datetime.now().isoformat(),
            "liver_volume": seg.get("liver_volume", 0),
            "lesion_summary": "Tumor detection weak; counts not reported",
            "lirads_assessment": cls.get("primary_category", "LR-NC"),
            "differential_diagnosis": diff.get("most_likely", "Unknown"),
            "clinical_recommendations": ", ".join(guid.get("management_pathways", ["Undefined"])),
            "follow_up_plan": guid.get("follow_up_recommendations", "Follow standard protocol"),
            "radiologist_note": response.content,
            "generated_timestamp": datetime.now().isoformat()
        }

        log_entry["status"] = "COMPLETED"
        log_entry["details"] = f"Clinical report generated (MCP: {len(tool_calls)} tool calls)"

        return {
            "final_report": report,
            "execution_log": [log_entry],
            "mcp_tool_calls": tool_calls
        }

    except Exception as e:
        log_entry["status"] = "ERROR"
        log_entry["details"] = str(e)
        return {"execution_log": [log_entry], "mcp_tool_calls": tool_calls}

# ============================================================================
# LANGGRAPH ORCHESTRATION
# ============================================================================

def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("segmentation", segmentation_agent)
    graph.add_node("classification", lirads_classification_agent)
    graph.add_node("evidence", clinical_evidence_agent)
    graph.add_node("differential", differential_diagnosis_agent)
    graph.add_node("report", report_agent)

    graph.add_edge(START, "segmentation")
    graph.add_edge("segmentation", "report")
    graph.add_edge("report", END)

    return graph.compile()

# ============================================================================
# EXECUTION
# ============================================================================

async def run_liver_intelligence_pipeline(ct_volume: np.ndarray, patient_id: str = "UNKNOWN", voxel_mm3: float = 1.0) -> Dict[str, Any]:
    # Check MCP server availability
    mcp_available = check_mcp_health()

    initial_state: AgentState = {
        "ct_volume": ct_volume,
        "voxel_mm3": voxel_mm3,
        "patient_id": patient_id,
        "imaging_modality": "CT",
        "segmentation_result": None,
        "classification_result": None,
        "guidance_result": None,
        "differential_result": None,
        "final_report": None,
        "execution_log": [],
        "mcp_tool_calls": []
    }

    agent_graph = build_agent_graph()
    final_state = agent_graph.invoke(initial_state)

    results = {
        "patient_id": final_state.get("patient_id"),
        "segmentation": final_state.get("segmentation_result"),
        "classification": final_state.get("classification_result"),
        "guidance": final_state.get("guidance_result"),
        "differential": final_state.get("differential_result"),
        "report": final_state.get("final_report"),
        "execution_log": final_state.get("execution_log", []),
        "mcp_tool_calls": final_state.get("mcp_tool_calls", []),
        "mcp_server_available": mcp_available
    }

    # Cache the full pipeline result so MCP generate_report returns it next time
    try:
        cache_dir = Path(__file__).resolve().parent / "slice_cache" / patient_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "pipeline_result.json"
        # Make a serializable copy (exclude numpy arrays and non-serializable data)
        serializable = {}
        for k, v in results.items():
            if k == "mcp_tool_calls":
                serializable[k] = v
            elif isinstance(v, dict):
                serializable[k] = v
            elif isinstance(v, list):
                serializable[k] = v
            else:
                serializable[k] = v
        with open(cache_file, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
    except Exception:
        pass  # Caching is best-effort, don't break the pipeline

    # Post-cache MCP calls: each tool reads data that exists only after caching
    post_calls = []
    for tool, args in [
        ("lookup_volume_metadata", {"volume_id": patient_id}),
        ("retrieve_slice", {"job_id": patient_id}),
        ("generate_report", {"job_id": patient_id}),
        ("get_segmentation_stats", {"job_id": patient_id}),
    ]:
        try:
            r = requests.post(f"{MCP_SERVER_URL}/mcp/call",
                              json={"name": tool, "arguments": args}, timeout=10)
            body = r.json()
            ok = body.get("status") == "success" and "error" not in (body.get("result") or {})
        except Exception:
            ok = False
        post_calls.append({"tool_name": tool, "status": "success" if ok else "error"})
    results["mcp_tool_calls"] = list(results.get("mcp_tool_calls") or []) + post_calls

    return results

if __name__ == "__main__":
    mcp_ok = check_mcp_health()
    print("Liver Intelligence Agent System initialized")
    print(f"MCP Tool Server: {'CONNECTED' if mcp_ok else 'NOT AVAILABLE'} ({MCP_SERVER_URL})")
    print("Use run_liver_intelligence_pipeline() to execute the agent pipeline")
