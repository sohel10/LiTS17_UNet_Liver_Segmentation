# LiTS17 Liver and Tumor CT Segmentation

A research prototype that segments the liver and liver tumors in abdominal CT scans, using a 3D U-Net, a small multi-agent pipeline, and a React dashboard. Built on the public LiTS17 dataset.

> **Research prototype. Not for clinical use.** Liver segmentation is reasonable. Tumor detection is weak (validation tumor Dice 0.216), so tumor results should be treated as reference only.

## What it does

1. Upload a CT volume in NIfTI format (`.nii` or `.nii.gz`).
2. A 3D U-Net predicts liver and tumor masks.
3. A two-agent pipeline measures the liver volume, estimates tumor burden, and writes a structured research report.
4. The dashboard shows the results, the MCP tool calls, and overlay images for each slice.

## Architecture

```
React frontend (port 8000, served by the backend)
        |
        v
FastAPI backend: main_agent_api.py (port 8000)
        |  upload -> job -> background pipeline
        v
LangGraph pipeline: agent_orchestrator.py
   Agent 1: Segmentation Agent   -> 3D U-Net (predict_v1.py)
   Agent 2: Report Agent         -> builds report, calls MCP
        |
        v
MCP Tool Server: mcp_tool_server.py (port 8001)
   get_model_metrics, lookup_volume_metadata,
   retrieve_slice, generate_report, get_segmentation_stats
```

### Preprocessing and model

- CT values are clipped to the soft-tissue window of -200 to 250 HU.
- Each volume is scaled to 0-1 with min-max normalization.
- The volume is resized to 128 x 128 x 268 for the network.
- The network is a 3D U-Net with three output classes: background, liver, and tumor.
- Training used a weighted cross-entropy loss plus a Dice loss, with class weights of 0.1, 1.0, and 5.0.
- The predicted masks are resized back to the original scan grid before measuring.

### Agent 1: Segmentation Agent

- Runs the 3D U-Net and returns the liver and tumor masks.
- Measures the liver volume in millilitres, using the voxel spacing from the scan header.
- Finds tumor regions with 3D connected components and keeps only regions larger than 50 voxels.
- Computes the tumor burden as a percentage of the liver volume.

### Agent 2: Report Agent

- Calls the MCP tool `get_model_metrics` to get the validation scores from the last training run.
- Builds a structured report from the measured numbers. It has no language model and no LI-RADS scoring. The report contains:
  - **Findings:** liver volume, predicted tumor burden, and a tumor detection warning.
  - **Model performance:** liver and tumor validation Dice.
  - **Limitations:** public data only, possible false positives and misses, one training run.
  - **Recommendation:** research and demonstration use only.

### MCP tool calls

After the result is cached, the pipeline calls four more MCP tools. Each one returns real data:

| Tool | Called by | Purpose |
|---|---|---|
| `get_model_metrics` | Report Agent | Validation Dice from the training run |
| `lookup_volume_metadata` | Pipeline | Volume and cached-analysis lookup |
| `retrieve_slice` | Pipeline | Number of overlay slices for the job |
| `generate_report` | Pipeline | Returns the cached full result |
| `get_segmentation_stats` | Pipeline | Liver volume and tumor burden for the job |

The Results page lists each call with its status. A green check means the server returned a successful result.

## Results

Validation Dice, from 15% of the training data held out during training:

| Metric | Score |
|---|---|
| Liver Dice | 0.768 |
| Tumor Dice | 0.216 |

Overlay example (liver in green, tumor in red):

![Model prediction compared with ground truth](docs/compare_slice51.png)

## Project structure

```
LITS17/
  .gitignore
  requirements.txt
  README.md
  code/
    main_agent_api.py       FastAPI backend and API routes
    agent_orchestrator.py   LangGraph pipeline and both agents
    mcp_tool_server.py      MCP tool server (port 8001)
    predict_v1.py           Standalone inference for the v1 model
    train_3d_unet.py        Training script for the v1 model
    evaluate_segmentation.py  Evaluation metrics
    LiTS17_UNet_portfolio.ipynb  Notebook
    docs/                   Images for this README
    frontend/               React dashboard
```

Data and trained models are not in this repository. They are excluded in `.gitignore`.

## How to run

You need Python with TensorFlow, nibabel, FastAPI, and LangGraph, plus Node.js for the frontend.

1. Start the MCP tool server on port 8001.
2. Start the backend on port 8000.
3. Build the frontend once, then open `http://localhost:8000` in a browser.
4. Upload a NIfTI volume on the Analyze page, then open the Results page.

Use port 8000 for everything. The frontend is served by the backend, so there is no separate development server to start.

## Limitations

- Trained and tested only on the public LiTS17 dataset.
- Tumor detection is weak. The model misses many tumors and produces false positives.
- Metrics come from one training run and one validation split. Some evaluation volumes may have been used in training.
- The report is generated from measurements and fixed text. It is not a radiologist's report.
- Not validated for clinical use.

## Credits

Built by **Sohel Ahmed, PhD**.
