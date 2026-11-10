import React, { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, FileText, Play, AlertCircle, Loader } from 'lucide-react';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const PIPELINE_STEPS = [
  { key: 'SegmentationAgent', name: 'Segmentation Agent', desc: 'Running U-Net on CT volume' },
  { key: 'ReportAgent', name: 'Report Agent', desc: 'Summarizing results with real model metrics' },
];

function AnalysisPage({ onJobCreated }) {
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [progress, setProgress] = useState(0);
  const [currentAgent, setCurrentAgent] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();
  const pollingRef = useRef(null);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && (dropped.name.endsWith('.nii') || dropped.name.endsWith('.nii.gz'))) {
      setFile(dropped);
      setError(null);
    } else {
      setError('Please upload a NIfTI file (.nii or .nii.gz)');
    }
  }, []);

  const handleFileSelect = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setError(null);
    }
  };

  const pollJob = useCallback((id) => {
    pollingRef.current = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE}/api/jobs/${id}`);
        const data = res.data;
        setJobStatus(data.status);
        setProgress(data.progress || 0);
        setCurrentAgent(data.current_agent);

        if (data.status === 'completed') {
          clearInterval(pollingRef.current);
          if (onJobCreated) onJobCreated(id);
          setTimeout(() => navigate(`/results/${id}`), 1500);
        } else if (data.status === 'error') {
          clearInterval(pollingRef.current);
          setError(data.error || 'Analysis failed');
        }
      } catch (err) {
        clearInterval(pollingRef.current);
        setError('Lost connection to server');
      }
    }, 2000);
  }, [navigate, onJobCreated]);

  const handleSubmit = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post(`${API_BASE}/api/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const id = res.data.job_id;
      setJobId(id);
      setJobStatus('queued');
      pollJob(id);
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
      setUploading(false);
    }
  };

  const getStepStatus = (stepKey) => {
    if (!currentAgent) return 'pending';
    const currentIdx = PIPELINE_STEPS.findIndex(s => s.key === currentAgent);
    const stepIdx = PIPELINE_STEPS.findIndex(s => s.key === stepKey);
    if (jobStatus === 'completed') return 'completed';
    if (stepIdx < currentIdx) return 'completed';
    if (stepIdx === currentIdx) return 'active';
    return 'pending';
  };

  return (
    <div>
      <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8 }}>
        Analyze CT Volume
      </h1>
      <p style={{ color: 'var(--text-secondary)', fontSize: 15, marginBottom: 32 }}>
        Upload a NIfTI CT volume for autonomous liver lesion analysis
      </p>

      {!jobId && (
        <>
          {/* Upload Area */}
          <div
            className={`upload-area ${dragging ? 'dragging' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload size={48} className="upload-icon" />
            <div className="upload-text">
              {file ? file.name : 'Drop NIfTI file here or click to browse'}
            </div>
            <div className="upload-hint">
              Accepts .nii and .nii.gz files
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".nii,.nii.gz"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
          </div>

          {/* Actions */}
          <div style={{ display: 'flex', gap: 12, marginTop: 20 }}>
            <button
              className="btn btn-primary"
              onClick={handleSubmit}
              disabled={!file || uploading}
            >
              <Play size={16} />
              {uploading ? 'Submitting...' : 'Run Analysis'}
            </button>
            </div>
        </>
      )}

      {/* Error */}
      {error && (
        <div className="card" style={{ borderColor: 'var(--danger)', marginTop: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--danger)' }}>
            <AlertCircle size={20} />
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Pipeline Progress */}
      {jobId && (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="card-header">
            <Loader size={20} className="icon" style={jobStatus !== 'completed' ? { animation: 'spin 1s linear infinite' } : {}} />
            <h2>
              {jobStatus === 'completed' ? 'Analysis Complete' : `Analyzing ${file?.name || '...'}`}
            </h2>
          </div>

          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 16 }}>
            {progress}% — Job ID: {jobId}
          </div>

          <div className="pipeline">
            {PIPELINE_STEPS.map((step, idx) => {
              const stepStatus = getStepStatus(step.key);
              return (
                <div key={step.key} className={`pipeline-step ${stepStatus}`}>
                  <div className={`step-indicator ${stepStatus}`}>
                    {stepStatus === 'completed' ? '✓' : idx + 1}
                  </div>
                  <div className="step-info">
                    <div className="step-name">{step.name}</div>
                    <div className="step-detail">
                      {stepStatus === 'active' ? step.desc : stepStatus === 'completed' ? 'Done' : 'Waiting'}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default AnalysisPage;
