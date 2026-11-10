import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Activity, Server, Cpu, Brain, Upload, CheckCircle } from 'lucide-react';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const AGENTS = [
  { name: 'Segmentation Agent', desc: '3D U-Net (v1) liver and tumor segmentation on CT volumes' },
  { name: 'Report Agent', desc: 'Builds the research report from real results and MCP model metrics' },
];

function Dashboard() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get(`${API_BASE}/api/status`)
      .then(res => { setStatus(res.data); setLoading(false); })
      .catch(() => { setStatus(null); setLoading(false); });
  }, []);

  return (
    <div>
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8 }}>
          Autonomous Liver Lesion Intelligence
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: 15 }}>
          Deep Learning + Multi-Agent Reasoning for Liver Imaging Analysis
        </p>
      </div>

      {/* Status Cards */}
      <div className="stats-grid">
        <div className="stat-card">
          <span className="stat-label">System Status</span>
          <span className={`stat-value ${status ? 'success' : 'danger'}`}>
            {loading ? '...' : status ? 'Online' : 'Offline'}
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Model</span>
          <span className="stat-value accent" style={{ fontSize: 18 }}>
            U-Net LiTS17
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Active Agents</span>
          <span className="stat-value accent">{AGENTS.length}</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Input Shape</span>
          <span className="stat-value" style={{ fontSize: 18 }}>
            128 × 128 × 268
          </span>
        </div>
      </div>

      {/* Agent Pipeline */}
      <div className="card">
        <div className="card-header">
          <Brain size={20} className="icon" />
          <h2>Agent Pipeline</h2>
        </div>
        <div className="pipeline">
          {AGENTS.map((agent, idx) => (
            <div key={idx} className="pipeline-step">
              <div className="step-indicator completed">
                {idx + 1}
              </div>
              <div className="step-info">
                <div className="step-name">{agent.name}</div>
                <div className="step-detail">{agent.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card">
        <div className="card-header">
          <Activity size={20} className="icon" />
          <h2>Quick Actions</h2>
        </div>
        <div style={{ display: 'flex', gap: 12 }}>
          <Link to="/analyze" className="btn btn-primary">
            <Upload size={16} />
            Upload CT Volume
          </Link>
          <Link to="/results" className="btn btn-secondary">
            <CheckCircle size={16} />
            View Results
          </Link>
        </div>
      </div>

      {/* System Info */}
      <div className="card">
        <div className="card-header">
          <Server size={20} className="icon" />
          <h2>System Configuration</h2>
        </div>
        <div className="result-grid">
          <div className="result-item">
            <div className="label">Framework</div>
            <div className="value">TensorFlow 2.21.0</div>
          </div>
          <div className="result-item">
            <div className="label">Orchestration</div>
            <div className="value">LangGraph 1.2.20</div>
          </div>
          <div className="result-item">
            <div className="label">LLM</div>
            <div className="value">None (template report)</div>
          </div>
          <div className="result-item">
            <div className="label">GPU</div>
            <div className="value">RTX 5060 Ti 16GB</div>
          </div>
          <div className="result-item">
            <div className="label">Dataset</div>
            <div className="value">LiTS17</div>
          </div>
          <div className="result-item">
            <div className="label">Dice Score</div>
            <div className="value">0.8131</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
