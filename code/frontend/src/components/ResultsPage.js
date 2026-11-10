import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { FileText, Target, Stethoscope, ClipboardList, Clock, AlertTriangle, Activity, Download, Eye, Heart, Shield, TrendingUp, Wrench, Server, CheckCircle, XCircle } from 'lucide-react';
import axios from 'axios';

var API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
var DIAGNOSIS_COLORS = ['#6366f1','#06b6d4','#f59e0b','#ef4444','#8b5cf6','#ec4899'];

/* --- Shared Styles -------------------------------------------------- */

var glassCard = {
  background: 'rgba(15, 23, 42, 0.6)',
  backdropFilter: 'blur(16px)',
  WebkitBackdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 16,
  padding: 24,
  marginBottom: 20
};

var glassCardHeader = {
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  marginBottom: 18,
  paddingBottom: 14,
  borderBottom: '1px solid rgba(255,255,255,0.06)'
};

var headerTitle = {
  fontSize: 17,
  fontWeight: 600,
  color: '#f1f5f9',
  letterSpacing: '-0.01em'
};

function getLiRadsColor(cat) {
  if (!cat) return '#64748b';
  var upper = cat.toUpperCase();
  if (upper.includes('LR1') || upper.includes('LR-1')) return '#22c55e';
  if (upper.includes('LR2') || upper.includes('LR-2')) return '#4ade80';
  if (upper.includes('LR3') || upper.includes('LR-3')) return '#f59e0b';
  if (upper.includes('LR4') || upper.includes('LR-4')) return '#f97316';
  if (upper.includes('LR5') || upper.includes('LR-5')) return '#ef4444';
  if (upper.includes('LRM') || upper.includes('LR-M')) return '#dc2626';
  return '#f59e0b';
}

/* --- MCP Tool Call Log Component ------------------------------------ */

function MCPToolCallLog({ toolCalls, mcpAvailable }) {
  if (!toolCalls || toolCalls.length === 0) return null;

  var successCount = toolCalls.filter(function(t) { return t.status === 'success'; }).length;
  var errorCount = toolCalls.length - successCount;

  return (
    <div style={glassCard}>
      <div style={glassCardHeader}>
        <Wrench size={18} color="#a78bfa" />
        <span style={headerTitle}>MCP Tool Calls</span>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <Server size={13} color={mcpAvailable ? '#4ade80' : '#f87171'} />
            <span style={{ fontSize: 11, color: mcpAvailable ? '#4ade80' : '#f87171', fontWeight: 600 }}>
              {mcpAvailable ? 'MCP Online' : 'MCP Offline'}
            </span>
          </div>
          <span style={{ fontSize: 11, color: '#64748b' }}>
            {successCount}/{toolCalls.length} succeeded
          </span>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {toolCalls.map(function(call, idx) {
          var isSuccess = call.status === 'success';
          return (
            <div key={idx} style={{
              display: 'flex', alignItems: 'flex-start', gap: 12,
              padding: '12px 16px',
              background: isSuccess ? 'rgba(34,197,94,0.04)' : 'rgba(239,68,68,0.04)',
              border: '1px solid ' + (isSuccess ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)'),
              borderRadius: 10,
            }}>
              {isSuccess
                ? React.createElement(CheckCircle, { size: 15, color: '#4ade80', style: { marginTop: 2, flexShrink: 0 } })
                : React.createElement(XCircle, { size: 15, color: '#f87171', style: { marginTop: 2, flexShrink: 0 } })
              }
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  <span style={{
                    fontFamily: 'monospace', fontSize: 12, fontWeight: 600,
                    color: '#e2e8f0',
                    padding: '2px 8px', borderRadius: 6,
                    background: 'rgba(139,92,246,0.15)',
                  }}>
                    {call.tool_name}
                  </span>
                  <span style={{ fontSize: 11, color: '#475569', fontFamily: 'monospace' }}>
                    {call.timestamp ? new Date(call.timestamp).toLocaleTimeString() : ''}
                  </span>
                </div>
                {call.result && !call.result.error && (
                  <div style={{
                    fontSize: 11, color: '#94a3b8', lineHeight: 1.5,
                    maxHeight: 60, overflow: 'hidden', textOverflow: 'ellipsis',
                    fontFamily: 'monospace',
                  }}>
                    {JSON.stringify(call.result).substring(0, 200)}
                    {JSON.stringify(call.result).length > 200 ? '...' : ''}
                  </div>
                )}
                {call.result && call.result.error && (
                  <div style={{ fontSize: 11, color: '#f87171' }}>
                    {call.result.error}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* --- Main ResultsPage ----------------------------------------------- */

function ResultsPage(props) {
  var propJobId = props.jobId;
  var params = useParams();
  var paramJobId = params.jobId;
  var jobId = paramJobId || propJobId;

  var stateResults = useState(null);
  var results = stateResults[0];
  var setResults = stateResults[1];

  var stateLoading = useState(true);
  var loading = stateLoading[0];
  var setLoading = stateLoading[1];

  var stateError = useState(null);
  var error = stateError[0];
  var setError = stateError[1];

  useEffect(function() {
    if (!jobId) { setLoading(false); return; }
    axios.get(API_BASE + '/api/jobs/' + jobId)
      .then(function(res) {
        if (res.data.status === 'completed' && res.data.results) {
          setResults(res.data.results);
        } else {
          setError('Job status: ' + res.data.status);
        }
        setLoading(false);
      })
      .catch(function() { setError('Failed to load results'); setLoading(false); });
  }, [jobId]);

  if (!jobId) {
    return React.createElement('div', { style: { textAlign: 'center', padding: 64 } },
      React.createElement(Activity, { size: 48, style: { color: '#475569', marginBottom: 16 } }),
      React.createElement('h1', { style: { fontSize: 24, fontWeight: 700, marginBottom: 8, color: '#f1f5f9' } }, 'No Analysis Selected'),
      React.createElement('p', { style: { color: '#64748b' } }, 'Submit a CT volume to view results.')
    );
  }
  if (loading) {
    return React.createElement('div', { style: { textAlign: 'center', padding: 64 } },
      React.createElement('div', { style: { width: 40, height: 40, border: '3px solid #334155', borderTopColor: '#6366f1', borderRadius: '50%', animation: 'spin 0.8s linear infinite', margin: '0 auto 16px' } }),
      React.createElement('p', { style: { color: '#64748b' } }, 'Loading results...'),
      React.createElement('style', null, '@keyframes spin { to { transform: rotate(360deg); } }')
    );
  }

  if (error) {
    return React.createElement('div', { style: Object.assign({}, glassCard, { borderColor: 'rgba(239,68,68,0.3)' }) },
      React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 10, color: '#f87171' } },
        React.createElement(AlertTriangle, { size: 20 }),
        React.createElement('span', { style: { fontWeight: 500 } }, error)
      )
    );
  }

  var seg = (results && results.segmentation) || {};
  var cls = (results && results.classification) || {};
  var diff = (results && results.differential) || {};
  var report = (results && results.report) || {};
  var guid = (results && results.guidance) || {};
  var log = (results && results.execution_log) || [];
  var mcpCalls = (results && results.mcp_tool_calls) || [];
  var mcpAvailable = results && results.mcp_server_available;

  var allLesions = (seg.lesion_sizes || []).map(function(size, idx) {
    return { name: 'L' + (idx + 1), size: parseFloat(size.toFixed(1)) };
  });
  var lesionData = allLesions.length > 30
    ? allLesions.filter(function(_, i) { return i % Math.ceil(allLesions.length / 20) === 0; })
    : allLesions;

  var diagnosisData = (diff.ranked_diagnoses || []).slice(0, 5).map(function(d) {
    return {
      name: (d.diagnosis) || 'Unknown',
      probability: Math.round((d.probability || 0) * 100)
    };
  });

  var liRadsColor = getLiRadsColor(cls.primary_category);

  var statCards = [
    { label: 'Tumor Detection', value: 'Weak', icon: Target, color: '#f4a261' },
    { label: 'Liver Volume', value: (seg.liver_volume || 0).toFixed(1) + ' mL', icon: Heart, color: '#06b6d4' },
    { label: 'Liver Dice (validation)', value: results.report && results.report.validation_liver_dice != null ? Number(results.report.validation_liver_dice).toFixed(3) : 'n/a', icon: Shield, color: '#818cf8' },
    { label: 'Tumor Burden', value: 'Not reliable', icon: TrendingUp, color: '#94a3b8' }
  ];

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 28, flexWrap: 'wrap', gap: 16 }}>
        <div>
          <h1 style={{ fontSize: 26, fontWeight: 700, color: '#f1f5f9', marginBottom: 4 }}>Analysis Results</h1>
          <p style={{ color: '#64748b', fontSize: 13, fontFamily: 'monospace' }}>ID: {jobId}</p>
        </div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <a href={API_BASE + '/api/jobs/' + jobId + '/report/pdf'} download style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '9px 18px', background: 'linear-gradient(135deg, #dc2626, #b91c1c)', color: '#fff', borderRadius: 10, textDecoration: 'none', fontSize: '0.85rem', fontWeight: 600, boxShadow: '0 2px 8px rgba(220,38,38,0.3)' }}>
            <Download size={15} /> PDF Report
          </a>
          <Link to={'/imaging/' + jobId} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '9px 18px', background: 'linear-gradient(135deg, #6366f1, #4f46e5)', color: '#fff', borderRadius: 10, textDecoration: 'none', fontSize: '0.85rem', fontWeight: 600, boxShadow: '0 2px 8px rgba(99,102,241,0.3)' }}>
            <Eye size={15} /> CT Imaging
          </Link>
        </div>
      </div>

      {/* Stat Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14, marginBottom: 20 }}>
        {statCards.map(function(s, i) {
          var IconComp = s.icon;
          return (
            <div key={i} style={{
              background: 'rgba(15,23,42,0.6)',
              backdropFilter: 'blur(16px)',
              WebkitBackdropFilter: 'blur(16px)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: 16,
              padding: 20,
              position: 'relative',
              overflow: 'hidden'
            }}>
              <div style={{ position: 'absolute', top: 14, right: 14, opacity: 0.15 }}>
                <IconComp size={36} color={s.color} />
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>{s.label}</div>
              <div style={{ fontSize: 26, fontWeight: 700, color: s.color }}>{s.value}</div>
            </div>
          );
        })}
      </div>

      {/* Charts Row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
        {lesionData.length > 0 && (
          <div style={glassCard}>
            <div style={glassCardHeader}>
              <Target size={18} color="#6366f1" />
              <span style={headerTitle}>Lesion Sizes (mm)</span>
              {allLesions.length > 30 && <span style={{ fontSize: 11, color: '#64748b', marginLeft: 'auto' }}>Showing {lesionData.length} of {allLesions.length}</span>}
            </div>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={lesionData} margin={{ bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="name" stroke="#475569" fontSize={10} interval={lesionData.length > 15 ? Math.ceil(lesionData.length / 8) : 0} angle={-35} textAnchor="end" height={40} />
                <YAxis stroke="#475569" fontSize={11} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 13 }} labelStyle={{ color: '#f1f5f9' }} />
                <Bar dataKey="size" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {diagnosisData.length > 0 && (
          <div style={glassCard}>
            <div style={glassCardHeader}>
              <Stethoscope size={18} color="#06b6d4" />
              <span style={headerTitle}>Differential Diagnosis</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
              <div style={{ flex: '0 0 170px' }}>
                <ResponsiveContainer width={170} height={170}>
                  <PieChart>
                    <Pie data={diagnosisData} dataKey="probability" nameKey="name" cx="50%" cy="50%" innerRadius={45} outerRadius={75} paddingAngle={3} label={false}>
                      {diagnosisData.map(function(_, idx) {
                        return <Cell key={idx} fill={DIAGNOSIS_COLORS[idx % DIAGNOSIS_COLORS.length]} />;
                      })}
                    </Pie>
                    <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 13 }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
                {diagnosisData.map(function(d, idx) {
                  return (
                    <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div style={{ width: 10, height: 10, borderRadius: 3, flexShrink: 0, background: DIAGNOSIS_COLORS[idx % DIAGNOSIS_COLORS.length] }} />
                      <span style={{ fontSize: 13, color: '#cbd5e1', flex: 1, lineHeight: 1.3 }}>{d.name}</span>
                      <span style={{ fontSize: 13, fontWeight: 600, color: '#f1f5f9', flexShrink: 0 }}>{d.probability}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Clinical Guidance */}
      {guid.management_pathways && (
        <div style={glassCard}>
          <div style={glassCardHeader}>
            <ClipboardList size={18} color="#22c55e" />
            <span style={headerTitle}>Clinical Guidance</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 20 }}>
            <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 12, padding: 18, border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 10 }}>Management</div>
              <div style={{ fontSize: 14, color: '#e2e8f0', lineHeight: 1.6 }}>{guid.management_pathways.join(', ')}</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 12, padding: 18, border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 10 }}>Follow-up</div>
              <div style={{ fontSize: 14, color: '#e2e8f0', lineHeight: 1.6 }}>{guid.follow_up_recommendations || 'N/A'}</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 12, padding: 18, border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 10 }}>Guidelines</div>
              <div style={{ fontSize: 14, color: '#e2e8f0', lineHeight: 1.6 }}>{(guid.applicable_guidelines || []).join(', ')}</div>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 12, padding: 18, border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 10 }}>Most Likely Diagnosis</div>
              <div style={{ fontSize: 14, color: '#e2e8f0', lineHeight: 1.6 }}>{diff.most_likely || 'N/A'} ({((diff.confidence || 0) * 100).toFixed(0)}%)</div>
            </div>
          </div>
        </div>
      )}

      {/* Radiology Report */}
      {report.radiologist_note && (
        <div style={glassCard}>
          <div style={glassCardHeader}>
            <FileText size={18} color="#f59e0b" />
            <span style={headerTitle}>Radiology Report</span>
          </div>
          <div style={{ fontSize: 14, color: '#cbd5e1', lineHeight: 1.8, whiteSpace: 'pre-wrap', background: 'rgba(255,255,255,0.02)', borderRadius: 10, padding: 20 }}>
            {report.radiologist_note}
          </div>
        </div>
      )}

      {/* MCP Tool Call Log -- NEW */}
      <MCPToolCallLog toolCalls={mcpCalls} mcpAvailable={mcpAvailable} />
      {results.report && (
        <div style={{ marginTop: 24, display: 'flex', flexDirection: 'column', gap: 14 }}>
          {[
            { title: 'Findings', items: results.report.findings || [] },
            { title: 'Model performance', items: results.report.model_performance || [] },
            { title: 'Limitations', items: results.report.limitations || [] },
          ].map(sec => (
            <div key={sec.title} style={{ padding: 20, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 14 }}>
              <div style={{ fontWeight: 600, marginBottom: 8, color: '#e2e8f0', fontSize: 14 }}>{sec.title}</div>
              <ul style={{ margin: 0, paddingLeft: 18, color: '#cbd5e1', fontSize: 13, lineHeight: 1.7 }}>
                {sec.items.map((t, i) => <li key={i}>{t}</li>)}
              </ul>
            </div>
          ))}
          <div style={{ padding: 20, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 14, color: '#cbd5e1', fontSize: 13, lineHeight: 1.7 }}>
            <div style={{ fontWeight: 600, marginBottom: 6, color: '#e2e8f0', fontSize: 14 }}>Recommendation</div>
            <div>{results.report.recommendation || results.report.statement}</div>
          </div>
        </div>
      )}
      <div style={{ marginTop: 32, paddingTop: 16, borderTop: '1px solid rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.45)', fontSize: 12, lineHeight: 1.6 }}>
        Built by Sohel Ahmed, PhD. Research prototype: liver segmentation is reasonable; tumor detection is weak. Not for clinical use.
      </div>

      {/* Execution Log */}
      {log.length > 0 && (
        <div style={glassCard}>
          <div style={glassCardHeader}>
            <Clock size={18} color="#94a3b8" />
            <span style={headerTitle}>Pipeline Execution Log</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {log.map(function(entry, idx) {
              return (
                <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '8px 12px', background: 'rgba(255,255,255,0.02)', borderRadius: 8, fontSize: 13 }}>
                  <span style={{ color: '#64748b', fontFamily: 'monospace', fontSize: 12, flexShrink: 0 }}>
                    {new Date(entry.timestamp).toLocaleTimeString()}
                  </span>
                  <span style={{ padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600, background: 'rgba(99,102,241,0.15)', color: '#818cf8', flexShrink: 0 }}>
                    {entry.agent}
                  </span>
                  <span style={{ padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600, background: entry.status === 'COMPLETED' ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)', color: entry.status === 'COMPLETED' ? '#4ade80' : '#f87171', flexShrink: 0 }}>
                    {entry.status}
                  </span>
                  <span style={{ color: '#94a3b8', fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {entry.details}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default ResultsPage;
