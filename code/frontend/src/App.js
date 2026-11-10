import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import AnalysisPage from './components/AnalysisPage';
import ImagingPage from './components/ImagingPage';
import ResultsPage from './components/ResultsPage';
import MetricsPage from './components/MetricsPage';
import LoginPage from './components/LoginPage';
import { Activity, Upload, FileText, Home, BarChart3, LogOut, User } from 'lucide-react';
import './App.css';

function NavBar({ session, onLogout }) {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: <Home size={18} /> },
    { path: '/analyze', label: 'Analyze', icon: <Upload size={18} /> },
    { path: '/results', label: 'Results', icon: <FileText size={18} /> },
    { path: '/imaging', label: 'Imaging', icon: <Activity size={18} /> },
    { path: '/metrics', label: 'Metrics', icon: <BarChart3 size={18} /> },
  ];
  return (
    <nav className="navbar">
      <div className="nav-brand">
        <Activity size={24} className="brand-icon" />
        <span className="brand-text">Liver Lesion Intelligence</span>
      </div>
      <div className="nav-links">
        {navItems.map(item => (
          <Link
            key={item.path}
            to={item.path}
            className={`nav-link ${location.pathname === item.path ? 'active' : ''}`}
          >
            {item.icon}
            <span>{item.label}</span>
          </Link>
        ))}
      </div>
      {session && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginLeft: 'auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <User size={14} color="#94a3b8" />
            <span style={{ fontSize: 12, color: '#94a3b8' }}>{session.name}</span>
            <span style={{
              fontSize: 10, padding: '2px 8px', borderRadius: 6,
              background: 'rgba(99,102,241,0.15)', color: '#818cf8', fontWeight: 600
            }}>{session.role}</span>
          </div>
          <button
            onClick={onLogout}
            style={{
              display: 'flex', alignItems: 'center', gap: 5,
              padding: '6px 12px', background: 'rgba(239,68,68,0.1)',
              border: '1px solid rgba(239,68,68,0.2)', borderRadius: 8,
              color: '#f87171', fontSize: 12, cursor: 'pointer', fontWeight: 500
            }}
          >
            <LogOut size={13} /> Logout
          </button>
        </div>
      )}
    </nav>
  );
}

function App() {
  const [latestJobId, setLatestJobId] = useState(null);
  const [session, setSession] = useState(null);

  // Check for existing session on mount
  useEffect(function() {
    try {
      var saved = localStorage.getItem('liver_intel_session');
      if (saved) {
        setSession(JSON.parse(saved));
      }
    } catch (e) {
      // ignore
    }
  }, []);

  function handleLogin(sess) {
    setSession(sess);
  }

  function handleLogout() {
    localStorage.removeItem('liver_intel_session');
    setSession(null);
  }

  // Show login page if not authenticated
  if (!session) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return (
    <Router>
      <div className="app">
        <NavBar session={session} onLogout={handleLogout} />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/analyze" element={<AnalysisPage onJobCreated={setLatestJobId} />} />
            <Route path="/results" element={<ResultsPage jobId={latestJobId} />} />
            <Route path="/imaging/:jobId" element={<ImagingPage />} />
            <Route path="/imaging" element={<ImagingPage />} />
            <Route path="/metrics" element={<MetricsPage />} />
            <Route path="/results/:jobId" element={<ResultsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
