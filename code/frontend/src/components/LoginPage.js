import React, { useState } from 'react';
import { Lock, User, Activity, AlertCircle } from 'lucide-react';

/* --- Demo credentials --- */
var DEMO_USERS = {
  'admin': { password: 'admin123', name: 'Administrator', role: 'Admin' },
  'researcher': { password: 'research123', name: 'Researcher', role: 'Researcher' },
  'demo': { password: 'demo', name: 'Guest User', role: 'Viewer' }
};

function LoginPage({ onLogin }) {
  var stateUser = useState('');
  var username = stateUser[0];
  var setUsername = stateUser[1];

  var statePass = useState('');
  var password = statePass[0];
  var setPassword = statePass[1];

  var stateError = useState('');
  var error = stateError[0];
  var setError = stateError[1];

  var stateLoading = useState(false);
  var loading = stateLoading[0];
  var setLoading = stateLoading[1];

  function handleSubmit(e) {
    e.preventDefault();
    setError('');
    setLoading(true);

    // Simulate network delay
    setTimeout(function() {
      var user = DEMO_USERS[username.toLowerCase()];
      if (user && user.password === password) {
        var session = {
          username: username.toLowerCase(),
          name: user.name,
          role: user.role,
          loginTime: new Date().toISOString()
        };
        localStorage.setItem('liver_intel_session', JSON.stringify(session));
        onLogin(session);
      } else {
        setError('Invalid username or password');
      }
      setLoading(false);
    }, 800);
  }

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)',
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
      padding: 20
    }}>
      {/* Background glow effects */}
      <div style={{
        position: 'fixed', top: '20%', left: '15%', width: 300, height: 300,
        background: 'radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%)',
        borderRadius: '50%', filter: 'blur(60px)', pointerEvents: 'none'
      }} />
      <div style={{
        position: 'fixed', bottom: '20%', right: '15%', width: 250, height: 250,
        background: 'radial-gradient(circle, rgba(6,182,212,0.12) 0%, transparent 70%)',
        borderRadius: '50%', filter: 'blur(60px)', pointerEvents: 'none'
      }} />

      <div style={{
        width: '100%',
        maxWidth: 420,
        background: 'rgba(15, 23, 42, 0.7)',
        backdropFilter: 'blur(24px)',
        WebkitBackdropFilter: 'blur(24px)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: 24,
        padding: 40,
        boxShadow: '0 25px 50px rgba(0,0,0,0.4)'
      }}>
        {/* Logo / Header */}
        <div style={{ textAlign: 'center', marginBottom: 36 }}>
          <div style={{
            width: 64, height: 64, margin: '0 auto 16px',
            background: 'linear-gradient(135deg, #6366f1, #06b6d4)',
            borderRadius: 16,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 8px 24px rgba(99,102,241,0.3)'
          }}>
            <Activity size={32} color="#fff" />
          </div>
          <h1 style={{
            fontSize: 22, fontWeight: 700, color: '#f1f5f9',
            marginBottom: 6, letterSpacing: '-0.02em'
          }}>
            Liver Intelligence System
          </h1>
          <p style={{ fontSize: 13, color: '#64748b' }}>
            AI-Powered Liver Lesion Analysis Platform
          </p>
        </div>

        {/* Login Form */}
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 16 }}>
            <label style={{
              display: 'block', fontSize: 12, fontWeight: 600,
              color: '#94a3b8', marginBottom: 6, textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}>
              Username
            </label>
            <div style={{ position: 'relative' }}>
              <User size={16} color="#475569" style={{ position: 'absolute', left: 14, top: 13 }} />
              <input
                type="text"
                value={username}
                onChange={function(e) { setUsername(e.target.value); }}
                placeholder="Enter username"
                style={{
                  width: '100%', padding: '12px 14px 12px 40px',
                  background: 'rgba(255,255,255,0.04)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: 12, color: '#f1f5f9', fontSize: 14,
                  outline: 'none', boxSizing: 'border-box',
                  transition: 'border-color 0.2s'
                }}
                onFocus={function(e) { e.target.style.borderColor = 'rgba(99,102,241,0.5)'; }}
                onBlur={function(e) { e.target.style.borderColor = 'rgba(255,255,255,0.1)'; }}
              />
            </div>
          </div>

          <div style={{ marginBottom: 24 }}>
            <label style={{
              display: 'block', fontSize: 12, fontWeight: 600,
              color: '#94a3b8', marginBottom: 6, textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}>
              Password
            </label>
            <div style={{ position: 'relative' }}>
              <Lock size={16} color="#475569" style={{ position: 'absolute', left: 14, top: 13 }} />
              <input
                type="password"
                value={password}
                onChange={function(e) { setPassword(e.target.value); }}
                placeholder="Enter password"
                style={{
                  width: '100%', padding: '12px 14px 12px 40px',
                  background: 'rgba(255,255,255,0.04)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: 12, color: '#f1f5f9', fontSize: 14,
                  outline: 'none', boxSizing: 'border-box',
                  transition: 'border-color 0.2s'
                }}
                onFocus={function(e) { e.target.style.borderColor = 'rgba(99,102,241,0.5)'; }}
                onBlur={function(e) { e.target.style.borderColor = 'rgba(255,255,255,0.1)'; }}
              />
            </div>
          </div>

          {/* Error message */}
          {error && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '10px 14px', marginBottom: 16,
              background: 'rgba(239,68,68,0.08)',
              border: '1px solid rgba(239,68,68,0.2)',
              borderRadius: 10, fontSize: 13, color: '#f87171'
            }}>
              <AlertCircle size={15} />
              <span>{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !username || !password}
            style={{
              width: '100%', padding: '13px 0',
              background: loading ? 'rgba(99,102,241,0.4)' : 'linear-gradient(135deg, #6366f1, #4f46e5)',
              color: '#fff', border: 'none', borderRadius: 12,
              fontSize: 15, fontWeight: 600, cursor: loading ? 'wait' : 'pointer',
              boxShadow: '0 4px 16px rgba(99,102,241,0.3)',
              transition: 'all 0.2s',
              opacity: (!username || !password) ? 0.5 : 1
            }}
          >
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        {/* Guest Access Button */}
        <div style={{ textAlign: 'center', margin: '20px 0 0 0' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
            <div style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.08)' }} />
            <span style={{ fontSize: 11, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em' }}>or</span>
            <div style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.08)' }} />
          </div>
          <button
            onClick={function() {
              var session = {
                username: 'guest',
                name: 'Guest',
                role: 'Viewer',
                loginTime: new Date().toISOString()
              };
              localStorage.setItem('liver_intel_session', JSON.stringify(session));
              onLogin(session);
            }}
            style={{
              width: '100%', padding: '12px 0',
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 12, color: '#94a3b8', fontSize: 14,
              fontWeight: 500, cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseOver={function(e) { e.target.style.background = 'rgba(255,255,255,0.08)'; e.target.style.color = '#f1f5f9'; }}
            onMouseOut={function(e) { e.target.style.background = 'rgba(255,255,255,0.04)'; e.target.style.color = '#94a3b8'; }}
          >
            Continue as Guest
          </button>
        </div>

        {/* Demo credentials hint */}
        <div style={{
          marginTop: 28, padding: 16,
          background: 'rgba(99,102,241,0.06)',
          border: '1px solid rgba(99,102,241,0.12)',
          borderRadius: 12
        }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: '#818cf8', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Demo Credentials
          </div>
          <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.8, fontFamily: 'monospace' }}>
            <div><span style={{ color: '#cbd5e1' }}>admin</span> / admin123 — Admin</div>
            <div><span style={{ color: '#cbd5e1' }}>researcher</span> / research123 — Researcher</div>
            <div><span style={{ color: '#cbd5e1' }}>demo</span> / demo — Viewer</div>
          </div>
        </div>

        <div style={{
          marginTop: 20, padding: '10px 14px',
          background: 'rgba(245,158,11,0.06)',
          border: '1px solid rgba(245,158,11,0.15)',
          borderRadius: 10, textAlign: 'center'
        }}>
          <p style={{ fontSize: 11, color: '#f59e0b', fontWeight: 600, margin: '0 0 4px 0' }}>
            For Research &amp; Experimental Purposes Only
          </p>
          <p style={{ fontSize: 10, color: '#64748b', margin: 0 }}>
            Not intended for clinical decision-making
          </p>
        </div>

        <p style={{ textAlign: 'center', fontSize: 11, color: '#475569', marginTop: 16 }}>
          LiTS17 Liver Segmentation &middot; 3D U-Net + MCP Tools
        </p>
      </div>
    </div>
  );
}

export default LoginPage;
