import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';

const SliceViewer = ({ jobId }) => {
  const [slices, setSlices] = useState([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [loading, setLoading] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [fullscreen, setFullscreen] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showOverlay, setShowOverlay] = useState(true);
  const playRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!jobId) return;
    axios.get(`/api/jobs/${jobId}/slices`)
      .then(res => { setSlices(res.data.slices || []); setLoading(false); })
      .catch(() => setLoading(false));
  }, [jobId]);

  // Video playback
  const togglePlay = useCallback(() => {
    if (playing) {
      clearInterval(playRef.current);
      setPlaying(false);
    } else {
      setPlaying(true);
      playRef.current = setInterval(() => {
        setCurrentIdx(i => {
          if (i >= slices.length - 1) { clearInterval(playRef.current); setPlaying(false); return 0; }
          return i + 1;
        });
      }, Math.max(50, 500 / speed));
    }
  }, [playing, slices.length, speed]);

  useEffect(() => { return () => clearInterval(playRef.current); }, []);
  useEffect(() => {
    if (playing) {
      clearInterval(playRef.current);
      playRef.current = setInterval(() => {
        setCurrentIdx(i => {
          if (i >= slices.length - 1) { clearInterval(playRef.current); setPlaying(false); return 0; }
          return i + 1;
        });
      }, Math.max(50, 500 / speed));
    }
  }, [speed, playing, slices.length]);

  const handleWheel = (e) => { e.preventDefault(); setZoom(z => Math.max(1, Math.min(5, z + (e.deltaY > 0 ? -0.2 : 0.2)))); };
  const handleMouseDown = (e) => { if (zoom > 1) { setDragging(true); setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y }); } };
  const handleMouseMove = (e) => { if (dragging) setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y }); };
  const handleMouseUp = () => setDragging(false);
  const resetView = () => { setZoom(1); setPan({ x: 0, y: 0 }); };

  const handleKeyDown = (e) => {
    if (e.key === 'ArrowLeft') setCurrentIdx(i => Math.max(0, i - 1));
    if (e.key === 'ArrowRight') setCurrentIdx(i => Math.min(slices.length - 1, i + 1));
    if (e.key === '+' || e.key === '=') setZoom(z => Math.min(5, z + 0.3));
    if (e.key === '-') setZoom(z => Math.max(1, z - 0.3));
    if (e.key === '0') resetView();
    if (e.key === 'f') setFullscreen(f => !f);
    if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  };

  if (loading) return <div style={s.loadingContainer}><div style={s.spinner}></div><p style={{color:'#94a3b8',marginTop:'1rem'}}>Loading CT volume...</p></div>;
  if (slices.length === 0) return null;

  const progress = ((currentIdx + 1) / slices.length * 100).toFixed(0);

  const outerStyle = fullscreen ? {
    ...s.container, position:'fixed', top:0, left:0, right:0, bottom:0,
    zIndex:9999, borderRadius:0, margin:0, maxWidth:'none', display:'flex', flexDirection:'column'
  } : s.container;

  return (
    <div style={outerStyle} tabIndex={0} onKeyDown={handleKeyDown} ref={containerRef}>
      {/* Header */}
      <div style={s.header}>
        <div style={{display:'flex',alignItems:'center',gap:'0.75rem'}}>
          <div style={s.pulse}></div>
          <h3 style={s.title}>CT Slice Visualization</h3>
        </div>
        <div style={{display:'flex',gap:'0.5rem',alignItems:'center'}}>
          <span style={s.badge}>{slices.length} slices</span>
          <button onClick={() => setShowOverlay(o => !o)} style={{...s.toolBtn, background: showOverlay ? '#065f46' : '#1e293b'}}>
            {showOverlay ? '● Overlay ON' : '○ Overlay OFF'}
          </button>
          <button onClick={() => setFullscreen(f => !f)} style={s.toolBtn}>
            {fullscreen ? '✕ Exit' : '⛶ Fullscreen'}
          </button>
        </div>
      </div>

      <div style={{...s.viewerGrid, flex: fullscreen ? 1 : 'none'}}>
        {/* Main viewer */}
        <div style={s.mainView}>
          {/* Zoom controls */}
          <div style={s.topBar}>
            <span style={s.viewLabel}>AXIAL VIEW</span>
            <div style={s.zoomControls}>
              <button onClick={() => setZoom(z => Math.min(5, z + 0.5))} style={s.zoomBtn}>+</button>
              <span style={s.zoomText}>{Math.round(zoom * 100)}%</span>
              <button onClick={() => setZoom(z => Math.max(1, z - 0.5))} style={s.zoomBtn}>−</button>
              <button onClick={resetView} style={s.zoomBtn}>↺</button>
            </div>
          </div>

          {/* Image canvas */}
          <div style={s.imageContainer}
            onWheel={handleWheel} onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
            <img
              src={`/api/jobs/${jobId}/slices/${slices[currentIdx]}`}
              alt={`Slice ${slices[currentIdx]}`}
              style={{
                ...s.mainImage,
                transform: `scale(${zoom}) translate(${pan.x/zoom}px, ${pan.y/zoom}px)`,
                cursor: zoom > 1 ? 'grab' : 'zoom-in',
                opacity: showOverlay ? 1 : 0.85,
                filter: showOverlay ? 'none' : 'grayscale(100%)'
              }}
              draggable={false}
            />

            {/* Floating metrics */}
            <div style={s.metricsFloat}>
              <div style={s.metricRow}><span style={s.metricLabel}>Slice</span><span style={s.metricVal}>{slices[currentIdx]}</span></div>
              <div style={s.metricRow}><span style={s.metricLabel}>Position</span><span style={s.metricVal}>{currentIdx + 1} of {slices.length} sampled</span></div>
              <div style={s.metricRow}><span style={s.metricLabel}>Progress</span><span style={s.metricVal}>{progress}%</span></div>
              <div style={s.metricRow}><span style={s.metricLabel}>Zoom</span><span style={s.metricVal}>{Math.round(zoom*100)}%</span></div>
            </div>
          </div>
        </div>

        {/* Thumbnails */}
        <div style={s.thumbPanel}>
          <div style={s.viewLabel}>SLICE NAVIGATOR</div>
          <div style={s.thumbGrid}>
            {slices.map((sl, idx) => (
              <div key={sl} onClick={() => { setCurrentIdx(idx); resetView(); }}
                style={{...s.thumbWrap, border: idx===currentIdx ? '2px solid #3b82f6' : '2px solid transparent',
                  boxShadow: idx===currentIdx ? '0 0 12px rgba(59,130,246,0.4)' : 'none'}}>
                <img src={`/api/jobs/${jobId}/slices/${sl}`} alt={`${sl}`} style={s.thumbImg} draggable={false}/>
                <span style={s.thumbLabel}>{sl}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Playback controls */}
      <div style={s.playbackBar}>
        <button onClick={togglePlay} style={s.playBtn}>{playing ? '⏸' : '▶'}</button>
        <button onClick={() => { setCurrentIdx(Math.max(0, currentIdx - 1)); resetView(); }} style={s.navBtn}>◀</button>

        <div style={s.progressWrap}>
          <div style={{...s.progressFill, width: `${progress}%`}}></div>
          <input type="range" min={0} max={slices.length - 1} value={currentIdx}
            onChange={e => { setCurrentIdx(Number(e.target.value)); resetView(); }}
            style={s.slider} />
        </div>

        <button onClick={() => { setCurrentIdx(Math.min(slices.length - 1, currentIdx + 1)); resetView(); }} style={s.navBtn}>▶</button>

        <div style={s.speedControl}>
          {[1,2,5].map(sp => (
            <button key={sp} onClick={() => setSpeed(sp)}
              style={{...s.speedBtn, ...(speed===sp ? {background:'#3b82f6',color:'#fff'} : {})}}>
              {sp}x
            </button>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div style={s.footer}>
        <div style={s.legend}>
          <div style={s.legendItem}><span style={{...s.dot, background:'#22c55e', boxShadow:'0 0 6px #22c55e'}}></span>Liver</div>
          <div style={s.legendItem}><span style={{...s.dot, background:'#ef4444', boxShadow:'0 0 6px #ef4444'}}></span>Tumor (weak)</div>
          <div style={s.legendItem}><span style={{...s.dot, background:'#64748b'}}></span>CT Background</div>
        </div>
        <div style={{color:'#475569',fontSize:'0.7rem'}}>
          Space: play/pause · Scroll: zoom · Drag: pan · ← →: navigate · F: fullscreen
        </div>
      </div>
    </div>
  );
};

const s = {
  container: { background:'linear-gradient(145deg, #0c1222 0%, #111827 50%, #0f172a 100%)', borderRadius:'16px', padding:'1.25rem', border:'1px solid rgba(59,130,246,0.15)', outline:'none', boxShadow:'0 8px 32px rgba(0,0,0,0.4)' },
  loadingContainer: { display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', padding:'4rem' },
  spinner: { width:'40px', height:'40px', border:'3px solid #1e293b', borderTop:'3px solid #3b82f6', borderRadius:'50%', animation:'spin 1s linear infinite' },
  header: { display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'1rem' },
  pulse: { width:'8px', height:'8px', borderRadius:'50%', background:'#22c55e', boxShadow:'0 0 8px #22c55e' },
  title: { color:'#e2e8f0', fontSize:'1.15rem', fontWeight:700, margin:0, letterSpacing:'-0.02em' },
  badge: { background:'rgba(59,130,246,0.15)', color:'#60a5fa', padding:'4px 12px', borderRadius:'20px', fontSize:'0.75rem', fontWeight:600, border:'1px solid rgba(59,130,246,0.2)' },
  toolBtn: { background:'#1e293b', color:'#cbd5e1', border:'1px solid #334155', borderRadius:'8px', padding:'5px 12px', cursor:'pointer', fontSize:'0.75rem', transition:'all 0.2s' },
  viewerGrid: { display:'grid', gridTemplateColumns:'3fr 1fr', gap:'0.75rem', marginBottom:'0.75rem' },
  mainView: { background:'radial-gradient(ellipse at center, #1a1a2e 0%, #0d0d1a 60%, #000 100%)', borderRadius:'12px', padding:'0.5rem', display:'flex', flexDirection:'column', position:'relative' },
  topBar: { display:'flex', justifyContent:'space-between', alignItems:'center', padding:'2px 8px' },
  viewLabel: { color:'#3b82f6', fontSize:'0.65rem', fontWeight:700, letterSpacing:'1.5px', padding:'4px 8px', textTransform:'uppercase' },
  zoomControls: { display:'flex', gap:'4px', alignItems:'center' },
  zoomBtn: { background:'rgba(30,41,59,0.8)', border:'1px solid #334155', color:'#e2e8f0', borderRadius:'6px', padding:'3px 8px', cursor:'pointer', fontSize:'0.8rem' },
  zoomText: { color:'#94a3b8', fontSize:'0.7rem', minWidth:'35px', textAlign:'center' },
  imageContainer: { overflow:'hidden', borderRadius:'8px', flex:1, display:'flex', alignItems:'center', justifyContent:'center', minHeight:'450px', background:'transparent', position:'relative' },
  mainImage: { maxWidth:'100%', maxHeight:'100%', display:'block', transformOrigin:'center center', transition:'transform 0.15s ease', userSelect:'none' },
  metricsFloat: { position:'absolute', top:'8px', right:'8px', background:'rgba(15,23,42,0.85)', backdropFilter:'blur(8px)', borderRadius:'8px', padding:'8px 12px', border:'1px solid rgba(59,130,246,0.2)' },
  metricRow: { display:'flex', justifyContent:'space-between', gap:'12px', padding:'2px 0' },
  metricLabel: { color:'#64748b', fontSize:'0.65rem', textTransform:'uppercase', letterSpacing:'0.5px' },
  metricVal: { color:'#e2e8f0', fontSize:'0.7rem', fontWeight:600, fontFamily:'monospace' },
  thumbPanel: { background:'rgba(30,41,59,0.5)', borderRadius:'12px', padding:'0.5rem', overflowY:'auto', maxHeight:'600px', backdropFilter:'blur(4px)' },
  thumbGrid: { display:'grid', gridTemplateColumns:'1fr 1fr', gap:'4px' },
  thumbWrap: { borderRadius:'6px', cursor:'pointer', overflow:'hidden', position:'relative', transition:'all 0.2s', background:'#000' },
  thumbImg: { width:'100%', display:'block', opacity:0.9 },
  thumbLabel: { position:'absolute', bottom:'2px', right:'4px', color:'#94a3b8', fontSize:'0.6rem', background:'rgba(0,0,0,0.7)', padding:'1px 4px', borderRadius:'3px', fontFamily:'monospace' },
  playbackBar: { display:'flex', alignItems:'center', gap:'0.5rem', padding:'0.5rem 0.25rem', background:'rgba(30,41,59,0.4)', borderRadius:'10px', marginBottom:'0.5rem' },
  playBtn: { background:'#3b82f6', border:'none', color:'#fff', borderRadius:'50%', width:'36px', height:'36px', cursor:'pointer', fontSize:'1rem', display:'flex', alignItems:'center', justifyContent:'center', boxShadow:'0 0 12px rgba(59,130,246,0.3)' },
  navBtn: { background:'#1e293b', border:'1px solid #334155', color:'#e2e8f0', borderRadius:'8px', padding:'6px 10px', cursor:'pointer', fontSize:'0.85rem' },
  progressWrap: { flex:1, position:'relative', height:'6px', background:'#1e293b', borderRadius:'3px', overflow:'visible' },
  progressFill: { position:'absolute', top:0, left:0, height:'100%', background:'linear-gradient(90deg, #3b82f6, #60a5fa)', borderRadius:'3px', transition:'width 0.1s ease', pointerEvents:'none' },
  slider: { position:'absolute', top:'-8px', left:0, width:'100%', height:'22px', opacity:0, cursor:'pointer', margin:0 },
  speedControl: { display:'flex', gap:'2px' },
  speedBtn: { background:'#1e293b', border:'1px solid #334155', color:'#94a3b8', borderRadius:'4px', padding:'4px 8px', cursor:'pointer', fontSize:'0.7rem', fontWeight:600 },
  footer: { display:'flex', justifyContent:'space-between', alignItems:'center', flexWrap:'wrap', gap:'0.5rem' },
  legend: { display:'flex', gap:'1.25rem', color:'#94a3b8', fontSize:'0.75rem' },
  legendItem: { display:'flex', alignItems:'center', gap:'6px' },
  dot: { width:'8px', height:'8px', borderRadius:'50%', display:'inline-block' }
};

export default SliceViewer;
