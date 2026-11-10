import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import SliceViewer from './SliceViewer';
import axios from 'axios';

const ImagingPage = () => {
  const { jobId } = useParams();
  const [activeJob, setActiveJob] = useState(jobId || null);
  const [loading, setLoading] = useState(!jobId);

  useEffect(() => {
    if (!jobId) {
      axios.get('/api/slice-jobs').then(res => {
        const jobs = res.data.jobs || [];
        if (jobs.length > 0) setActiveJob(jobs[0]);
        setLoading(false);
      }).catch(() => setLoading(false));
    }
  }, [jobId]);

  if (loading) return <div style={{color:'#94a3b8',padding:'3rem',textAlign:'center'}}>Loading...</div>;

  if (!activeJob) return (
    <div style={{color:'#94a3b8',padding:'3rem',textAlign:'center'}}>
      <h3 style={{color:'#e2e8f0'}}>CT Imaging Viewer</h3>
      <p>No scans available. Run an analysis from the Analyze tab first.</p>
    </div>
  );

  return (
    <div style={{padding:'1.5rem',maxWidth:'1100px',margin:'0 auto'}}>
      <SliceViewer jobId={activeJob} />
    </div>
  );
};

export default ImagingPage;
