import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend } from 'recharts';
import { Target, Layers, AlertTriangle, CheckCircle } from 'lucide-react';

/* ─── Real results from the retrained 3D U-Net (models/retrained/best_retrained_3d_unet.keras) ───
   Input: CT volume resampled to 128 x 128 x 268, 3 classes (Background / Liver / Tumor)
   Validation: 15% of the 131 LiTS17 volumes, held out from training (best epoch 46 of 50)
   Eval run: evaluate_segmentation.py on volumes 17, 18, 19 */

var validationScores = [
  { name: 'Liver', dice: 0.7680, fill: '#00b4d8' },
  { name: 'Tumor', dice: 0.2164, fill: '#e76f51' }
];

var perVolume = [
  { vol: 'Vol 17', liver: 0.7680, tumor: 0.1328 },
  { vol: 'Vol 18', liver: 0.7846, tumor: 0.0000 },
  { vol: 'Vol 19', liver: 0.7972, tumor: 0.0671 }
];

var classMetrics = [
  { cls: 'Background', dice: 0.9961, iou: 0.9922, precision: 0.9974, recall: 0.9948 },
  { cls: 'Liver',      dice: 0.7851, iou: 0.6462, precision: 0.7429, recall: 0.8323 },
  { cls: 'Tumor',      dice: 0.0812, iou: 0.0423, precision: 0.0478, recall: 0.2690 }
];

var cardStyle = {
  background: 'rgba(255,255,255,0.04)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: '16px',
  padding: '24px',
  marginBottom: '24px'
};

var sectionTitle = {
  color: '#e76f51',
  fontSize: '13px',
  fontWeight: 700,
  letterSpacing: '1.5px',
  textTransform: 'uppercase',
  marginBottom: '16px',
  display: 'flex',
  alignItems: 'center',
  gap: '8px'
};

var labelStyle = {
  color: 'rgba(255,255,255,0.5)',
  fontSize: '12px',
  fontWeight: 600,
  letterSpacing: '1px',
  textTransform: 'uppercase'
};

var tooltipStyle = {
  background: 'rgba(0,0,0,0.85)',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: '8px',
  color: '#fff'
};

var thStyle = {
  textAlign: 'center',
  padding: '10px 12px',
  color: 'rgba(255,255,255,0.5)',
  fontSize: '11px',
  fontWeight: 600,
  letterSpacing: '1px',
  textTransform: 'uppercase'
};

var tdStyle = {
  padding: '12px',
  color: '#fff',
  textAlign: 'center',
  fontSize: '14px'
};

function StatCard(props) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.04)',
      border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: '16px',
      padding: '20px',
      flex: '1',
      minWidth: '180px',
      position: 'relative',
      overflow: 'hidden'
    }}>
      <div style={{ position: 'absolute', top: '12px', right: '16px', opacity: 0.1 }}>{props.icon}</div>
      <div style={labelStyle}>{props.label}</div>
      <div style={{ color: props.color, fontSize: '36px', fontWeight: 700, marginTop: '4px' }}>{props.value}</div>
      <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '11px' }}>{props.sub}</div>
    </div>
  );
}

function MetricsPage() {
  return (
    <div style={{ padding: '32px', maxWidth: '1200px', margin: '0 auto' }}>
      <h1 style={{ color: '#fff', fontSize: '28px', fontWeight: 700, marginBottom: '4px' }}>
        Model Evaluation Metrics
      </h1>
      <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: '14px', marginBottom: '32px' }}>
        3D U-Net Liver and Tumor Segmentation | LiTS17 Dataset | 131 CT volumes
      </p>

      {/* Headline numbers */}
      <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', marginBottom: '32px' }}>
        <StatCard label="Liver Dice (validation)" value="0.768" color="#00b4d8"
          sub="Held-out 15% of volumes, best epoch" icon={<Target size={48} />} />
        <StatCard label="Tumor Dice (validation)" value="0.216" color="#e76f51"
          sub="Detects tumors, but weakly" icon={<AlertTriangle size={48} />} />
        <StatCard label="Liver Dice (eval run)" value="0.785" color="#00d4aa"
          sub="Volumes 17, 18, 19 (see note below)" icon={<CheckCircle size={48} />} />
        <StatCard label="Tumor Dice (eval run)" value="0.081" color="#f4a261"
          sub="Same 3 volumes" icon={<Layers size={48} />} />
      </div>

      {/* Dice by class + per-class table */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        <div style={cardStyle}>
          <div style={sectionTitle}><Target size={16} />VALIDATION DICE</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={validationScores} layout="vertical" margin={{ left: 20, right: 30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis type="number" domain={[0, 1]} tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }} />
              <YAxis type="category" dataKey="name" tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 13 }} width={70} />
              <Tooltip contentStyle={tooltipStyle} formatter={function(v) { return v.toFixed(4); }} />
              <Bar dataKey="dice" radius={[0, 6, 6, 0]}>
                {validationScores.map(function(entry, i) {
                  return <Cell key={i} fill={entry.fill} />;
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={cardStyle}>
          <div style={sectionTitle}><Layers size={16} />PER-CLASS METRICS (EVAL RUN)</div>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                <th style={Object.assign({}, thStyle, { textAlign: 'left' })}>Class</th>
                <th style={thStyle}>Dice</th>
                <th style={thStyle}>IoU</th>
                <th style={thStyle}>Precision</th>
                <th style={thStyle}>Recall</th>
              </tr>
            </thead>
            <tbody>
              {classMetrics.map(function(row, i) {
                return (
                  <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    <td style={Object.assign({}, tdStyle, { textAlign: 'left', fontWeight: 600 })}>{row.cls}</td>
                    <td style={tdStyle}>{row.dice.toFixed(3)}</td>
                    <td style={tdStyle}>{row.iou.toFixed(3)}</td>
                    <td style={tdStyle}>{(row.precision * 100).toFixed(1)}%</td>
                    <td style={tdStyle}>{(row.recall * 100).toFixed(1)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Per-volume results */}
      <div style={cardStyle}>
        <div style={sectionTitle}><Target size={16} />PER-VOLUME DICE (EVAL RUN)</div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={perVolume} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="vol" tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 12 }} />
            <YAxis domain={[0, 1]} tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }} />
            <Tooltip contentStyle={tooltipStyle} formatter={function(v) { return v.toFixed(4); }} />
            <Legend wrapperStyle={{ color: 'rgba(255,255,255,0.6)', fontSize: '12px' }} />
            <Bar dataKey="liver" name="Liver" fill="#00b4d8" radius={[6, 6, 0, 0]} />
            <Bar dataKey="tumor" name="Tumor" fill="#e76f51" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Limitations */}
      <div style={cardStyle}>
        <div style={sectionTitle}><AlertTriangle size={16} />LIMITATIONS AND NEXT STEPS</div>
        <ul style={{ margin: 0, paddingLeft: '18px', color: 'rgba(255,255,255,0.6)', fontSize: '14px', lineHeight: 1.8 }}>
          <li>Tumors are under 1% of voxels. Downsampling the full scan to 128 x 128 x 268 shrinks small tumors to a few pixels or erases them.</li>
          <li>The U-Net has no skip connections, so fine detail is lost on the way back up. Adding them is the first planned fix.</li>
          <li>Next step: train on full-resolution 3D patches centred on tumor voxels, with a tumor-weighted loss, instead of whole downsampled volumes.</li>
          <li>Validation tumor Dice (0.216) is measured on held-out volumes. The eval-run numbers are on volumes that were part of training, so treat them as a sanity check rather than a true test score.</li>
        </ul>
      </div>

      {/* Model info */}
      <div style={cardStyle}>
        <div style={sectionTitle}><Layers size={16} />MODEL INFORMATION</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
          {[
            { label: 'ARCHITECTURE', value: '3D U-Net', sub: 'Encoder-decoder, 32 / 64 / 128 filters' },
            { label: 'INPUT', value: '128 x 128 x 268', sub: 'Single-channel CT, resampled' },
            { label: 'CLASSES', value: '3', sub: 'Background, liver, tumor' },
            { label: 'PARAMETERS', value: '554,243', sub: 'Softmax output per voxel' },
            { label: 'LOSS', value: 'Weighted CE + Dice', sub: 'Class weights 0.1 / 1.0 / 5.0' },
            { label: 'OPTIMIZER', value: 'Adam', sub: 'lr = 1e-4, ReduceLROnPlateau' },
            { label: 'TRAINING', value: '50 epochs', sub: 'Best at epoch 46, about 3 hours' },
            { label: 'GPU', value: 'RTX 5060 Ti', sub: '16 GB VRAM, CUDA 12.9' }
          ].map(function(item, i) {
            return (
              <div key={i} style={{
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: '10px',
                padding: '16px'
              }}>
                <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '10px', fontWeight: 600, letterSpacing: '1.5px', textTransform: 'uppercase' }}>
                  {item.label}
                </div>
                <div style={{ color: '#00d4aa', fontSize: '16px', fontWeight: 700, marginTop: '4px' }}>{item.value}</div>
                <div style={{ color: 'rgba(255,255,255,0.35)', fontSize: '11px', marginTop: '2px' }}>{item.sub}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default MetricsPage;