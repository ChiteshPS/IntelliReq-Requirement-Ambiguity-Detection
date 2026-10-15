// frontend/src/components/Overview.tsx — Academic & Power BI Analytics Dashboard
import React from 'react';
import {
  BarChart3,
  Award,
  ShieldCheck,
  Cpu,
  Layers,
  FileText,
  Info,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Sliders,
} from 'lucide-react';
import type { ModelInfoResponse } from '../types';

interface OverviewProps {
  onNavigate: (tab: 'overview' | 'single' | 'batch' | 'eval' | 'guide') => void;
  modelInfo: ModelInfoResponse | null;
}

export const Overview: React.FC<OverviewProps> = ({ onNavigate, modelInfo }) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* Section A: Page Introduction & Fast Navigation Bar                 */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      <div className="bi-card" style={{ marginBottom: 0 }}>
        <div className="bi-card-body" style={{ padding: '24px 28px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '20px' }}>
            <div style={{ flex: '1 1 600px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                <span className="badge-status clear">
                  <CheckCircle2 size={14} />
                  Verified Baseline Production Model
                </span>
                <span className="badge-status badge-neutral">
                  Dataset: <code>{modelInfo?.dataset_provenance?.dataset_name || 'intellireq_clean_v4'}</code> (715 Samples)
                </span>
                <span className="badge-status badge-info">
                  Features: {modelInfo?.total_features || 408} Dimensions
                </span>
              </div>

              <h2 style={{ fontSize: '26px', color: 'var(--text-heading)', fontWeight: 700, letterSpacing: '-0.3px', marginBottom: '8px' }}>
                IntelliReq — Requirement Quality Analysis
              </h2>
              <p style={{ fontSize: '15px', color: 'var(--text-secondary)', lineHeight: 1.6, maxWidth: '880px', margin: 0 }}>
                Research dashboard for software requirement ambiguity detection and model evaluation. 
                Combines 384-dimensional dense semantic sentence embeddings (<code>{modelInfo?.embedding_model || 'all-MiniLM-L6-v2'}</code>) 
                with {modelInfo?.handcrafted_feature_count || 24} domain-engineered linguistic indicators, calibrated under a 3-tier operational decision policy.
              </p>
            </div>

            {/* Quick Action Shortcuts */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', alignSelf: 'center' }}>
              <button
                className="btn btn-primary"
                onClick={() => onNavigate('single')}
                id="overview-quick-analyze-btn"
                style={{ fontSize: '14.5px', padding: '10px 18px' }}
              >
                <FileText size={16} />
                Single Requirement Analysis
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => onNavigate('batch')}
                id="overview-quick-batch-btn"
                style={{ fontSize: '14.5px', padding: '10px 18px' }}
              >
                <Layers size={16} />
                Batch Specification Grid
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* Section B: Key Performance Indicators (Aligned Multi-Card Grid)    */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <h3 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-heading)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Award size={20} style={{ color: 'var(--accent-primary)' }} />
            Core Empirical Performance Metrics
          </h3>
          <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Audited Against Independent Held-Out Splits
          </span>
        </div>

        <div className="kpi-grid">
          {/* KPI 1: In-Domain Test Accuracy */}
          <div className="kpi-tile success">
            <div className="kpi-tile-header">
              <span className="kpi-tile-label">Held-Out Test Accuracy</span>
              <Award size={20} className="kpi-tile-icon" style={{ color: '#16A34A' }} />
            </div>
            <div className="kpi-tile-value" style={{ color: '#15803D' }}>98.15%</div>
            <div className="kpi-tile-subtext">
              <span><strong>106 / 108</strong> correct predictions</span>
            </div>
            <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '8px', lineHeight: 1.4 }}>
              Evaluated on the in-domain held-out test split (108 samples) with group-aware stratification.
            </div>
          </div>

          {/* KPI 2: Selective Decision Accuracy */}
          <div className="kpi-tile primary">
            <div className="kpi-tile-header">
              <span className="kpi-tile-label">Selective Accuracy</span>
              <ShieldCheck size={20} className="kpi-tile-icon" style={{ color: 'var(--accent-primary)' }} />
            </div>
            <div className="kpi-tile-value" style={{ color: 'var(--accent-primary)' }}>100.00%</div>
            <div className="kpi-tile-subtext">
              <span><strong>103 / 103</strong> confident predictions</span>
            </div>
            <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '8px', lineHeight: 1.4 }}>
              Zero classification errors on automated decisions after routing uncertain cases to review.
            </div>
          </div>

          {/* KPI 3: Decision Coverage */}
          <div className="kpi-tile primary">
            <div className="kpi-tile-header">
              <span className="kpi-tile-label">Decision Coverage</span>
              <Sliders size={20} className="kpi-tile-icon" style={{ color: 'var(--accent-primary)' }} />
            </div>
            <div className="kpi-tile-value" style={{ color: 'var(--accent-primary)' }}>95.37%</div>
            <div className="kpi-tile-subtext">
              <span><strong>103 / 108</strong> automated decisions</span>
            </div>
            <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '8px', lineHeight: 1.4 }}>
              Only 4.63% (5 samples) deferred to human analyst review under the calibrated 3-tier policy.
            </div>
          </div>

          {/* KPI 4: Challenge-Set Accuracy */}
          <div className="kpi-tile warning">
            <div className="kpi-tile-header">
              <span className="kpi-tile-label">Challenge Generalization</span>
              <AlertTriangle size={20} className="kpi-tile-icon" style={{ color: '#D97706' }} />
            </div>
            <div className="kpi-tile-value" style={{ color: '#B45309' }}>85.00%</div>
            <div className="kpi-tile-subtext">
              <span><strong>68 / 80</strong> raw accuracy (Selective: 92.54%)</span>
            </div>
            <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '8px', lineHeight: 1.4 }}>
              Evaluated on 80 unseen trap specifications designed to test subtle semantic boundaries.
            </div>
          </div>

          {/* KPI 5: Ambiguous Requirement Recall */}
          <div className="kpi-tile purple">
            <div className="kpi-tile-header">
              <span className="kpi-tile-label">Ambiguity Recall</span>
              <Cpu size={20} className="kpi-tile-icon" style={{ color: '#9333EA' }} />
            </div>
            <div className="kpi-tile-value" style={{ color: '#6B21A8' }}>100.00%</div>
            <div className="kpi-tile-subtext">
              <span><strong>31 / 31</strong> ambiguous cases flagged</span>
            </div>
            <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '8px', lineHeight: 1.4 }}>
              Zero false negatives on the test set; no ambiguous requirement passed silently as clear.
            </div>
          </div>
        </div>
      </div>

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* Section C: Empirical Benchmark Comparison (Full-Width Primary Table) */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      <div className="bi-card">
        <div className="bi-card-header">
          <div>
            <div className="bi-card-title">
              <BarChart3 size={18} style={{ color: 'var(--accent-primary)' }} />
              Empirical Benchmark Evaluation Across Multiple Corpora
            </div>
            <div className="bi-card-subtitle">
              Comprehensive comparison distinguishing in-domain held-out test performance from external benchmark distributions.
            </div>
          </div>
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => onNavigate('eval')}
          >
            <span>View Full Evaluation Viva Report</span>
            <ArrowRight size={14} />
          </button>
        </div>

        <div className="bi-card-body" style={{ padding: 0 }}>
          <div className="bi-table-container" style={{ border: 'none', borderRadius: 0 }}>
            <table className="bi-table">
              <thead>
                <tr>
                  <th style={{ width: '22%' }}>Evaluation Corpus / Suite</th>
                  <th style={{ width: '18%' }}>Domain Classification</th>
                  <th style={{ width: '10%' }}>Samples (N)</th>
                  <th style={{ width: '12%' }}>Accuracy</th>
                  <th style={{ width: '12%' }}>Macro-F1</th>
                  <th style={{ width: '26%' }}>Empirical Interpretation</th>
                </tr>
              </thead>
              <tbody>
                {/* 1. Internal Held-Out */}
                <tr style={{ backgroundColor: '#F0FDF4' }}>
                  <td>
                    <strong style={{ color: '#15803D' }}>IntelliReq Clean v4 Test Set</strong>
                  </td>
                  <td>
                    <span className="badge-status clear">In-Domain Held-Out</span>
                  </td>
                  <td className="mono" style={{ fontWeight: 600 }}>108</td>
                  <td className="mono" style={{ fontWeight: 700, color: '#15803D' }}>98.15%</td>
                  <td className="mono" style={{ fontWeight: 700, color: '#15803D' }}>98.05%</td>
                  <td style={{ fontSize: '13px', color: '#166534', lineHeight: 1.45 }}>
                    Primary in-domain test set. Zero false negatives (100% recall). Selective accuracy reaches 100.00% with 4.63% deferral rate.
                  </td>
                </tr>

                {/* 2. Challenge Set */}
                <tr>
                  <td>
                    <strong style={{ color: 'var(--text-heading)' }}>Diagnostic Challenge Benchmark</strong>
                  </td>
                  <td>
                    <span className="badge-status review">Out-of-Domain Traps</span>
                  </td>
                  <td className="mono" style={{ fontWeight: 600 }}>80</td>
                  <td className="mono" style={{ fontWeight: 600, color: '#B45309' }}>85.00%</td>
                  <td className="mono" style={{ fontWeight: 600 }}>84.85%</td>
                  <td style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                    Adversarial boundary cases. Selective accuracy reaches <strong>92.54%</strong> on confident predictions (16.25% deferral rate).
                  </td>
                </tr>

                {/* 3. Adversarial Traps */}
                <tr>
                  <td>
                    <strong style={{ color: 'var(--text-heading)' }}>Adversarial Stress Test</strong>
                  </td>
                  <td>
                    <span className="badge-status review">Syntactic Distractors</span>
                  </td>
                  <td className="mono" style={{ fontWeight: 600 }}>16</td>
                  <td className="mono" style={{ fontWeight: 600 }}>87.50%</td>
                  <td className="mono" style={{ fontWeight: 600 }}>86.67%</td>
                  <td style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                    Requirements containing vague keywords but qualified by exact measurable numeric criteria.
                  </td>
                </tr>

                {/* 4. PURE IEEE */}
                <tr>
                  <td>
                    <strong style={{ color: 'var(--text-heading)' }}>PURE Requirements Dataset</strong>
                  </td>
                  <td>
                    <span className="badge-status skipped">External IEEE SRS</span>
                  </td>
                  <td className="mono" style={{ fontWeight: 600 }}>6,725</td>
                  <td className="mono">79.53%</td>
                  <td className="mono">79.40%</td>
                  <td style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                    Natural language requirements from published IEEE software projects. Selective accuracy reaches <strong>93.08%</strong>.
                  </td>
                </tr>

                {/* 5. Agile User Stories */}
                <tr>
                  <td>
                    <strong style={{ color: 'var(--text-heading)' }}>Agile User Stories Corpus</strong>
                  </td>
                  <td>
                    <span className="badge-status skipped">External User Stories</span>
                  </td>
                  <td className="mono" style={{ fontWeight: 600 }}>1,680</td>
                  <td className="mono">75.30%</td>
                  <td className="mono">74.88%</td>
                  <td style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                    Template-based Agile stories (<em>"As a user..."</em>) exhibiting syntactic domain shift. Selective accuracy is <strong>87.23%</strong>.
                  </td>
                </tr>

                {/* 6. Fault-Prone SRS */}
                <tr>
                  <td>
                    <strong style={{ color: 'var(--text-heading)' }}>Fault-Prone SRS Dataset (Kaggle)</strong>
                  </td>
                  <td>
                    <span className="badge-status skipped">External Defect Corpus</span>
                  </td>
                  <td className="mono" style={{ fontWeight: 600 }}>7,061</td>
                  <td className="mono">70.12%</td>
                  <td className="mono">69.85%</td>
                  <td style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                    Kaggle defect dataset conflating syntax errors with ambiguity. Serves as lower-bound cross-dataset baseline.
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <div style={{ padding: '14px 24px', backgroundColor: 'var(--bg-subtle)', borderTop: '1px solid var(--border-subtle)', fontSize: '13px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Info size={16} style={{ color: 'var(--accent-primary)', flexShrink: 0 }} />
            <span>
              <strong>Academic Audit Notice:</strong> 98.15% accuracy is strictly an in-domain held-out test result (108 samples). Across diverse external datasets, generalization ranges between 70.12% and 85.00% due to legitimate domain shifts.
            </span>
          </div>
        </div>
      </div>

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* Section D: Three-Tier Decision Policy (Detailed Separate Section)  */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      <div className="bi-card">
        <div className="bi-card-header">
          <div>
            <div className="bi-card-title">
              <ShieldCheck size={18} style={{ color: '#16A34A' }} />
              Operational Three-Tier Decision Policy & Selective Accuracy
            </div>
            <div className="bi-card-subtitle">
              Calibrated probability routing ensures that safety-critical engineering backlogs receive only high-confidence classifications.
            </div>
          </div>
          <span className="badge-status badge-info">Platt Sigmoid Calibrated</span>
        </div>

        <div className="bi-card-body">
          <p style={{ fontSize: '14.5px', color: 'var(--text-primary)', lineHeight: 1.6, marginBottom: '20px' }}>
            Rather than forcing a hard binary threshold at 50% probability, IntelliReq utilizes two calibrated thresholds (<strong>0.25</strong> and <strong>0.75</strong>) 
            to define an uncertainty guardrail. Borderline specifications are routed to human requirements engineers for inspection, achieving 
            <strong> 100.00% selective accuracy</strong> on in-domain test data and <strong>92.54%</strong> on challenge benchmarks.
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
            {/* Category 1: Clear */}
            <div style={{ background: '#F0FDF4', border: '1px solid #BBF7D0', padding: '18px 20px', borderRadius: 'var(--radius-sm)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
                <span className="badge-status clear" style={{ fontSize: '13px', fontWeight: 700 }}>
                  NOT AMBIGUOUS
                </span>
                <span className="mono" style={{ fontSize: '13px', fontWeight: 700, color: '#15803D' }}>
                  P &le; 25.0%
                </span>
              </div>
              <div style={{ fontSize: '14px', fontWeight: 600, color: '#166534', marginBottom: '6px' }}>
                High-Confidence Verifiable Specification
              </div>
              <p style={{ fontSize: '13px', color: '#14532D', lineHeight: 1.5, margin: 0 }}>
                Requirement possesses explicit, quantifiable acceptance criteria (e.g. latency &lt; 200ms, 99.9% uptime) and is devoid of subjective qualitative adjectives. Ready for technical sprint implementation.
              </p>
            </div>

            {/* Category 2: Needs Review */}
            <div style={{ background: '#FFFBEB', border: '1px solid #FDE68A', padding: '18px 20px', borderRadius: 'var(--radius-sm)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
                <span className="badge-status review" style={{ fontSize: '13px', fontWeight: 700 }}>
                  NEEDS REVIEW
                </span>
                <span className="mono" style={{ fontSize: '13px', fontWeight: 700, color: '#B45309' }}>
                  25.0% &lt; P &lt; 75.0%
                </span>
              </div>
              <div style={{ fontSize: '14px', fontWeight: 600, color: '#92400E', marginBottom: '6px' }}>
                Uncertainty Guardrail (Deferred to Human)
              </div>
              <p style={{ fontSize: '13px', color: '#78350F', lineHeight: 1.5, margin: 0 }}>
                Requirement exhibits contradictory or mixed signals (e.g. contains a numeric target alongside passive voice or weak modals). Deferred to human analyst review to eliminate false positives and false negatives.
              </p>
            </div>

            {/* Category 3: Ambiguous */}
            <div style={{ background: '#FEF2F2', border: '1px solid #FECACA', padding: '18px 20px', borderRadius: 'var(--radius-sm)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
                <span className="badge-status ambiguous" style={{ fontSize: '13px', fontWeight: 700 }}>
                  AMBIGUOUS
                </span>
                <span className="mono" style={{ fontSize: '13px', fontWeight: 700, color: '#B91C1C' }}>
                  P &ge; 75.0%
                </span>
              </div>
              <div style={{ fontSize: '14px', fontWeight: 600, color: '#991B1B', marginBottom: '6px' }}>
                High-Confidence Requirement Defect
              </div>
              <p style={{ fontSize: '13px', color: '#7F1D1D', lineHeight: 1.5, margin: 0 }}>
                Contains subjective adjectives (<em>"fast"</em>, <em>"intuitive"</em>), vague terms (<em>"appropriate"</em>), or untestable specifications. Triggers automated diagnostic suggestions and rewrite assistance.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* Section E: Production Model Architecture & Implementation Specs    */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      <div className="bi-card">
        <div className="bi-card-header">
          <div>
            <div className="bi-card-title">
              <Cpu size={18} style={{ color: 'var(--accent-primary)' }} />
              Production Model Architecture & Implementation Methodology
            </div>
            <div className="bi-card-subtitle">
              Verified pipeline components running in <code>saved_models/model.joblib</code> and <code>saved_models/scaler.joblib</code>.
            </div>
          </div>
          <span className="badge-status badge-neutral">Offline Capable • &lt; 20ms Latency</span>
        </div>

        <div className="bi-card-body">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '18px' }}>
            {/* Component 1 */}
            <div style={{ border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '16px', background: '#FFFFFF' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-primary)', fontWeight: 600, fontSize: '14.5px', marginBottom: '6px' }}>
                <Layers size={16} /> Channel 1: SentenceTransformer
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
                Pretrained <code>all-MiniLM-L6-v2</code> produces <strong>384-dimensional dense semantic embeddings</strong>, capturing deep contextual syntax and sentence meaning without internet dependency.
              </p>
            </div>

            {/* Component 2 */}
            <div style={{ border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '16px', background: '#FFFFFF' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#16A34A', fontWeight: 600, fontSize: '14.5px', marginBottom: '6px' }}>
                <FileText size={16} /> Channel 2: 24 Handcrafted Features
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
                Deterministic regex extractors detect lexical vagueness, subjective adjectives, weak quantifiers, modal verbs, and quantifiable metric indicators (time, size, throughput).
              </p>
            </div>

            {/* Component 3 */}
            <div style={{ border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '16px', background: '#FFFFFF' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#9333EA', fontWeight: 600, fontSize: '14.5px', marginBottom: '6px' }}>
                <Sliders size={16} /> StandardScaler Normalization
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
                Standardizes the concatenated <strong>408-dimensional hybrid vector</strong> to zero mean and unit variance, ensuring consistent gradient scaling across heterogeneous feature types.
              </p>
            </div>

            {/* Component 4 */}
            <div style={{ border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '16px', background: '#FFFFFF' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#D97706', fontWeight: 600, fontSize: '14.5px', marginBottom: '6px' }}>
                <Cpu size={16} /> Calibrated L1 Logistic Regression
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
                L1 regularization (Lasso) performs automatic feature selection. Platt Sigmoid Calibration (<code>CalibratedClassifierCV</code>) produces empirical posterior probabilities.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
