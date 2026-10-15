// frontend/src/components/SystemGuide.tsx
import React from 'react';
import {
  Layers,
  FileCheck2,
  Lock,
  GitBranch,
  AlertTriangle,
  UploadCloud,
  Search,
  Database,
} from 'lucide-react';

export const SystemGuide: React.FC = () => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Overview & System Workflow Banner */}
      <div className="bi-card">
        <div className="bi-card-header">
          <div className="bi-card-title">
            <Layers size={18} style={{ color: 'var(--color-primary)' }} />
            System Architecture & End-to-End Pipeline
          </div>
          <span className="badge-status badge-neutral">Production Version 4.0</span>
        </div>
        <div className="bi-card-body">
          <p style={{ fontSize: '14px', color: 'var(--color-text-primary)', lineHeight: 1.6, marginBottom: '20px' }}>
            IntelliReq employs a hybrid dual-channel Natural Language Processing architecture that combines 
            dense contextual sentence embeddings with 24 domain-engineered requirements engineering rules. 
            Calibrated probabilities feed into an operational 3-tier decision policy designed for high precision in software engineering workflows.
          </p>

          {/* Workflow Stepper */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px', marginBottom: '20px' }}>
            <div style={{ background: '#F8FAFC', border: '1px solid var(--color-border)', borderRadius: '6px', padding: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', fontWeight: 700, color: 'var(--color-primary)', textTransform: 'uppercase' }}>
                <span style={{ width: '18px', height: '18px', borderRadius: '50%', background: 'var(--color-primary)', color: '#fff', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '10px' }}>1</span>
                Input Ingestion
              </div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-heading)', marginTop: '6px' }}>Requirement Text</div>
              <div style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginTop: '4px' }}>Normalized text stripped of noisy whitespace and encoding anomalies.</div>
            </div>

            <div style={{ background: '#F8FAFC', border: '1px solid var(--color-border)', borderRadius: '6px', padding: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', fontWeight: 700, color: 'var(--color-primary)', textTransform: 'uppercase' }}>
                <span style={{ width: '18px', height: '18px', borderRadius: '50%', background: 'var(--color-primary)', color: '#fff', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '10px' }}>2</span>
                Dual Extraction
              </div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-heading)', marginTop: '6px' }}>384-d + 24 Features</div>
              <div style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginTop: '4px' }}>MiniLM dense embeddings + regex-based lexical & specificity indicators.</div>
            </div>

            <div style={{ background: '#F8FAFC', border: '1px solid var(--color-border)', borderRadius: '6px', padding: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', fontWeight: 700, color: 'var(--color-primary)', textTransform: 'uppercase' }}>
                <span style={{ width: '18px', height: '18px', borderRadius: '50%', background: 'var(--color-primary)', color: '#fff', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '10px' }}>3</span>
                Feature Scaling
              </div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-heading)', marginTop: '6px' }}>StandardScaler</div>
              <div style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginTop: '4px' }}>Zero-mean, unit-variance normalization over the 408-dimensional vector.</div>
            </div>

            <div style={{ background: '#F8FAFC', border: '1px solid var(--color-border)', borderRadius: '6px', padding: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', fontWeight: 700, color: 'var(--color-primary)', textTransform: 'uppercase' }}>
                <span style={{ width: '18px', height: '18px', borderRadius: '50%', background: 'var(--color-primary)', color: '#fff', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '10px' }}>4</span>
                Inference & Platt
              </div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-heading)', marginTop: '6px' }}>L1 Logistic Reg.</div>
              <div style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginTop: '4px' }}>Calibrated probability output calibrated with Platt sigmoid scaling.</div>
            </div>

            <div style={{ background: '#F8FAFC', border: '1px solid var(--color-border)', borderRadius: '6px', padding: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', fontWeight: 700, color: 'var(--color-primary)', textTransform: 'uppercase' }}>
                <span style={{ width: '18px', height: '18px', borderRadius: '50%', background: 'var(--color-primary)', color: '#fff', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '10px' }}>5</span>
                Decision Guardrails
              </div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-heading)', marginTop: '6px' }}>3-Tier Policy</div>
              <div style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginTop: '4px' }}>Categorizes into Clear, Needs Review, or Ambiguous with rewrite suggestions.</div>
            </div>
          </div>

          {/* Channel Detail Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
            <div style={{ border: '1px solid var(--color-border)', borderRadius: '6px', padding: '16px', background: '#FFFFFF' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--color-primary)', fontWeight: 700, fontSize: '14px', marginBottom: '8px' }}>
                <Layers size={16} /> Channel 1: Dense Semantic Embeddings (384-d)
              </div>
              <p style={{ fontSize: '13px', color: 'var(--color-text-secondary)', lineHeight: 1.5, margin: 0 }}>
                Generated using local weights of <code>sentence-transformers/all-MiniLM-L6-v2</code> (bundled in <code>saved_models/all-MiniLM-L6-v2</code>). 
                Captures semantic syntax, contextual relatedness, and sentence structure without requiring an internet connection.
              </p>
            </div>

            <div style={{ border: '1px solid var(--color-border)', borderRadius: '6px', padding: '16px', background: '#FFFFFF' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#16A34A', fontWeight: 700, fontSize: '14px', marginBottom: '8px' }}>
                <FileCheck2 size={16} /> Channel 2: 24 Handcrafted Features
              </div>
              <p style={{ fontSize: '13px', color: 'var(--color-text-secondary)', lineHeight: 1.5, margin: 0 }}>
                Extracted via <code>FeatureExtractor</code> using verified lexicons across 4 pillars: Lexical Vagueness (e.g. <em>efficient</em>, <em>user-friendly</em>), 
                Syntactic Structure (passive voice, conjunctions), Semantic Quantifiers (weak bounds), and Metric Specificity (numerical constraints, time units).
              </p>
            </div>

            <div style={{ border: '1px solid var(--color-border)', borderRadius: '6px', padding: '16px', background: '#FFFFFF' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#D97706', fontWeight: 700, fontSize: '14px', marginBottom: '8px' }}>
                <Lock size={16} /> Regularization & Platt Calibration
              </div>
              <p style={{ fontSize: '13px', color: 'var(--color-text-secondary)', lineHeight: 1.5, margin: 0 }}>
                L1 regularization (Lasso) performs automatic feature selection over the 408 inputs, pruning redundant dimensions. 
                Platt Sigmoid Calibration via <code>CalibratedClassifierCV</code> ensures predicted probabilities correspond to empirical probabilities.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Three-Tier Operational Decision Policy */}
      <div className="bi-card">
        <div className="bi-card-header">
          <div className="bi-card-title">
            <GitBranch size={18} style={{ color: 'var(--color-primary)' }} />
            Three-Tier Operational Decision Policy & Selective Accuracy
          </div>
          <span className="badge-status badge-info">Confidence Guardrails</span>
        </div>
        <div className="bi-card-body">
          <p style={{ fontSize: '13.5px', color: 'var(--color-text-primary)', lineHeight: 1.6, marginBottom: '16px' }}>
            In software requirement analysis, false negatives (missed ambiguities that enter development) and false positives 
            (needlessly blocking clear specifications) carry significant costs. IntelliReq utilizes calibrated uncertainty bands 
            to isolate borderline requirements for human review, ensuring high selective accuracy on automated decisions.
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '14px' }}>
            <div style={{ background: '#F0FDF4', border: '1px solid #BBF7D0', padding: '16px', borderRadius: '6px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span className="badge-status badge-success" style={{ fontWeight: 700 }}>NOT AMBIGUOUS</span>
                <span style={{ fontSize: '12px', fontWeight: 700, color: '#15803D' }}>P &le; 25.0%</span>
              </div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: '#166534', marginBottom: '6px' }}>High-Confidence Specificity</div>
              <p style={{ fontSize: '12px', color: '#14532D', lineHeight: 1.5, margin: 0 }}>
                Requirement contains quantifiable targets (response times, capacities, throughput), verifiable constraints, 
                and no subjective qualitative adjectives. Safe to proceed to engineering backlog.
              </p>
            </div>

            <div style={{ background: '#FFFBEB', border: '1px solid #FDE68A', padding: '16px', borderRadius: '6px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span className="badge-status badge-warning" style={{ fontWeight: 700 }}>NEEDS REVIEW</span>
                <span style={{ fontSize: '12px', fontWeight: 700, color: '#B45309' }}>25.0% &lt; P &lt; 75.0%</span>
              </div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: '#92400E', marginBottom: '6px' }}>Uncertainty Guardrail Band</div>
              <p style={{ fontSize: '12px', color: '#78350F', lineHeight: 1.5, margin: 0 }}>
                Requirement exhibits mixed or conflicting indicators (e.g. contains numerical criteria but also passive voice or weak modals). 
                Deferred to business analyst review. This selective filtering delivers <strong>100%</strong> in-domain and <strong>92.54%</strong> challenge selective accuracy.
              </p>
            </div>

            <div style={{ background: '#FEF2F2', border: '1px solid #FECACA', padding: '16px', borderRadius: '6px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span className="badge-status badge-error" style={{ fontWeight: 700 }}>AMBIGUOUS</span>
                <span style={{ fontSize: '12px', fontWeight: 700, color: '#B91C1C' }}>P &ge; 75.0%</span>
              </div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: '#991B1B', marginBottom: '6px' }}>High-Confidence Ambiguity</div>
              <p style={{ fontSize: '12px', color: '#7F1D1D', lineHeight: 1.5, margin: 0 }}>
                Requirement contains subjective adjectives (<em>intuitive</em>, <em>fast</em>), vague terms (<em>appropriate</em>, <em>suitable</em>), 
                or open-ended clauses without acceptance criteria. Requires clarification and rewriting before sprint planning.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Usage Guide: Single vs Batch Analysis */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
        {/* Single Analyzer Guide */}
        <div className="bi-card">
          <div className="bi-card-header">
            <div className="bi-card-title">
              <Search size={18} style={{ color: 'var(--color-primary)' }} />
              Single Requirement Analysis Instructions
            </div>
          </div>
          <div className="bi-card-body">
            <ol style={{ paddingLeft: '18px', margin: 0, fontSize: '13px', color: 'var(--color-text-primary)', lineHeight: 1.7 }}>
              <li>
                <strong>Input Requirement:</strong> Type or paste an individual software specification into the input area on the <em>Single Requirement</em> workspace.
              </li>
              <li>
                <strong>Preset Evaluation:</strong> Use the one-click preset buttons (e.g., <em>Latency Ambiguity</em>, <em>Measurable SLA</em>) to quickly test benchmark edge cases.
              </li>
              <li>
                <strong>Execute Analysis:</strong> Click <strong>Analyze Requirement</strong>. Typical inference latency is 15–30ms on standard CPU.
              </li>
              <li>
                <strong>Inspect Cues & Rewrite:</strong> Review highlighted linguistic triggers, missing measurement specifications, and the rule-based suggested rewrite.
              </li>
            </ol>
          </div>
        </div>

        {/* Batch Analyzer Guide */}
        <div className="bi-card">
          <div className="bi-card-header">
            <div className="bi-card-title">
              <UploadCloud size={18} style={{ color: 'var(--color-primary)' }} />
              Batch Analysis & File Ingestion
            </div>
          </div>
          <div className="bi-card-body">
            <ol style={{ paddingLeft: '18px', margin: 0, fontSize: '13px', color: 'var(--color-text-primary)', lineHeight: 1.7 }}>
              <li>
                <strong>Supported Formats:</strong> Upload standard <code>.csv</code>, <code>.xlsx</code>, or <code>.txt</code> files containing requirements.
              </li>
              <li>
                <strong>Column Auto-Detection:</strong> The parser automatically detects common headers (<code>requirement</code>, <code>text</code>, <code>description</code>, <code>srs_requirement</code>).
              </li>
              <li>
                <strong>Download Sample Files:</strong> Use the header links to download verified sample CSV or XLSX templates formatted for immediate ingestion.
              </li>
              <li>
                <strong>BI Data Table & Export:</strong> Filter records by predicted status, search requirement text, and export complete analytical results with confidence scores to CSV.
              </li>
            </ol>
          </div>
        </div>
      </div>

      {/* Academic Rigor & Data Leakage Prevention */}
      <div className="bi-card">
        <div className="bi-card-header">
          <div className="bi-card-title">
            <Database size={18} style={{ color: 'var(--color-primary)' }} />
            Dataset Provenance & Rigorous Leakage Prevention
          </div>
          <span className="badge-status badge-success">Academic Rigor Verified</span>
        </div>
        <div className="bi-card-body">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
            <div style={{ border: '1px solid var(--color-border)', borderRadius: '6px', padding: '14px', background: '#F8FAFC' }}>
              <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--color-heading)', marginBottom: '4px' }}>
                Group-Aware Stratification (GroupShuffleSplit)
              </div>
              <p style={{ fontSize: '12.5px', color: 'var(--color-text-secondary)', lineHeight: 1.5, margin: 0 }}>
                Every requirement belongs to a semantic cluster (<code>semantic_group_id</code>). All paraphrases and clear/ambiguous pairs 
                are isolated together into train, validation, or test sets. No shared sentence structures cross the 80/20 partition boundary.
              </p>
            </div>

            <div style={{ border: '1px solid var(--color-border)', borderRadius: '6px', padding: '14px', background: '#F8FAFC' }}>
              <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--color-heading)', marginBottom: '4px' }}>
                Zero Tautological Heuristic Leakage
              </div>
              <p style={{ fontSize: '12.5px', color: 'var(--color-text-secondary)', lineHeight: 1.5, margin: 0 }}>
                Earlier prototype versions suffered from circular heuristic labeling. Production dataset v4 strictly decoupled regex feature extraction 
                from ground-truth labels, requiring semantic validity and independent human consensus for every sample.
              </p>
            </div>

            <div style={{ border: '1px solid var(--color-border)', borderRadius: '6px', padding: '14px', background: '#F8FAFC' }}>
              <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--color-heading)', marginBottom: '4px' }}>
                Unseen Challenge Set Isolation
              </div>
              <p style={{ fontSize: '12.5px', color: 'var(--color-text-secondary)', lineHeight: 1.5, margin: 0 }}>
                The 80-sample Challenge Benchmark (<code>challenge_test_set.csv</code>) and external corpora (PURE, User Stories, Fault-Prone SRS) 
                were never exposed during model training, hyperparameter tuning, or vocabulary construction.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Known Limitations & Operational Boundaries */}
      <div className="bi-card">
        <div className="bi-card-header">
          <div className="bi-card-title">
            <AlertTriangle size={18} style={{ color: '#D97706' }} />
            Operational Boundaries & Known Limitations
          </div>
          <span className="badge-status badge-warning">Engineering Boundaries</span>
        </div>
        <div className="bi-card-body">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px' }}>
            <div style={{ display: 'flex', gap: '10px' }}>
              <div style={{ color: '#D97706', marginTop: '2px' }}><AlertTriangle size={16} /></div>
              <div>
                <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-heading)' }}>Agile User Story Format Shift</div>
                <div style={{ fontSize: '12.5px', color: 'var(--color-text-secondary)', lineHeight: 1.5, marginTop: '2px' }}>
                  Requirements written in the format <em>"As a [role], I want [action] so that [benefit]"</em> without explicit acceptance criteria 
                  often score in the Needs Review band (75.30% raw accuracy). Adding acceptance criteria resolves this ambiguity.
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '10px' }}>
              <div style={{ color: '#D97706', marginTop: '2px' }}><AlertTriangle size={16} /></div>
              <div>
                <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-heading)' }}>Domain-Specific Acronyms</div>
                <div style={{ fontSize: '12.5px', color: 'var(--color-text-secondary)', lineHeight: 1.5, marginTop: '2px' }}>
                  Highly specialized industry abbreviations (e.g. medical DICOM standards, avionics DO-178C protocols) 
                  are evaluated based on syntactic context rather than proprietary domain dictionaries.
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '10px' }}>
              <div style={{ color: '#D97706', marginTop: '2px' }}><AlertTriangle size={16} /></div>
              <div>
                <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-heading)' }}>Passive Voice Intent</div>
                <div style={{ fontSize: '12.5px', color: 'var(--color-text-secondary)', lineHeight: 1.5, marginTop: '2px' }}>
                  Grammatical passive voice (<em>"The report shall be generated"</em>) is flagged as potentially ambiguous regarding agent attribution, 
                  even when the action itself is clearly described.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
