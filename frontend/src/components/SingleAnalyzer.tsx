// frontend/src/components/SingleAnalyzer.tsx — Power BI-Inspired Single Requirement Workspace
import React, { useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  FileCheck2,
  Info,
  RefreshCw,
  Search,
} from 'lucide-react';
import type { SingleAnalyzeResponse } from '../types';
import { analyzeRequirement } from '../services/api';

const SAMPLE_REQUIREMENTS = [
  {
    label: 'Vague Temporal',
    text: 'The system shall respond to user requests quickly.',
  },
  {
    label: 'Measurable Constraint',
    text: 'The system shall return search results within 2 seconds for 95% of requests.',
  },
  {
    label: 'Subjective Security',
    text: 'The system shall store user data securely.',
  },
  {
    label: 'Hard Pool Limit',
    text: 'The database connection pool shall have a maximum size of 50 connections.',
  },
  {
    label: 'Borderline Review',
    text: 'The background process takes approximately 10 minutes.',
  },
  {
    label: 'Adversarial Trap',
    text: 'The system shall provide adequate capacity of 100 GB.',
  },
];

export const SingleAnalyzer: React.FC = () => {
  const [requirementText, setRequirementText] = useState(
    'The system shall respond to user requests quickly.'
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SingleAnalyzeResponse | null>(null);
  const [copied, setCopied] = useState(false);
  const [showFeatures, setShowFeatures] = useState(false);

  const handleAnalyze = async () => {
    if (!requirementText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeRequirement(requirementText);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'An error occurred during requirement analysis.');
    } finally {
      setLoading(false);
    }
  };

  const handleCopyRewrite = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const renderHighlightedText = (text: string, cues: string[]) => {
    if (!cues || cues.length === 0) {
      return <span>{text}</span>;
    }

    const escapedCues = cues.map((c) => c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const regex = new RegExp(`\\b(${escapedCues.join('|')})\\b`, 'gi');
    const parts = text.split(regex);

    return (
      <span>
        {parts.map((part, index) => {
          const isCue = cues.some((c) => c.toLowerCase() === part.toLowerCase());
          if (isCue) {
            return (
              <mark
                key={index}
                style={{
                  backgroundColor: '#FEE2E2',
                  color: '#991B1B',
                  fontWeight: 600,
                  padding: '1px 5px',
                  borderRadius: '3px',
                  border: '1px solid #FCA5A5',
                }}
                title={`Flagged ambiguity trigger phrase: "${part}"`}
              >
                {part}
              </mark>
            );
          }
          return <span key={index}>{part}</span>;
        })}
      </span>
    );
  };

  const getOutcomeTheme = (predictedClass: string) => {
    switch (predictedClass) {
      case 'AMBIGUOUS':
        return {
          badgeClass: 'badge-status ambiguous',
          borderColor: '#EF4444',
          bgColor: '#FEF2F2',
          textColor: '#B91C1C',
          icon: <AlertTriangle size={15} />,
          title: 'Ambiguous — Specification Defect Flagged',
          subtext: 'Lacks measurable constraints or contains subjective terminology.',
        };
      case 'NOT AMBIGUOUS':
        return {
          badgeClass: 'badge-status clear',
          borderColor: '#22C55E',
          bgColor: '#F0FDF4',
          textColor: '#15803D',
          icon: <CheckCircle2 size={15} />,
          title: 'Not Ambiguous — Verifiable Specification',
          subtext: 'Provides adequate testable criteria with measurable units or standard protocols.',
        };
      case 'NEEDS REVIEW':
        return {
          badgeClass: 'badge-status review',
          borderColor: '#F59E0B',
          bgColor: '#FFFBEB',
          textColor: '#B45309',
          icon: <HelpCircle size={15} />,
          title: 'Needs Review — Uncertainty Band Detected',
          subtext: 'Conflicting signals present. Human requirements engineer review required.',
        };
      default:
        return {
          badgeClass: 'badge-status skipped',
          borderColor: '#94A3B8',
          bgColor: '#F8FAFC',
          textColor: '#475569',
          icon: <Info size={15} />,
          title: 'Invalid Input',
          subtext: 'Input does not meet the 3-word minimum requirement threshold.',
        };
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Input Workspace Card */}
      <div className="bi-card">
        <div className="bi-card-header">
          <div>
            <div className="bi-card-title">
              <FileCheck2 size={17} color="var(--accent-primary)" />
              Single Requirement Workspace
            </div>
            <div className="bi-card-subtitle">
              Enter or load a specification statement to diagnose semantic ambiguity, detect vague terms, and verify numerical bounds.
            </div>
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
            Word count: <strong>{requirementText.trim().split(/\s+/).filter(Boolean).length}</strong> | Chars: <strong>{requirementText.length}</strong>
          </div>
        </div>

        <div className="bi-card-body">
          {/* Quick Preset Chips */}
          <div style={{ marginBottom: '12px' }}>
            <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginRight: '8px' }}>
              Load Sample:
            </span>
            <div className="quick-fill-bar" style={{ display: 'inline-flex' }}>
              {SAMPLE_REQUIREMENTS.map((sample, idx) => (
                <button
                  key={idx}
                  className="chip-btn"
                  onClick={() => {
                    setRequirementText(sample.text);
                    setResult(null);
                    setError(null);
                  }}
                  type="button"
                >
                  {sample.label}
                </button>
              ))}
            </div>
          </div>

          {/* Text Area */}
          <div style={{ marginBottom: '16px' }}>
            <label htmlFor="requirement-input-textarea" className="form-label">
              Software Requirement Statement:
            </label>
            <textarea
              id="requirement-input-textarea"
              className="form-textarea"
              rows={3}
              value={requirementText}
              onChange={(e) => setRequirementText(e.target.value)}
              placeholder="e.g. The system shall respond to user requests within 150 ms for 99% of queries."
            />
          </div>

          {/* Action Row */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <button
              className="btn btn-primary"
              id="analyze-submit-btn"
              onClick={handleAnalyze}
              disabled={loading || !requirementText.trim()}
            >
              {loading ? (
                <>
                  <RefreshCw size={15} style={{ animation: 'spin 1s linear infinite' }} />
                  Processing via SBERT Engine...
                </>
              ) : (
                <>
                  <Search size={15} />
                  Analyze Requirement
                </>
              )}
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => {
                setRequirementText('');
                setResult(null);
                setError(null);
              }}
              type="button"
            >
              Clear
            </button>
          </div>

          {error && (
            <div
              style={{
                marginTop: '16px',
                padding: '12px 16px',
                borderRadius: 'var(--radius-sm)',
                backgroundColor: 'var(--status-ambiguous-bg)',
                border: '1px solid var(--status-ambiguous-border)',
                color: 'var(--status-ambiguous-text)',
                fontSize: '13px',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
              }}
            >
              <AlertTriangle size={16} />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>

      {/* Analysis Results View */}
      {result && (
        <div className="bi-card" id="analysis-result-card" style={{ borderLeft: `4px solid ${getOutcomeTheme(result.predicted_class).borderColor}` }}>
          {/* Header Summary Banner */}
          <div className="bi-card-header" style={{ backgroundColor: getOutcomeTheme(result.predicted_class).bgColor }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
              <span className={getOutcomeTheme(result.predicted_class).badgeClass} id="result-status-badge">
                {getOutcomeTheme(result.predicted_class).icon}
                {result.predicted_class}
              </span>
              <div>
                <div style={{ fontSize: '14.5px', fontWeight: 700, color: getOutcomeTheme(result.predicted_class).textColor }}>
                  {getOutcomeTheme(result.predicted_class).title}
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                  {getOutcomeTheme(result.predicted_class).subtext}
                </div>
              </div>
            </div>

            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Calibrated Ambiguity Probability</div>
              <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-heading)', fontFamily: 'var(--font-mono)' }}>
                {result.confidence_score.toFixed(1)}%
              </div>
            </div>
          </div>

          <div className="bi-card-body">
            {/* Probability Scale Visual */}
            <div style={{ marginBottom: '20px', padding: '14px 18px', backgroundColor: 'var(--bg-subtle)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
                <span style={{ color: '#15803D', fontWeight: 600 }}>Clear (0% - 25%)</span>
                <span style={{ color: '#B45309', fontWeight: 600 }}>Needs Review (25% - 75%)</span>
                <span style={{ color: '#B91C1C', fontWeight: 600 }}>Ambiguous (75% - 100%)</span>
              </div>
              <div className="progress-track" style={{ height: '10px' }}>
                <div
                  className="progress-fill"
                  style={{
                    width: `${Math.max(4, Math.min(100, result.confidence_score))}%`,
                    backgroundColor:
                      result.predicted_class === 'AMBIGUOUS'
                        ? '#EF4444'
                        : result.predicted_class === 'NOT AMBIGUOUS'
                        ? '#22C55E'
                        : '#F59E0B',
                  }}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                <span>0.0%</span>
                <span>Posterior Probability: <strong>{result.confidence_score.toFixed(1)}%</strong></span>
                <span>100.0%</span>
              </div>
            </div>

            {/* Three-Tier Human Review Callout */}
            {result.predicted_class === 'NEEDS REVIEW' && (
              <div
                style={{
                  backgroundColor: 'var(--status-review-bg)',
                  border: '1px solid var(--status-review-border)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '14px 16px',
                  marginBottom: '20px',
                  display: 'flex',
                  gap: '12px',
                  alignItems: 'flex-start',
                }}
              >
                <HelpCircle size={18} color="#D97706" style={{ flexShrink: 0, marginTop: '2px' }} />
                <div style={{ fontSize: '13px', color: '#78350F', lineHeight: 1.5 }}>
                  <strong style={{ color: '#92400E' }}>Three-Tier Decision Policy: Automated Verification Deferred</strong>
                  <p style={{ marginTop: '2px' }}>
                    The calibrated probability is <strong>{result.confidence_score.toFixed(1)}%</strong>, resting within the uncertainty band (25.0% to 75.0%). The system detected conflicting cues (e.g. an approximate quantity or soft modal accompanied by a numeric constraint). A human engineer must verify this requirement to maintain 100% production accuracy.
                  </p>
                </div>
              </div>
            )}

            {/* Diagnostic Token Inspection */}
            <div style={{ marginBottom: '20px' }}>
              <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.4px', marginBottom: '8px' }}>
                Linguistic Inspection & Diagnostic Tokens
              </div>
              <div
                style={{
                  padding: '14px 18px',
                  backgroundColor: '#FFFFFF',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '14px',
                  lineHeight: 1.6,
                  color: 'var(--text-primary)',
                }}
              >
                {renderHighlightedText(result.requirement, result.detected_cues)}
              </div>

              {result.detected_cues && result.detected_cues.length > 0 && (
                <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 600 }}>Detected Cues:</span>
                  {result.detected_cues.map((cue, idx) => (
                    <span key={idx} className="badge-status ambiguous" style={{ fontSize: '11px' }}>
                      "{cue}"
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Two Column Grid: Missing Specs + Suggested Rewrite */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px', marginBottom: '20px' }}>
              {/* Missing Specs */}
              <div style={{ padding: '16px', backgroundColor: 'var(--bg-subtle)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)' }}>
                <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-heading)', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '10px' }}>
                  <AlertTriangle size={15} color="#D97706" />
                  Missing Measurable Constraints
                </div>
                {result.missing_specifications && result.missing_specifications.length > 0 ? (
                  <ul style={{ listStyle: 'none', paddingLeft: 0, display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {result.missing_specifications.map((item, idx) => (
                      <li key={idx} style={{ fontSize: '12.5px', color: 'var(--text-primary)', display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                        <span style={{ width: '5px', height: '5px', borderRadius: '50%', backgroundColor: '#D97706', marginTop: '6px', flexShrink: 0 }} />
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>
                    No missing metric specifications identified. Requirement provides measurable bounds.
                  </p>
                )}
              </div>

              {/* Suggested Rewrite */}
              <div style={{ padding: '16px', backgroundColor: '#F0FDF4', border: '1px solid #BBF7D0', borderRadius: 'var(--radius-sm)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <div style={{ fontSize: '13px', fontWeight: 700, color: '#15803D', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <CheckCircle2 size={15} />
                    Suggested Objective Rewrite
                  </div>
                  {result.suggested_rewrite && (
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => handleCopyRewrite(result.suggested_rewrite!)}
                      style={{ padding: '2px 8px', fontSize: '11.5px', gap: '4px' }}
                      type="button"
                    >
                      {copied ? <Check size={12} color="#16A34A" /> : <Copy size={12} />}
                      {copied ? 'Copied' : 'Copy'}
                    </button>
                  )}
                </div>
                {result.suggested_rewrite ? (
                  <>
                    <p style={{ fontSize: '13px', color: '#14532D', fontStyle: 'italic', fontWeight: 500, marginBottom: '8px' }}>
                      "{result.suggested_rewrite}"
                    </p>
                    <div style={{ fontSize: '11px', color: '#166534', display: 'flex', alignItems: 'center', gap: '5px' }}>
                      <Info size={12} />
                      <span>{result.rewrite_disclaimer}</span>
                    </div>
                  </>
                ) : (
                  <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>
                    Requirement already provides unambiguous objective criteria; no rewrite template necessary.
                  </p>
                )}
              </div>
            </div>

            {/* Explanation Rationale Breakdown */}
            <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '16px', marginBottom: '16px' }}>
              <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.4px', marginBottom: '10px' }}>
                Classification Rationale (Rule-Based Evidence vs ML Confidence)
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '14px' }}>
                <div style={{ padding: '12px 14px', backgroundColor: 'var(--bg-subtle)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--accent-primary)', marginBottom: '4px' }}>
                    Rule-Engine Evidence:
                  </div>
                  {result.explanation.rule_based_evidence && result.explanation.rule_based_evidence.length > 0 ? (
                    <ul style={{ listStyle: 'none', paddingLeft: 0, fontSize: '12.5px', color: 'var(--text-primary)', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                      {result.explanation.rule_based_evidence.map((ev, i) => (
                        <li key={i}>• {ev}</li>
                      ))}
                    </ul>
                  ) : (
                    <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                      No deterministic trigger terms matched lexicon rules.
                    </p>
                  )}
                  <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '6px' }}>
                    {result.explanation.reason}
                  </p>
                </div>

                <div style={{ padding: '12px 14px', backgroundColor: 'var(--bg-subtle)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-heading)', marginBottom: '4px' }}>
                    Machine Learning Posterior:
                  </div>
                  <p style={{ fontSize: '12.5px', color: 'var(--text-primary)', marginBottom: '6px' }}>
                    {result.explanation.ml_confidence_evidence}
                  </p>
                  <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                    {result.explanation.decision_policy_note}
                  </p>
                </div>
              </div>
            </div>

            {/* Collapsible 24-Feature Data Grid */}
            <div>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => setShowFeatures(!showFeatures)}
                style={{ width: '100%', justifyContent: 'space-between', padding: '8px 12px' }}
                type="button"
              >
                <span>Inspect 24 Handcrafted Feature Values (FeatureExtractor)</span>
                {showFeatures ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
              </button>

              {showFeatures && result.linguistic_features && (
                <div style={{ marginTop: '10px', padding: '14px', backgroundColor: 'var(--bg-subtle)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(190px, 1fr))', gap: '8px', fontSize: '12px' }}>
                    {Object.entries(result.linguistic_features).map(([key, val]) => (
                      <div key={key} style={{ padding: '6px 10px', backgroundColor: '#FFFFFF', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-xs)' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '11px' }}>{key}</div>
                        <div style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-heading)' }}>{String(val)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
