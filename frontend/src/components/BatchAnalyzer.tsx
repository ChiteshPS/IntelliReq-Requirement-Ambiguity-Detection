// frontend/src/components/BatchAnalyzer.tsx — Power BI-Inspired Batch Specification Grid
import React, { useState } from 'react';
import {
  Layers,
  Upload,
  RefreshCw,
  Download,
  Search,
  Filter,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  AlertCircle,
  FileSpreadsheet,
  X,
} from 'lucide-react';
import type { BatchAnalyzeResponse, BatchItemResponse } from '../types';
import { analyzeBatchRequirements, analyzeFile } from '../services/api';

const DEFAULT_BATCH_TEXT = `The system shall respond to user requests quickly.
The system shall return search results within 2 seconds for 95% of queries.
The user profile page shall include fields for First Name, Last Name, and Email Address.
The system shall store user data securely.
The database connection pool shall have a maximum size of 50 connections.
The background process takes approximately 10 minutes.
The platform shall be highly scalable and robust against unexpected errors.
The backend will safely parse inputs.
Short
The system shall synchronize data with the master node every 5 minutes.`;

export const BatchAnalyzer: React.FC = () => {
  const [inputText, setInputText] = useState(DEFAULT_BATCH_TEXT);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [batchResponse, setBatchResponse] = useState<BatchAnalyzeResponse | null>(null);
  const [statusFilter, setStatusFilter] = useState<'ALL' | 'AMBIGUOUS' | 'NOT AMBIGUOUS' | 'NEEDS REVIEW' | 'SKIPPED'>('ALL');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedItem, setSelectedItem] = useState<BatchItemResponse | null>(null);

  const handleProcessTextBatch = async () => {
    const lines = inputText
      .split('\n')
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    if (lines.length === 0) {
      setError('Please enter at least one requirement sentence.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const items = lines.map((text, idx) => ({
        id: `REQ-${idx + 1}`,
        requirement: text,
      }));
      const data = await analyzeBatchRequirements(items);
      setBatchResponse(data);
    } catch (err: any) {
      setError(err.message || 'Batch processing failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    try {
      const data = await analyzeFile(file);
      setBatchResponse(data);
    } catch (err: any) {
      setError(err.message || `Failed to process file ${file.name}`);
    } finally {
      setLoading(false);
      e.target.value = ''; // Reset input
    }
  };

  const handleExportCSV = () => {
    if (!batchResponse || batchResponse.results.length === 0) return;

    const headers = ['ID', 'Requirement', 'Predicted_Class', 'Confidence_%', 'Reason', 'Detected_Cues', 'Suggested_Rewrite'];
    const rows = batchResponse.results.map((r) => [
      `"${r.id}"`,
      `"${r.requirement.replace(/"/g, '""')}"`,
      `"${r.predicted_class}"`,
      r.confidence_score !== null && r.confidence_score !== undefined ? r.confidence_score.toFixed(1) : 'N/A',
      `"${(r.reason || '').replace(/"/g, '""')}"`,
      `"${(r.detected_cues || []).join('; ')}"`,
      `"${(r.suggested_rewrite || '').replace(/"/g, '""')}"`,
    ]);

    const csvContent = [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `intellireq_batch_results_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const filteredResults =
    batchResponse?.results.filter((item) => {
      const matchesStatus = statusFilter === 'ALL' || item.predicted_class === statusFilter;
      const matchesSearch =
        searchTerm.trim() === '' ||
        item.requirement.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (item.reason && item.reason.toLowerCase().includes(searchTerm.toLowerCase()));
      return matchesStatus && matchesSearch;
    }) || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Input Configuration Card */}
      <div className="bi-card">
        <div className="bi-card-header">
          <div>
            <div className="bi-card-title">
              <FileSpreadsheet size={17} color="var(--accent-primary)" />
              Batch Specification Inspection
            </div>
            <div className="bi-card-subtitle">
              Audit software requirement workbooks at scale. Enter specifications line-by-line or upload <code>.txt</code> / <code>.xlsx</code> files.
            </div>
          </div>

          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <label
              className="btn btn-secondary btn-sm"
              style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}
              title="Upload an Excel workbook or plain text document"
            >
              <Upload size={14} />
              <span>Import .TXT / .XLSX</span>
              <input
                type="file"
                accept=".txt,.xlsx"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
              />
            </label>
          </div>
        </div>

        <div className="bi-card-body">
          <label htmlFor="batch-input-textarea" className="form-label">
            Requirements Text (One specification per row):
          </label>
          <textarea
            id="batch-input-textarea"
            className="form-textarea"
            rows={5}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Paste requirements, one per line..."
            style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', marginBottom: '14px' }}
          />

          <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              className="btn btn-primary"
              id="batch-process-btn"
              onClick={handleProcessTextBatch}
              disabled={loading || !inputText.trim()}
            >
              {loading ? (
                <>
                  <RefreshCw size={15} style={{ animation: 'spin 1s linear infinite' }} />
                  Auditing Batch Specifications...
                </>
              ) : (
                <>
                  <Layers size={15} />
                  Analyze Batch ({inputText.split('\n').filter((l) => l.trim()).length} Items)
                </>
              )}
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => setInputText(DEFAULT_BATCH_TEXT)}
              type="button"
            >
              Reset to Sample Data
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => {
                setInputText('');
                setBatchResponse(null);
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
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>

      {/* Batch Results & Summary View */}
      {batchResponse && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* KPI Summary Tiles */}
          <div className="kpi-grid">
            <div className="kpi-tile primary">
              <div className="kpi-tile-header">
                <span className="kpi-tile-label">Total Processed</span>
                <Layers size={16} className="kpi-tile-icon" />
              </div>
              <div className="kpi-tile-value">{batchResponse.summary.total_processed}</div>
              <div className="kpi-tile-subtext">Requirements inspected</div>
            </div>

            <div className="kpi-tile" style={{ borderTop: '3px solid #EF4444' }}>
              <div className="kpi-tile-header">
                <span className="kpi-tile-label">Ambiguous Items</span>
                <AlertTriangle size={16} style={{ color: '#EF4444' }} />
              </div>
              <div className="kpi-tile-value" style={{ color: '#B91C1C' }}>
                {batchResponse.summary.ambiguous_count}
              </div>
              <div className="kpi-tile-subtext">
                Ambiguity Rate: <strong>{batchResponse.summary.ambiguity_rate_pct.toFixed(1)}%</strong>
              </div>
            </div>

            <div className="kpi-tile success">
              <div className="kpi-tile-header">
                <span className="kpi-tile-label">Unambiguous (Clear)</span>
                <CheckCircle2 size={16} style={{ color: '#16A34A' }} />
              </div>
              <div className="kpi-tile-value" style={{ color: '#15803D' }}>
                {batchResponse.summary.not_ambiguous_count}
              </div>
              <div className="kpi-tile-subtext">Passes testable constraints</div>
            </div>

            <div className="kpi-tile warning">
              <div className="kpi-tile-header">
                <span className="kpi-tile-label">Needs Review</span>
                <HelpCircle size={16} style={{ color: '#D97706' }} />
              </div>
              <div className="kpi-tile-value" style={{ color: '#B45309' }}>
                {batchResponse.summary.needs_review_count}
              </div>
              <div className="kpi-tile-subtext">Uncertainty band (25% - 75%)</div>
            </div>

            {batchResponse.summary.skipped_count > 0 && (
              <div className="kpi-tile" style={{ borderTop: '3px solid #94A3B8' }}>
                <div className="kpi-tile-header">
                  <span className="kpi-tile-label">Skipped (Too Short)</span>
                  <AlertCircle size={16} style={{ color: '#64748B' }} />
                </div>
                <div className="kpi-tile-value" style={{ color: '#475569' }}>
                  {batchResponse.summary.skipped_count}
                </div>
                <div className="kpi-tile-subtext">&lt; 3 words minimum</div>
              </div>
            )}
          </div>

          {/* Results Table Card */}
          <div className="bi-card">
            {/* Filter and Search Toolbar */}
            <div className="bi-card-header" style={{ padding: '12px 18px', backgroundColor: 'var(--bg-subtle)' }}>
              <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', alignItems: 'center' }}>
                <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '4px', marginRight: '6px' }}>
                  <Filter size={13} /> Filter:
                </span>
                <button
                  className={`btn btn-sm ${statusFilter === 'ALL' ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setStatusFilter('ALL')}
                  type="button"
                >
                  All ({batchResponse.results.length})
                </button>
                <button
                  className={`btn btn-sm ${statusFilter === 'AMBIGUOUS' ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setStatusFilter('AMBIGUOUS')}
                  style={{ color: statusFilter === 'AMBIGUOUS' ? '#FFF' : '#B91C1C' }}
                  type="button"
                >
                  Ambiguous ({batchResponse.summary.ambiguous_count})
                </button>
                <button
                  className={`btn btn-sm ${statusFilter === 'NOT AMBIGUOUS' ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setStatusFilter('NOT AMBIGUOUS')}
                  style={{ color: statusFilter === 'NOT AMBIGUOUS' ? '#FFF' : '#15803D' }}
                  type="button"
                >
                  Clear ({batchResponse.summary.not_ambiguous_count})
                </button>
                <button
                  className={`btn btn-sm ${statusFilter === 'NEEDS REVIEW' ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setStatusFilter('NEEDS REVIEW')}
                  style={{ color: statusFilter === 'NEEDS REVIEW' ? '#FFF' : '#B45309' }}
                  type="button"
                >
                  Review ({batchResponse.summary.needs_review_count})
                </button>
              </div>

              <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                <div style={{ position: 'relative' }}>
                  <Search size={13} style={{ position: 'absolute', left: '10px', top: '9px', color: 'var(--text-muted)' }} />
                  <input
                    type="text"
                    className="form-input"
                    placeholder="Search specifications..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    style={{ padding: '5px 10px 5px 28px', fontSize: '12.5px', width: '200px' }}
                  />
                </div>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={handleExportCSV}
                  style={{ gap: '6px' }}
                  type="button"
                  title="Export results table to CSV"
                >
                  <Download size={13} />
                  <span>Export CSV</span>
                </button>
              </div>
            </div>

            {/* Table */}
            <div className="bi-table-container" style={{ border: 'none' }}>
              <table className="bi-table" id="batch-results-table">
                <thead>
                  <tr>
                    <th style={{ width: '80px' }}>Row ID</th>
                    <th>Requirement Text</th>
                    <th style={{ width: '150px' }}>Classification</th>
                    <th style={{ width: '110px' }}>Confidence</th>
                    <th style={{ width: '180px' }}>Detected Cues</th>
                    <th style={{ width: '90px', textAlign: 'center' }}>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredResults.length === 0 ? (
                    <tr>
                      <td colSpan={6} style={{ textAlign: 'center', padding: '36px', color: 'var(--text-secondary)' }}>
                        No requirement records match the selected filter criteria.
                      </td>
                    </tr>
                  ) : (
                    filteredResults.map((row) => (
                      <tr key={row.id}>
                        <td className="mono" style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>
                          {row.id}
                        </td>
                        <td style={{ maxWidth: '440px', lineHeight: 1.45, color: 'var(--text-heading)' }}>
                          {row.requirement}
                        </td>
                        <td>
                          {row.predicted_class === 'AMBIGUOUS' ? (
                            <span className="badge-status ambiguous">
                              <AlertTriangle size={12} /> Ambiguous
                            </span>
                          ) : row.predicted_class === 'NOT AMBIGUOUS' ? (
                            <span className="badge-status clear">
                              <CheckCircle2 size={12} /> Not Ambiguous
                            </span>
                          ) : row.predicted_class === 'NEEDS REVIEW' ? (
                            <span className="badge-status review">
                              <HelpCircle size={12} /> Needs Review
                            </span>
                          ) : (
                            <span className="badge-status skipped">
                              Skipped
                            </span>
                          )}
                        </td>
                        <td className="mono">
                          {row.confidence_score !== null && row.confidence_score !== undefined
                            ? `${row.confidence_score.toFixed(1)}%`
                            : '—'}
                        </td>
                        <td>
                          {row.detected_cues && row.detected_cues.length > 0 ? (
                            <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                              {row.detected_cues.slice(0, 2).map((cue, i) => (
                                <span key={i} className="tag-cue" style={{ color: '#B91C1C' }}>
                                  "{cue}"
                                </span>
                              ))}
                              {row.detected_cues.length > 2 && (
                                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                                  +{row.detected_cues.length - 2}
                                </span>
                              )}
                            </div>
                          ) : (
                            <span style={{ color: 'var(--text-muted)', fontSize: '12px' }}>None</span>
                          )}
                        </td>
                        <td style={{ textAlign: 'center' }}>
                          <button
                            className="btn btn-secondary btn-sm"
                            style={{ padding: '2px 8px', fontSize: '11px' }}
                            onClick={() => setSelectedItem(row)}
                            type="button"
                          >
                            Details
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            <div style={{ padding: '10px 18px', backgroundColor: 'var(--bg-subtle)', borderTop: '1px solid var(--border-subtle)', fontSize: '12px', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between' }}>
              <span>Showing <strong>{filteredResults.length}</strong> of <strong>{batchResponse.results.length}</strong> entries</span>
              <span>Model: <strong>Calibrated Logistic Regression (L1)</strong></span>
            </div>
          </div>

          {/* Row Detail Slide-out / Modal */}
          {selectedItem && (
            <div
              style={{
                position: 'fixed',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                backgroundColor: 'rgba(15, 23, 42, 0.45)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 9999,
                padding: '20px',
              }}
              onClick={() => setSelectedItem(null)}
            >
              <div
                className="bi-card"
                style={{
                  maxWidth: '620px',
                  width: '100%',
                  margin: 0,
                  boxShadow: 'var(--shadow-md)',
                  maxHeight: '90vh',
                  overflowY: 'auto',
                }}
                onClick={(e) => e.stopPropagation()}
              >
                <div className="bi-card-header" style={{ padding: '14px 18px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-primary)' }}>
                      {selectedItem.id}
                    </span>
                    <span className={`badge-status ${selectedItem.predicted_class === 'AMBIGUOUS' ? 'ambiguous' : selectedItem.predicted_class === 'NOT AMBIGUOUS' ? 'clear' : 'review'}`}>
                      {selectedItem.predicted_class}
                    </span>
                  </div>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => setSelectedItem(null)}
                    style={{ padding: '3px 8px' }}
                    type="button"
                  >
                    <X size={14} /> Close
                  </button>
                </div>

                <div className="bi-card-body" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                  <div>
                    <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600, marginBottom: '4px' }}>
                      Requirement Statement:
                    </div>
                    <p style={{ fontSize: '14.5px', color: 'var(--text-heading)', fontWeight: 600, lineHeight: 1.5 }}>
                      {selectedItem.requirement}
                    </p>
                  </div>

                  <div>
                    <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600, marginBottom: '4px' }}>
                      Diagnostic Assessment:
                    </div>
                    <p style={{ fontSize: '13px', color: 'var(--text-primary)', lineHeight: 1.5 }}>
                      {selectedItem.reason || 'No specific explanation returned.'}
                    </p>
                  </div>

                  {selectedItem.detected_cues && selectedItem.detected_cues.length > 0 && (
                    <div>
                      <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600, marginBottom: '6px' }}>
                        Detected Ambiguity Cues:
                      </div>
                      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                        {selectedItem.detected_cues.map((cue, i) => (
                          <span key={i} className="badge-status ambiguous" style={{ fontSize: '11px' }}>
                            "{cue}"
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {selectedItem.missing_specifications && selectedItem.missing_specifications.length > 0 && (
                    <div style={{ padding: '12px 14px', backgroundColor: 'var(--bg-subtle)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '12px', color: '#B45309', fontWeight: 700, marginBottom: '6px' }}>
                        Missing Measurable Constraints:
                      </div>
                      <ul style={{ paddingLeft: '16px', fontSize: '12.5px', color: 'var(--text-primary)', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                        {selectedItem.missing_specifications.map((m, idx) => (
                          <li key={idx}>{m}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {selectedItem.suggested_rewrite && (
                    <div style={{ padding: '12px 14px', backgroundColor: '#F0FDF4', borderRadius: 'var(--radius-sm)', border: '1px solid #BBF7D0' }}>
                      <div style={{ fontSize: '12px', color: '#15803D', fontWeight: 700, marginBottom: '4px' }}>
                        Suggested Objective Rewrite:
                      </div>
                      <p style={{ fontSize: '13.5px', color: '#14532D', fontStyle: 'italic', fontWeight: 500 }}>
                        "{selectedItem.suggested_rewrite}"
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
