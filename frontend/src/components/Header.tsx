// frontend/src/components/Header.tsx — Top Application Bar for Power BI Canvas
import React from 'react';
import {
  CheckCircle2,
  AlertCircle,
  RefreshCw,
} from 'lucide-react';
import type { HealthResponse } from '../types';

interface HeaderProps {
  activeTab: 'overview' | 'single' | 'batch' | 'eval' | 'guide';
  health: HealthResponse | null;
  healthLoading: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  health,
  healthLoading,
}) => {
  const getPageMeta = () => {
    switch (activeTab) {
      case 'overview':
        return {
          title: 'Requirement Quality Analytics — Executive Overview',
          subtitle: 'Real-time ambiguity telemetry, verified held-out model benchmarks, and cross-dataset metrics.',
        };
      case 'single':
        return {
          title: 'Single Requirement Inspection Workspace',
          subtitle: 'Analyze software requirements for linguistic ambiguity, missing measurable bounds, and suggested rewrites.',
        };
      case 'batch':
        return {
          title: 'Batch Specification Inspection & Spreadsheets',
          subtitle: 'Upload requirement workbooks (.xlsx, .txt) or paste multiple statements for automated batch auditing.',
        };
      case 'eval':
        return {
          title: 'Model Evaluation & Benchmark Verification Report',
          subtitle: 'Audited empirical benchmark reports comparing in-domain held-out test data against external corpora.',
        };
      case 'guide':
        return {
          title: 'System Guide & Architecture Documentation',
          subtitle: 'Comprehensive technical documentation on hybrid feature fusion, decision policies, and dataset provenance.',
        };
      default:
        return {
          title: 'IntelliReq Analytics',
          subtitle: 'Software Requirement Ambiguity Detection System',
        };
    }
  };

  const meta = getPageMeta();

  return (
    <header className="top-appbar">
      <div className="appbar-heading">
        <h2>{meta.title}</h2>
        <p>{meta.subtitle}</p>
      </div>

      <div className="appbar-actions">
        {healthLoading ? (
          <div className="status-badge">
            <RefreshCw size={13} style={{ animation: 'spin 1.5s linear infinite' }} />
            <span>Connecting...</span>
          </div>
        ) : health?.model_loaded ? (
          <div className="status-badge online" title={`Engine: ${health.model_version || 'v2.1.0'} with ${health.embedding_model || 'all-MiniLM-L6-v2'}`}>
            <span className="status-indicator-dot" />
            <CheckCircle2 size={13} />
            <span>FastAPI Online • Calibrated L1 LR</span>
          </div>
        ) : (
          <div className="status-badge offline">
            <span className="status-indicator-dot" />
            <AlertCircle size={13} />
            <span>FastAPI Offline</span>
          </div>
        )}
      </div>
    </header>
  );
};
