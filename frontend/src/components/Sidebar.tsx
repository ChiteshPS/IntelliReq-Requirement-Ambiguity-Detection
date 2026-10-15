// frontend/src/components/Sidebar.tsx — Power BI Executive Sidebar Navigation
import React from 'react';
import {
  LayoutDashboard,
  FileText,
  Layers,
  BarChart3,
  BookOpen,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  Cpu,
} from 'lucide-react';
import type { HealthResponse } from '../types';

interface SidebarProps {
  activeTab: 'overview' | 'single' | 'batch' | 'eval' | 'guide';
  setActiveTab: (tab: 'overview' | 'single' | 'batch' | 'eval' | 'guide') => void;
  health: HealthResponse | null;
  healthLoading: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  health,
  healthLoading,
}) => {
  return (
    <aside className="app-sidebar">
      {/* Brand Header */}
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">
          <ShieldCheck size={20} />
        </div>
        <div className="sidebar-brand-info">
          <h1>IntelliReq</h1>
          <p>Requirements Quality Analytics</p>
        </div>
      </div>

      {/* Navigation Items */}
      <nav className="sidebar-nav">
        <div className="nav-section-title">Analytics & Workspaces</div>

        <button
          className={`sidebar-nav-btn ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
          id="nav-overview-btn"
          aria-selected={activeTab === 'overview'}
        >
          <LayoutDashboard size={18} />
          <span>Executive Overview</span>
        </button>

        <button
          className={`sidebar-nav-btn ${activeTab === 'single' ? 'active' : ''}`}
          onClick={() => setActiveTab('single')}
          id="nav-single-btn"
          aria-selected={activeTab === 'single'}
        >
          <FileText size={18} />
          <span>Single Requirement</span>
        </button>

        <button
          className={`sidebar-nav-btn ${activeTab === 'batch' ? 'active' : ''}`}
          onClick={() => setActiveTab('batch')}
          id="nav-batch-btn"
          aria-selected={activeTab === 'batch'}
        >
          <Layers size={18} />
          <span>Batch Analysis</span>
        </button>

        <div className="nav-section-title" style={{ marginTop: '12px' }}>
          Verification & Docs
        </div>

        <button
          className={`sidebar-nav-btn ${activeTab === 'eval' ? 'active' : ''}`}
          onClick={() => setActiveTab('eval')}
          id="nav-eval-btn"
          aria-selected={activeTab === 'eval'}
        >
          <BarChart3 size={18} />
          <span>Model Evaluation (97% Audit)</span>
        </button>

        <button
          className={`sidebar-nav-btn ${activeTab === 'guide' ? 'active' : ''}`}
          onClick={() => setActiveTab('guide')}
          id="nav-guide-btn"
          aria-selected={activeTab === 'guide'}
        >
          <BookOpen size={18} />
          <span>System Guide & Architecture</span>
        </button>
      </nav>

      {/* Model & System Health Footer */}
      <div className="sidebar-footer">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: '11px', color: 'var(--sidebar-text-muted)' }}>Core Engine</span>
          <span className="sidebar-badge">v2.1.0</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11.5px', marginTop: '2px' }}>
          <Cpu size={13} style={{ color: 'var(--accent-primary)' }} />
          <span>Calibrated L1 LR (408-d)</span>
        </div>

        <div style={{ marginTop: '6px', paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.08)' }}>
          {healthLoading ? (
            <div style={{ color: 'var(--sidebar-text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span className="status-indicator-dot" style={{ backgroundColor: '#F59E0B' }} />
              Connecting backend...
            </div>
          ) : health?.model_loaded ? (
            <div style={{ color: '#86EFAC', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <CheckCircle2 size={13} />
              <span>FastAPI Online (8000)</span>
            </div>
          ) : (
            <div style={{ color: '#FCA5A5', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <AlertCircle size={13} />
              <span>Backend Offline</span>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};
