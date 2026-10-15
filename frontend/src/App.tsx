// frontend/src/App.tsx — IntelliReq Enterprise Application (Power BI Theme)
import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { Overview } from './components/Overview';
import { SingleAnalyzer } from './components/SingleAnalyzer';
import { BatchAnalyzer } from './components/BatchAnalyzer';
import { ModelEvaluation } from './components/ModelEvaluation';
import { SystemGuide } from './components/SystemGuide';
import type { HealthResponse, ModelInfoResponse } from './types';
import { checkHealth, getModelInfo } from './services/api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'single' | 'batch' | 'eval' | 'guide'>('overview');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      setHealthLoading(true);
      try {
        const data = await checkHealth();
        setHealth(data);
      } catch (err) {
        console.warn('Health check warning:', err);
        setHealth(null);
      } finally {
        setHealthLoading(false);
      }
    };

    const fetchModel = async () => {
      try {
        const data = await getModelInfo();
        setModelInfo(data);
      } catch (err) {
        console.warn('Model info fetch error:', err);
      }
    };

    fetchHealth();
    fetchModel();

    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-layout">
      {/* Power BI Executive Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
        healthLoading={healthLoading}
      />

      {/* Main Canvas Area */}
      <div className="report-canvas">
        <Header
          activeTab={activeTab}
          health={health}
          healthLoading={healthLoading}
        />

        <main className="canvas-content" id="main-content">
          {activeTab === 'overview' && (
            <Overview onNavigate={setActiveTab} modelInfo={modelInfo} />
          )}
          {activeTab === 'single' && <SingleAnalyzer />}
          {activeTab === 'batch' && <BatchAnalyzer />}
          {activeTab === 'eval' && <ModelEvaluation />}
          {activeTab === 'guide' && <SystemGuide />}
        </main>
      </div>
    </div>
  );
};

export default App;
