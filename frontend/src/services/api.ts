// frontend/src/services/api.ts — API Service for IntelliReq

import type {
  SingleAnalyzeResponse,
  BatchAnalyzeResponse,
  HealthResponse,
  ModelInfoResponse,
} from '../types';

const API_BASE = '/api';

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed with HTTP ${res.status}`);
  }
  return res.json();
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${API_BASE}/model-info`);
  if (!res.ok) {
    throw new Error(`Failed to fetch model info: HTTP ${res.status}`);
  }
  return res.json();
}

export async function analyzeRequirement(text: string): Promise<SingleAnalyzeResponse> {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ requirement: text }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Analysis failed: HTTP ${res.status}`);
  }
  return res.json();
}

export async function analyzeBatchRequirements(
  requirements: { id?: string; requirement: string }[]
): Promise<BatchAnalyzeResponse> {
  const res = await fetch(`${API_BASE}/analyze-batch`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ requirements }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Batch analysis failed: HTTP ${res.status}`);
  }
  return res.json();
}

export async function analyzeFile(file: File): Promise<BatchAnalyzeResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/analyze-file`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `File upload analysis failed: HTTP ${res.status}`);
  }
  return res.json();
}
