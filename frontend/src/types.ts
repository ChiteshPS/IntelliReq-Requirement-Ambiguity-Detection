// frontend/src/types.ts — TypeScript Interfaces for IntelliReq

export interface ExplanationDetail {
  reason: string;
  rule_based_evidence: string[];
  ml_confidence_evidence: string;
  decision_policy_note: string;
}

export interface SingleAnalyzeResponse {
  is_valid: boolean;
  requirement: string;
  predicted_class: 'AMBIGUOUS' | 'NOT AMBIGUOUS' | 'NEEDS REVIEW' | 'INVALID';
  confidence_score: number;
  probabilities: {
    Ambiguous: number;
    'Not Ambiguous': number;
  };
  detected_cues: string[];
  missing_specifications: string[];
  suggested_rewrite: string | null;
  rewrite_disclaimer: string;
  explanation: ExplanationDetail;
  linguistic_features: Record<string, any>;
}

export interface BatchItemResponse {
  id: string;
  requirement: string;
  is_valid: boolean;
  predicted_class: 'AMBIGUOUS' | 'NOT AMBIGUOUS' | 'NEEDS REVIEW' | 'SKIPPED';
  confidence_score?: number | null;
  detected_cues: string[];
  missing_specifications: string[];
  suggested_rewrite?: string | null;
  reason: string;
}

export interface BatchSummary {
  total_processed: number;
  ambiguous_count: number;
  not_ambiguous_count: number;
  needs_review_count: number;
  skipped_count: number;
  ambiguity_rate_pct: number;
}

export interface BatchAnalyzeResponse {
  summary: BatchSummary;
  results: BatchItemResponse[];
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  model_loaded: boolean;
  service_ready: boolean;
  error?: string | null;
  model_version?: string | null;
  embedding_model?: string | null;
}

export interface BenchmarkItem {
  Model: string;
  Feature_Type: string;
  Heldout_Acc: number;
  Heldout_Bal_Acc: number;
  Heldout_Macro_F1: number;
  Heldout_F1_Amb: number;
  Heldout_Recall_Amb: number;
  Heldout_Precision_Amb: number;
  Heldout_MCC: number;
  Heldout_Brier: number;
  Heldout_Strict_3Way_Acc: number;
  Heldout_Deferral_Rate: number;
  Heldout_Selective_Acc: number;
  Challenge_Acc: number;
  Challenge_Bal_Acc: number;
  Challenge_Macro_F1: number;
  Challenge_F1_Amb: number;
  Challenge_Recall_Amb: number;
  Challenge_Precision_Amb: number;
  Challenge_MCC: number;
  Challenge_Brier: number;
  Challenge_Strict_3Way_Acc: number;
  Challenge_Deferral_Rate: number;
  Challenge_Selective_Acc: number;
  Challenge_Borderline_Review_Rate: number;
  Adversarial_Acc: number;
  Adversarial_Macro_F1: number;
  Adversarial_Brier: number;
}

export interface CrossDatasetBenchmarkItem {
  Experiment: string;
  Model: string;
  accuracy: number;
  balanced_accuracy: number;
  precision_amb: number;
  recall_amb: number;
  f1_amb: number;
  precision_clear: number;
  recall_clear: number;
  f1_clear: number;
  macro_f1: number;
  mcc: number;
  brier_loss: number;
  tn: number;
  fp: number;
  fn: number;
  tp: number;
  amb_error_rate_pct: number;
  strict_3way_acc: number;
  deferral_rate: number;
  selective_acc: number;
}

export interface FaultProneSRSAudit {
  dataset_name: string;
  official_source: string;
  associated_paper: string;
  total_rows: number;
  missing_values: number;
  exact_duplicates: number;
  normalized_duplicates: number;
  conflicting_labels_6class: number;
  conflicting_labels_binary: number;
  classes_6: Record<string, number>;
  binary_distribution: {
    ambiguous: number;
    not_ambiguous: number;
  };
}

export interface FaultProneSRSBenchmarkItem {
  task: string;
  model: string;
  accuracy: number;
  balanced_accuracy: number;
  macro_f1: number;
  weighted_f1?: number;
  mcc?: number;
  precision_amb?: number;
  recall_amb?: number;
  f1_amb?: number;
  amb_error_rate_pct?: number;
  per_class?: Record<string, { precision: number; recall: number; f1: number }>;
  confusion_matrix?: number[][];
  brier_loss?: number;
  tp?: number;
  fn?: number;
  fp?: number;
  tn?: number;
  eval_set?: string;
}

export interface ModelInfoResponse {
  model_name: string;
  model_version: string;
  architecture: string;
  embedding_model: string;
  embedding_dimension: number;
  total_features: number;
  handcrafted_feature_count: number;
  classifier: string;
  calibration_method: string;
  thresholds: {
    upper_ambiguous_threshold: number;
    lower_clear_threshold: number;
    needs_review_band_start: number;
    needs_review_band_end: number;
  };
  dataset_provenance: {
    dataset_name: string;
    total_training_samples: number;
    validation_samples: number;
    heldout_test_samples: number;
    leakage_prevention: string;
    labeling_strategy: string;
    three_tier_policy: string;
  };
  target_accuracy_claim: {
    requested_target: string;
    verified_on_heldout_clean_test: string;
    verified_on_challenge_out_of_domain: string;
    status: string;
  };
  verified_benchmarks: BenchmarkItem[];
  heldout_reading: Record<string, any>;
  challenge_reading: Record<string, any>;
  adversarial_reading: Record<string, any>;
  external_datasets?: Record<string, any>;
  cross_dataset_benchmarks?: CrossDatasetBenchmarkItem[];
  fault_prone_srs_benchmarks?: FaultProneSRSBenchmarkItem[];
  fault_prone_srs_audit?: FaultProneSRSAudit;
  fault_prone_srs_cross_benchmarks?: FaultProneSRSBenchmarkItem[];
}
