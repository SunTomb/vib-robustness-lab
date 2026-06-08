export type ExperimentMetadata = {
  id: string
  dataset: string
  model: 'cnn' | 'vib'
  beta: number
  path: string
}

export type ExperimentIndex = {
  datasets: string[]
  experiments: ExperimentMetadata[]
}

export type Metrics = {
  dataset: string
  model: string
  beta: number
  train_accuracy: number
  test_accuracy: number
  train_test_gap: number
  average_kl: number
  noise_accuracy: Record<string, number>
}

export type Curves = {
  noise: Array<{ sigma: number; accuracy: number }>
}

export type LatentPoint = {
  x: number
  y: number
  label: number
}

export type ExperimentPayload = {
  metadata: ExperimentMetadata
  metrics: Metrics
  curves: Curves
  latent_pca: { points: LatentPoint[] }
}

export type PhaseExperimentMetadata = ExperimentMetadata

export type PhaseIndex = {
  datasets: string[]
  experiments: PhaseExperimentMetadata[]
}

export type RobustnessRow = {
  corruption: string
  severity: number
  accuracy: number
  mean_confidence: number
  mean_entropy: number
}

export type LatentGeometry = {
  intra_class_variance: number
  inter_class_distance: number | null
  fisher_ratio: number | null
  silhouette_score: number | null
  per_class: Array<{ label: number; intra_variance: number; center_norm: number }>
}

export type PhaseIndicators = {
  robustness_auc: Record<string, number>
  normalized_robustness_auc: Record<string, number | null>
  compression_benefit_index: Record<string, number | null>
  kl_collapse_score: number | null
  phase_label: string
  over_compression_flag: boolean
}

export type LatentDriftRow = {
  corruption: string
  severity: number
  label: number
  mean_drift: number
  median_drift: number
}

export type PhaseExperimentPayload = {
  metadata: PhaseExperimentMetadata
  metrics: Omit<Metrics, 'noise_accuracy'>
  robustness: { rows: RobustnessRow[] }
  latent_geometry: LatentGeometry
  phase_indicators: PhaseIndicators
  latent_drift?: { rows: LatentDriftRow[] }
  latent_pca?: { points: LatentPoint[] }
  confusion_matrix?: { matrix: number[][] }
}

export type PhaseSummary = {
  datasets: Record<
    string,
    {
      best_clean_experiment: string
      best_clean_accuracy: number
      best_robust_by_corruption: Record<string, string>
      phase_label_counts: Record<string, number>
    }
  >
}
