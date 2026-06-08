import { datasetLabel, modelLabel } from '../lib/metrics'
import { betaLabel, formatPhaseMetric, phaseLabelText } from '../lib/phaseMetrics'
import type { PhaseExperimentPayload } from '../types'

export function CompressionLens({ experiment }: { experiment: PhaseExperimentPayload }) {
  const indicators = experiment.phase_indicators
  return (
    <section className="panel compression-lens">
      <div className="panel-heading">
        <div>
          <p className="section-kicker">Compression Lens</p>
          <h2>{datasetLabel(experiment.metadata.dataset)} · {modelLabel(experiment.metadata.model)} · {betaLabel(experiment.metadata.beta)}</h2>
        </div>
        <p className="muted">阶段：{phaseLabelText(indicators.phase_label)}</p>
      </div>
      <div className="phase-overview-grid">
        <article className="phase-overview-item">
          <span>Clean accuracy</span>
          <strong>{formatPhaseMetric(experiment.metrics.test_accuracy)}</strong>
        </article>
        <article className="phase-overview-item">
          <span>KL proxy</span>
          <strong>{formatPhaseMetric(experiment.metrics.average_kl)}</strong>
        </article>
        <article className="phase-overview-item">
          <span>KL collapse score</span>
          <strong>{formatPhaseMetric(indicators.kl_collapse_score)}</strong>
        </article>
      </div>
      <div className="phase-list">
        {Object.entries(indicators.robustness_auc).map(([corruption, value]) => (
          <p key={corruption}>{corruption}: robustness AUC {formatPhaseMetric(value)}</p>
        ))}
      </div>
    </section>
  )
}
