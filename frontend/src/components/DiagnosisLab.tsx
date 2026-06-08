import { formatPhaseMetric } from '../lib/phaseMetrics'
import type { PhaseExperimentPayload } from '../types'

export function DiagnosisLab({ experiment }: { experiment: PhaseExperimentPayload }) {
  const geometry = experiment.latent_geometry
  const driftRows = experiment.latent_drift?.rows ?? []
  return (
    <section className="panel diagnosis-lab">
      <div className="panel-heading">
        <div>
          <p className="section-kicker">Diagnosis Lab</p>
          <h2>表示空间诊断</h2>
        </div>
        <p className="muted">用 latent geometry 和 drift 解释当前 β 的压缩状态。</p>
      </div>
      <div className="phase-overview-grid">
        <article className="phase-overview-item">
          <span>类内方差</span>
          <strong>{formatPhaseMetric(geometry.intra_class_variance)}</strong>
        </article>
        <article className="phase-overview-item">
          <span>类间距离</span>
          <strong>{formatPhaseMetric(geometry.inter_class_distance)}</strong>
        </article>
        <article className="phase-overview-item">
          <span>Fisher ratio</span>
          <strong>{formatPhaseMetric(geometry.fisher_ratio)}</strong>
        </article>
        <article className="phase-overview-item">
          <span>Silhouette</span>
          <strong>{formatPhaseMetric(geometry.silhouette_score)}</strong>
        </article>
      </div>
      <div className="phase-list">
        {driftRows.slice(0, 8).map((row) => (
          <p key={`${row.corruption}-${row.severity}-${row.label}`}>
            {row.corruption} severity {row.severity} · class {row.label}: mean drift {formatPhaseMetric(row.mean_drift)}
          </p>
        ))}
        {driftRows.length === 0 ? <p className="muted">当前 artifact 暂无 latent drift 记录。</p> : null}
      </div>
    </section>
  )
}
