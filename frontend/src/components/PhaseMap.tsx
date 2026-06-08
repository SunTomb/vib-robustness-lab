import { datasetLabel, modelLabel } from '../lib/metrics'
import { betaLabel } from '../lib/phaseMetrics'
import type { PhaseExperimentMetadata } from '../types'

export function PhaseMap({
  experiments,
  selectedMetric,
  selectedId,
  onSelectExperiment,
}: {
  experiments: PhaseExperimentMetadata[]
  selectedMetric: string
  selectedId: string
  onSelectExperiment: (id: string) => void
}) {
  return (
    <section className="phase-map panel">
      <div className="panel-heading">
        <div>
          <p className="section-kicker">Phase Map</p>
          <h2>β 相变地图</h2>
        </div>
        <p className="muted">当前指标：{selectedMetric}。点击任一实验进入诊断视图。</p>
      </div>
      <div className="phase-map-grid">
        {experiments.map((experiment) => (
          <button
            key={experiment.id}
            className={experiment.id === selectedId ? 'phase-cell active' : 'phase-cell'}
            type="button"
            onClick={() => onSelectExperiment(experiment.id)}
          >
            <span>{datasetLabel(experiment.dataset)}</span>
            <strong>{modelLabel(experiment.model)}</strong>
            <em>{betaLabel(experiment.beta)}</em>
          </button>
        ))}
      </div>
    </section>
  )
}
