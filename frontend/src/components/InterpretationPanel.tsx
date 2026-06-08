import { compressionInterpretation, datasetLabel, modelLabel } from '../lib/metrics'
import type { ExperimentPayload } from '../types'

export function InterpretationPanel({ experiment }: { experiment: ExperimentPayload }) {
  const metrics = experiment.metrics
  return (
    <section className="interpretation-panel">
      <p className="section-kicker">实验解读</p>
      <h2>{datasetLabel(metrics.dataset)} · {modelLabel(metrics.model)}</h2>
      <p>{compressionInterpretation(metrics.beta, metrics.average_kl, metrics.model)}</p>
      <dl>
        <div>
          <dt>β</dt>
          <dd>{metrics.beta}</dd>
        </div>
        <div>
          <dt>KL proxy</dt>
          <dd>{metrics.average_kl.toFixed(4)}</dd>
        </div>
        <div>
          <dt>σ=0.4 准确率</dt>
          <dd>{((metrics.noise_accuracy['0.4'] ?? 0) * 100).toFixed(2)}%</dd>
        </div>
      </dl>
    </section>
  )
}
