import { datasetLabel } from '../lib/metrics'
import { formatPhaseMetric } from '../lib/phaseMetrics'
import type { PhaseSummary } from '../types'

export function PhaseOverview({ summary }: { summary: PhaseSummary }) {
  const entries = Object.entries(summary.datasets)
  return (
    <section className="phase-overview panel">
      <div className="panel-heading">
        <div>
          <p className="section-kicker">VIB PhaseLab</p>
          <h2>鲁棒性相变总览</h2>
        </div>
        <p className="muted">自动识别每个数据集的 clean 最优实验、鲁棒性优势和压缩阶段分布。</p>
      </div>
      <div className="phase-overview-grid">
        {entries.map(([dataset, item]) => (
          <article key={dataset} className="phase-overview-item">
            <span>{datasetLabel(dataset)}</span>
            <strong>{formatPhaseMetric(item.best_clean_accuracy)}</strong>
            <p>最佳 clean：{item.best_clean_experiment}</p>
            <p>阶段计数：{Object.entries(item.phase_label_counts).map(([label, count]) => `${label} ${count}`).join(' · ')}</p>
          </article>
        ))}
      </div>
    </section>
  )
}
