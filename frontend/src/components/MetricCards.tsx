import { formatPercent } from '../lib/metrics'
import type { Metrics } from '../types'

export function MetricCards({ metrics }: { metrics: Metrics }) {
  return (
    <div className="metric-grid">
      <article className="metric-card primary">
        <span>测试集准确率</span>
        <strong>{formatPercent(metrics.test_accuracy)}</strong>
      </article>
      <article className="metric-card">
        <span>训练-测试差距</span>
        <strong>{formatPercent(metrics.train_test_gap)}</strong>
      </article>
      <article className="metric-card">
        <span>KL proxy</span>
        <strong>{metrics.average_kl.toFixed(3)}</strong>
      </article>
      <article className="metric-card">
        <span>瓶颈系数 β</span>
        <strong>{metrics.beta}</strong>
      </article>
    </div>
  )
}
