import { Scatter, ScatterChart, ResponsiveContainer, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import type { LatentPoint } from '../types'

export function LatentPlot({ points }: { points: LatentPoint[] }) {
  return (
    <section className="panel latent-panel">
      <div className="panel-heading">
        <div>
          <p className="section-kicker">表示空间</p>
          <h2>Latent PCA 投影</h2>
        </div>
        <p className="muted">颜色维度对应类别标签，观察不同 β 下表示聚合情况。</p>
      </div>
      <ResponsiveContainer width="100%" height={360}>
        <ScatterChart margin={{ top: 12, right: 18, bottom: 12, left: 0 }}>
          <XAxis dataKey="x" type="number" name="PC1" stroke="#64746f" />
          <YAxis dataKey="y" type="number" name="PC2" stroke="#64746f" />
          <ZAxis dataKey="label" name="标签" />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          <Scatter data={points} fill="#0f8f7d" fillOpacity={0.72} />
        </ScatterChart>
      </ResponsiveContainer>
    </section>
  )
}
