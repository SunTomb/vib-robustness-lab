import { Scatter, ScatterChart, ResponsiveContainer, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'

export type PlanePoint = {
  id: string
  beta: number
  kl: number
  accuracy: number
  model: string
}

export function InformationPlane({ points }: { points: PlanePoint[] }) {
  return (
    <section className="panel chart-panel">
      <p className="section-kicker">压缩—预测平面</p>
      <h2>信息平面</h2>
      <p className="muted">横轴为 KL proxy，纵轴为测试准确率；每个点对应一个 β 设置。</p>
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart margin={{ top: 12, right: 16, bottom: 8, left: 0 }}>
          <XAxis dataKey="kl" type="number" name="KL proxy" stroke="#64746f" />
          <YAxis dataKey="accuracy" type="number" name="准确率" domain={[0, 1]} stroke="#64746f" />
          <ZAxis dataKey="beta" name="β" />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          <Scatter data={points} fill="#0f8f7d" />
        </ScatterChart>
      </ResponsiveContainer>
    </section>
  )
}
