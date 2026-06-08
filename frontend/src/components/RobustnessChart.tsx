import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Curves } from '../types'

export function RobustnessChart({ curves }: { curves: Curves }) {
  return (
    <section className="panel chart-panel">
      <p className="section-kicker">噪声扰动</p>
      <h2>高斯噪声鲁棒性</h2>
      <p className="muted">σ 越大，输入噪声越强；曲线显示模型在扰动下的准确率变化。</p>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={curves.noise} margin={{ top: 12, right: 16, bottom: 8, left: 0 }}>
          <XAxis dataKey="sigma" stroke="#64746f" />
          <YAxis domain={[0, 1]} stroke="#64746f" />
          <Tooltip />
          <Line type="monotone" dataKey="accuracy" stroke="#c26a2e" strokeWidth={3} dot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
    </section>
  )
}
