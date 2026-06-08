import type { ExperimentMetadata } from '../types'

function modelLabel(model: string) {
  return model === 'cnn' ? '普通 CNN' : 'VIB-CNN'
}

export function ExperimentSelector({
  experiments,
  selectedId,
  onSelect,
}: {
  experiments: ExperimentMetadata[]
  selectedId: string
  onSelect: (id: string) => void
}) {
  return (
    <label className="selector">
      选择实验组合
      <select value={selectedId} onChange={(event) => onSelect(event.target.value)}>
        {experiments.map((experiment) => (
          <option key={experiment.id} value={experiment.id}>
            {experiment.dataset} / {modelLabel(experiment.model)} / β={experiment.beta}
          </option>
        ))}
      </select>
    </label>
  )
}
