import type { ExperimentMetadata, ExperimentPayload } from '../types'

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`
}

export function informationPlanePoints(experiments: ExperimentPayload[]) {
  return experiments.map((experiment) => ({
    id: experiment.metadata.id,
    beta: experiment.metadata.beta,
    kl: experiment.metrics.average_kl,
    accuracy: experiment.metrics.test_accuracy,
    model: experiment.metadata.model,
  }))
}

export function datasetLabel(dataset: string): string {
  if (dataset === 'fashion_mnist') return 'Fashion-MNIST'
  if (dataset === 'mnist') return 'MNIST'
  return dataset
}

export function modelLabel(model: string): string {
  return model === 'cnn' ? '普通 CNN' : 'VIB-CNN'
}

export function compressionInterpretation(beta: number, kl: number, model = 'vib'): string {
  if (model === 'cnn') return '普通 CNN baseline 不包含 KL 瓶颈项，KL proxy 记为 0 仅用于和 VIB-CNN 对照。'
  if (beta >= 1 || kl < 0.1) return '压缩非常强，表示几乎被推向先验，可能已经损失任务信息。'
  if (beta >= 0.1) return '压缩较强，KL proxy 明显下降，适合观察鲁棒性与精度的折中。'
  if (beta > 0) return '压缩适中，通常能保留分类信息，同时减少输入细节依赖。'
  return '未施加 KL 惩罚，表示会保留更多输入信息，也可能保留噪声。'
}

export function needsDatasetSelectionReset(selectedId: string, selectedDataset: string): boolean {
  return selectedId !== '' && !selectedId.startsWith(`${selectedDataset}_`)
}

export function sortExperiments(experiments: ExperimentMetadata[]): ExperimentMetadata[] {
  return [...experiments].sort((a, b) => {
    if (a.dataset !== b.dataset) return a.dataset.localeCompare(b.dataset)
    if (a.model !== b.model) return a.model.localeCompare(b.model)
    return a.beta - b.beta
  })
}
