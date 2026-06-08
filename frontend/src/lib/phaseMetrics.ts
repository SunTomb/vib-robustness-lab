export function phaseLabelText(label: string): string {
  const labels: Record<string, string> = {
    'under-regularized': '压缩不足',
    'useful-compression': '有效压缩',
    'robustness-specialized': '鲁棒特化',
    'over-compressed': '过度压缩',
    unstable: '不稳定',
  }
  return labels[label] ?? label
}

export function betaLabel(beta: number): string {
  return `β=${beta.toPrecision(4).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1')}`
}

export function formatPhaseMetric(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—'
  return value.toFixed(4)
}
