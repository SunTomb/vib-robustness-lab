import { describe, expect, it } from 'vitest'
import { betaLabel, formatPhaseMetric, phaseLabelText } from './phaseMetrics'

describe('phaseLabelText', () => {
  it('maps phase labels to Chinese labels', () => {
    expect(phaseLabelText('useful-compression')).toBe('有效压缩')
    expect(phaseLabelText('over-compressed')).toBe('过度压缩')
  })
})

describe('betaLabel', () => {
  it('formats beta values compactly', () => {
    expect(betaLabel(0.001)).toBe('β=0.001')
  })
})

describe('formatPhaseMetric', () => {
  it('formats nullable numeric metrics', () => {
    expect(formatPhaseMetric(0.91234)).toBe('0.9123')
    expect(formatPhaseMetric(null)).toBe('—')
  })
})
