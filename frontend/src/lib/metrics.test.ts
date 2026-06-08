import { describe, expect, it } from 'vitest'
import { compressionInterpretation, needsDatasetSelectionReset } from './metrics'

describe('compressionInterpretation', () => {
  it('describes CNN baseline separately from VIB compression', () => {
    expect(compressionInterpretation(0, 0, 'cnn')).toContain('普通 CNN baseline')
  })
})

describe('needsDatasetSelectionReset', () => {
  it('keeps a selected experiment when it already belongs to the dataset', () => {
    expect(needsDatasetSelectionReset('fashion_mnist_vib_beta_1', 'fashion_mnist')).toBe(false)
  })

  it('resets selection when the current experiment belongs to another dataset', () => {
    expect(needsDatasetSelectionReset('mnist_vib_beta_1', 'fashion_mnist')).toBe(true)
  })
})
