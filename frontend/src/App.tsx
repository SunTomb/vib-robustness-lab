import { useEffect, useMemo, useState } from 'react'
import { fetchExperiment, fetchIndex, fetchPhaseExperiment, fetchPhaseIndex, fetchPhaseSummary } from './api'
import { CompressionLens } from './components/CompressionLens'
import { DatasetTabs } from './components/DatasetTabs'
import { DiagnosisLab } from './components/DiagnosisLab'
import { ExperimentSelector } from './components/ExperimentSelector'
import { InformationPlane } from './components/InformationPlane'
import { InterpretationPanel } from './components/InterpretationPanel'
import { LatentPlot } from './components/LatentPlot'
import { MetricCards } from './components/MetricCards'
import { PhaseMap } from './components/PhaseMap'
import { PhaseOverview } from './components/PhaseOverview'
import { RobustnessChart } from './components/RobustnessChart'
import { datasetLabel, informationPlanePoints, modelLabel, needsDatasetSelectionReset, sortExperiments } from './lib/metrics'
import type { ExperimentIndex, ExperimentPayload, PhaseExperimentPayload, PhaseIndex, PhaseSummary } from './types'

export function App() {
  const [index, setIndex] = useState<ExperimentIndex | null>(null)
  const [selectedDataset, setSelectedDataset] = useState<string>('')
  const [selectedId, setSelectedId] = useState<string>('')
  const [experiments, setExperiments] = useState<Record<string, ExperimentPayload>>({})
  const [phaseIndex, setPhaseIndex] = useState<PhaseIndex | null>(null)
  const [phaseSummary, setPhaseSummary] = useState<PhaseSummary | null>(null)
  const [selectedPhaseId, setSelectedPhaseId] = useState<string>('')
  const [phaseExperiments, setPhaseExperiments] = useState<Record<string, PhaseExperimentPayload>>({})
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchIndex()
      .then(async (payload) => {
        const sorted = sortExperiments(payload.experiments)
        const firstDataset = payload.datasets[0] ?? ''
        const filtered = sorted.filter((experiment) => experiment.dataset === firstDataset)
        const firstId = filtered[0]?.id ?? sorted[0]?.id ?? ''
        setIndex({ ...payload, experiments: sorted })
        setSelectedDataset(firstDataset)
        setSelectedId(firstId)
        if (firstId) {
          const first = await fetchExperiment(firstId)
          setExperiments({ [firstId]: first })
        }
      })
      .catch((err: Error) => setError(err.message))
  }, [])

  useEffect(() => {
    Promise.all([fetchPhaseIndex(), fetchPhaseSummary()])
      .then(([phaseIndexPayload, phaseSummaryPayload]) => {
        setPhaseIndex(phaseIndexPayload)
        setPhaseSummary(phaseSummaryPayload)
        setSelectedPhaseId(phaseIndexPayload.experiments[0]?.id ?? '')
      })
      .catch(() => {
        setPhaseIndex(null)
        setPhaseSummary(null)
      })
  }, [])

  useEffect(() => {
    if (!selectedPhaseId || phaseExperiments[selectedPhaseId]) return
    fetchPhaseExperiment(selectedPhaseId)
      .then((payload) => setPhaseExperiments((current) => ({ ...current, [selectedPhaseId]: payload })))
      .catch(() => undefined)
  }, [selectedPhaseId, phaseExperiments])

  useEffect(() => {
    if (!index || !selectedDataset || !needsDatasetSelectionReset(selectedId, selectedDataset)) return
    const next = index.experiments.find((experiment) => experiment.dataset === selectedDataset)
    if (next) setSelectedId(next.id)
  }, [index, selectedDataset, selectedId])

  useEffect(() => {
    if (!selectedId || experiments[selectedId]) return
    fetchExperiment(selectedId)
      .then((payload) => setExperiments((current) => ({ ...current, [selectedId]: payload })))
      .catch((err: Error) => setError(err.message))
  }, [selectedId, experiments])

  const selected = selectedId ? experiments[selectedId] : null
  const visibleExperiments = index?.experiments.filter((experiment) => experiment.dataset === selectedDataset) ?? []
  const planePoints = useMemo(() => informationPlanePoints(Object.values(experiments)), [experiments])

  if (error) return <main><h1>VIB 实验台</h1><p>{error}</p></main>
  if (!index || !selected) return <main><h1>VIB 实验台</h1><p>正在读取实验 artifacts...</p></main>

  return (
    <main>
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">信息论课程大作业 · 表示压缩实验</p>
          <h1>变分信息瓶颈鲁棒泛化实验台</h1>
          <p>
            调节 β，观察表示压缩、分类精度、KL proxy 与噪声鲁棒性之间的折中。
          </p>
        </div>
        <div className="hero-mark" aria-hidden="true">
          <span>I(X;Z)</span>
          <b>β</b>
          <span>I(Z;Y)</span>
        </div>
      </section>

      {phaseIndex && phaseSummary ? (
        <section className="phaselab-section">
          <PhaseOverview summary={phaseSummary} />
          <PhaseMap
            experiments={phaseIndex.experiments}
            selectedMetric="phase_label"
            selectedId={selectedPhaseId}
            onSelectExperiment={setSelectedPhaseId}
          />
          {selectedPhaseId && phaseExperiments[selectedPhaseId] ? (
            <>
              <CompressionLens experiment={phaseExperiments[selectedPhaseId]} />
              <DiagnosisLab experiment={phaseExperiments[selectedPhaseId]} />
            </>
          ) : null}
        </section>
      ) : (
        <section className="panel phaselab-section">
          <p className="section-kicker">VIB PhaseLab</p>
          <h2>PhaseLab artifacts not generated yet</h2>
          <p className="muted">生成 artifacts/phase 后，这里会显示鲁棒性相变地图与表示诊断入口。</p>
        </section>
      )}

      <section className="workspace">
        <div className="control-strip">
          <div>
            <p className="section-kicker">当前实验</p>
            <h2>{datasetLabel(selected.metadata.dataset)} · {modelLabel(selected.metadata.model)}</h2>
          </div>
          <div className="experiment-controls">
            <DatasetTabs datasets={index.datasets} selectedDataset={selectedDataset} onSelect={setSelectedDataset} />
            <ExperimentSelector experiments={visibleExperiments} selectedId={selectedId} onSelect={setSelectedId} />
          </div>
        </div>
        <MetricCards metrics={selected.metrics} />
        <InterpretationPanel experiment={selected} />
      </section>

      <section className="analysis-grid">
        <InformationPlane points={planePoints} />
        <RobustnessChart curves={selected.curves} />
      </section>
      <LatentPlot points={selected.latent_pca.points} />
    </main>
  )
}
