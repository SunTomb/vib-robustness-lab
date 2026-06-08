import type { ExperimentIndex, ExperimentPayload, PhaseExperimentPayload, PhaseIndex, PhaseSummary } from './types'

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`)
  }
  return response.json() as Promise<T>
}

export function fetchIndex(): Promise<ExperimentIndex> {
  return getJson<ExperimentIndex>('/api/index')
}

export function fetchExperiment(id: string): Promise<ExperimentPayload> {
  return getJson<ExperimentPayload>(`/api/experiments/${id}`)
}

export function fetchPhaseIndex(): Promise<PhaseIndex> {
  return getJson<PhaseIndex>('/api/phase/index')
}

export function fetchPhaseSummary(): Promise<PhaseSummary> {
  return getJson<PhaseSummary>('/api/phase/summary')
}

export function fetchPhaseExperiment(id: string): Promise<PhaseExperimentPayload> {
  return getJson<PhaseExperimentPayload>(`/api/phase/experiments/${id}`)
}

export function fetchPhaseDiagnostics(): Promise<Record<string, unknown>> {
  return getJson<Record<string, unknown>>('/api/phase/diagnostics')
}
