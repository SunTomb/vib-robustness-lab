import { datasetLabel } from '../lib/metrics'

export function DatasetTabs({
  datasets,
  selectedDataset,
  onSelect,
}: {
  datasets: string[]
  selectedDataset: string
  onSelect: (dataset: string) => void
}) {
  return (
    <div className="dataset-tabs" aria-label="数据集筛选">
      {datasets.map((dataset) => (
        <button
          key={dataset}
          className={dataset === selectedDataset ? 'active' : ''}
          type="button"
          onClick={() => onSelect(dataset)}
        >
          {datasetLabel(dataset)}
        </button>
      ))}
    </div>
  )
}
