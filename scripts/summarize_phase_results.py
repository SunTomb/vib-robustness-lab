import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vib_project.phase_metrics import build_phase_indicators, compute_normalized_robustness_auc, compute_robustness_auc


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_metric_rows(root: Path) -> list[dict[str, Any]]:
    index = _read_json(root / "index.json")
    rows: list[dict[str, Any]] = []
    for experiment in index["experiments"]:
        metrics = _read_json(root / experiment["path"] / "metrics.json")
        rows.append(
            {
                "experiment_id": experiment["id"],
                "dataset": experiment["dataset"],
                "model": experiment["model"],
                "beta": float(experiment["beta"]),
                "test_accuracy": float(metrics["test_accuracy"]),
                "average_kl": float(metrics.get("average_kl", 0.0)),
            }
        )
    return rows


def build_robustness_rows(root: Path) -> list[dict[str, Any]]:
    index = _read_json(root / "index.json")
    rows: list[dict[str, Any]] = []
    for experiment in index["experiments"]:
        robustness = _read_json(root / experiment["path"] / "robustness.json")
        for row in robustness["rows"]:
            rows.append(
                {
                    "experiment_id": experiment["id"],
                    "dataset": experiment["dataset"],
                    "model": experiment["model"],
                    "beta": float(experiment["beta"]),
                    "corruption": row["corruption"],
                    "severity": float(row["severity"]),
                    "accuracy": float(row["accuracy"]),
                    "mean_confidence": float(row.get("mean_confidence", 0.0)),
                    "mean_entropy": float(row.get("mean_entropy", 0.0)),
                }
            )
    return rows


def build_phase_rows(root: Path) -> list[dict[str, Any]]:
    index = _read_json(root / "index.json")
    rows: list[dict[str, Any]] = []
    for experiment in index["experiments"]:
        phase = _read_json(root / experiment["path"] / "phase_indicators.json")
        rows.append(
            {
                "experiment_id": experiment["id"],
                "dataset": experiment["dataset"],
                "model": experiment["model"],
                "beta": float(experiment["beta"]),
                "phase_label": phase.get("phase_label", "unstable"),
                "over_compression_flag": bool(phase.get("over_compression_flag", False)),
                "robustness_auc": phase.get("robustness_auc", {}),
            }
        )
    return rows


def refresh_phase_indicators(root: Path) -> None:
    index = _read_json(root / "index.json")
    records = []
    for experiment in index["experiments"]:
        experiment_dir = root / experiment["path"]
        metrics = _read_json(experiment_dir / "metrics.json")
        robustness = _read_json(experiment_dir / "robustness.json")
        rows = robustness["rows"]
        clean_accuracy = float(metrics["test_accuracy"])
        records.append(
            {
                "metadata": experiment,
                "directory": experiment_dir,
                "metrics": metrics,
                "robustness_rows": rows,
                "normalized_auc": compute_normalized_robustness_auc(compute_robustness_auc(rows), clean_accuracy),
            }
        )

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[str(record["metadata"]["dataset"])].append(record)

    for dataset_records in by_dataset.values():
        baseline_record = next((record for record in dataset_records if record["metadata"]["model"] == "cnn"), dataset_records[0])
        beta0_record = next(
            (
                record
                for record in dataset_records
                if record["metadata"]["model"] == "vib" and abs(float(record["metadata"]["beta"])) < 1e-12
            ),
            baseline_record,
        )
        best_accuracy = max(float(record["metrics"]["test_accuracy"]) for record in dataset_records)
        baseline_normalized_auc = baseline_record["normalized_auc"]
        beta0_kl = float(beta0_record["metrics"].get("average_kl", 0.0))
        for record in dataset_records:
            metrics = record["metrics"]
            phase_indicators = build_phase_indicators(
                record["robustness_rows"],
                clean_accuracy=float(metrics["test_accuracy"]),
                baseline_normalized_auc=baseline_normalized_auc,
                average_kl=float(metrics.get("average_kl", 0.0)),
                beta0_kl=beta0_kl,
                best_accuracy=best_accuracy,
            )
            _write_json(record["directory"] / "phase_indicators.json", phase_indicators)


def build_phase_summary(root: Path) -> dict[str, Any]:
    metric_rows = build_metric_rows(root)
    phase_rows = build_phase_rows(root)
    grouped_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    phase_by_experiment = {row["experiment_id"]: row for row in phase_rows}
    for row in metric_rows:
        grouped_metrics[row["dataset"]].append(row)

    datasets: dict[str, Any] = {}
    for dataset, rows in grouped_metrics.items():
        best_clean = max(rows, key=lambda row: row["test_accuracy"])
        corruptions = sorted(
            {
                corruption
                for row in phase_rows
                if row["dataset"] == dataset
                for corruption in row["robustness_auc"].keys()
            }
        )
        best_robust_by_corruption: dict[str, str] = {}
        for corruption in corruptions:
            candidates = [row for row in phase_rows if row["dataset"] == dataset and corruption in row["robustness_auc"]]
            best_robust = max(candidates, key=lambda row: float(row["robustness_auc"][corruption]))
            best_robust_by_corruption[corruption] = best_robust["experiment_id"]
        counts = Counter(phase_by_experiment[row["experiment_id"]]["phase_label"] for row in rows if row["experiment_id"] in phase_by_experiment)
        datasets[dataset] = {
            "best_clean_experiment": best_clean["experiment_id"],
            "best_clean_accuracy": best_clean["test_accuracy"],
            "best_robust_by_corruption": best_robust_by_corruption,
            "phase_label_counts": dict(counts),
        }
    return {"datasets": datasets}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    serializable_rows = []
    for row in rows:
        serializable_rows.append({key: json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value for key, value in row.items()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable_rows[0].keys()))
        writer.writeheader()
        writer.writerows(serializable_rows)


def _write_findings(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# VIB PhaseLab 关键发现", ""]
    for dataset, item in summary["datasets"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- 最佳 clean experiment：`{item['best_clean_experiment']}`，accuracy `{item['best_clean_accuracy']:.4f}`。",
                f"- 各 corruption 最佳 robust experiment：`{item['best_robust_by_corruption']}`。",
                f"- phase label 计数：`{item['phase_label_counts']}`。",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")



def write_phase_result_exports(artifact_root: Path, output_dir: Path) -> tuple[int, int, int]:
    refresh_phase_indicators(artifact_root)
    metric_rows = build_metric_rows(artifact_root)
    robustness_rows = build_robustness_rows(artifact_root)
    phase_rows = build_phase_rows(artifact_root)
    summary = build_phase_summary(artifact_root)
    _write_csv(output_dir / "phase_metrics.csv", metric_rows)
    _write_csv(output_dir / "phase_robustness.csv", robustness_rows)
    _write_csv(output_dir / "phase_labels.csv", phase_rows)
    _write_json(output_dir / "phase_summary.json", summary)
    _write_json(artifact_root / "summary.json", summary)
    _write_findings(output_dir / "key_findings.md", summary)
    return len(metric_rows), len(robustness_rows), len(phase_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize VIB PhaseLab artifacts.")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/phase"))
    parser.add_argument("--output-dir", type=Path, default=Path("report/phase_results"))
    args = parser.parse_args()
    metric_count, robustness_count, phase_count = write_phase_result_exports(args.artifact_root, args.output_dir)
    print(f"Wrote {metric_count} metric rows, {robustness_count} robustness rows, and {phase_count} phase rows to {args.output_dir}")


if __name__ == "__main__":
    main()
