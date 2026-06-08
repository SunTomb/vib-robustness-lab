import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
                "train_test_gap": float(metrics["train_test_gap"]),
                "average_kl": float(metrics["average_kl"]),
            }
        )
    return rows


def build_robustness_rows(root: Path) -> list[dict[str, Any]]:
    index = _read_json(root / "index.json")
    rows: list[dict[str, Any]] = []
    for experiment in index["experiments"]:
        metrics = _read_json(root / experiment["path"] / "metrics.json")
        for sigma, accuracy in metrics["noise_accuracy"].items():
            rows.append(
                {
                    "experiment_id": experiment["id"],
                    "dataset": experiment["dataset"],
                    "model": experiment["model"],
                    "beta": float(experiment["beta"]),
                    "sigma": float(sigma),
                    "accuracy": float(accuracy),
                }
            )
    return rows


def derive_key_findings(metric_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[row["dataset"]].append(row)
    findings: dict[str, dict[str, Any]] = {}
    for dataset, rows in grouped.items():
        vib_rows = [row for row in rows if row["model"] == "vib"]
        best_clean = max(rows, key=lambda row: row["test_accuracy"])
        best_vib = max(vib_rows, key=lambda row: row["test_accuracy"])
        lowest_kl_vib = min(vib_rows, key=lambda row: row["average_kl"])
        findings[dataset] = {
            "best_clean_experiment": best_clean["experiment_id"],
            "best_clean_accuracy": best_clean["test_accuracy"],
            "best_vib_experiment": best_vib["experiment_id"],
            "best_vib_beta": best_vib["beta"],
            "lowest_kl_vib_experiment": lowest_kl_vib["experiment_id"],
            "lowest_kl_vib_beta": lowest_kl_vib["beta"],
        }
    return findings


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_findings(path: Path, findings: dict[str, dict[str, Any]]) -> None:
    lines = ["# 实验关键发现", ""]
    for dataset, item in findings.items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- 最高 clean accuracy：`{item['best_clean_experiment']}`，准确率 `{item['best_clean_accuracy']:.4f}`。",
                f"- 最优 VIB β：`{item['best_vib_beta']}`，实验 `{item['best_vib_experiment']}`。",
                f"- 最低 KL 的 VIB 实验：`{item['lowest_kl_vib_experiment']}`，β `{item['lowest_kl_vib_beta']}`。",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _dataset_rows(rows: list[dict[str, Any]], dataset: str, model: str | None = None) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["dataset"] == dataset]
    if model is not None:
        selected = [row for row in selected if row["model"] == model]
    return sorted(selected, key=lambda row: row["beta"])


def write_figures(metric_rows: list[dict[str, Any]], robustness_rows: list[dict[str, Any]], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for dataset in sorted({row["dataset"] for row in metric_rows}):
        vib_rows = _dataset_rows(metric_rows, dataset, "vib")
        betas = [row["beta"] for row in vib_rows]
        accuracy = [row["test_accuracy"] for row in vib_rows]
        kl = [row["average_kl"] for row in vib_rows]

        plt.figure(figsize=(6, 4))
        plt.plot(betas, accuracy, marker="o")
        plt.xscale("symlog", linthresh=1e-4)
        plt.xlabel("beta")
        plt.ylabel("clean accuracy")
        plt.title(f"{dataset}: beta vs clean accuracy")
        plt.tight_layout()
        plt.savefig(figure_dir / f"{dataset}_beta_accuracy.svg")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(betas, kl, marker="o", color="#0f8f7d")
        plt.xscale("symlog", linthresh=1e-4)
        plt.xlabel("beta")
        plt.ylabel("average KL proxy")
        plt.title(f"{dataset}: beta vs KL proxy")
        plt.tight_layout()
        plt.savefig(figure_dir / f"{dataset}_beta_kl.svg")
        plt.close()

        plt.figure(figsize=(6, 4))
        for beta in betas:
            rows_for_beta = [row for row in robustness_rows if row["dataset"] == dataset and row["model"] == "vib" and row["beta"] == beta]
            rows_for_beta = sorted(rows_for_beta, key=lambda row: row["sigma"])
            plt.plot([row["sigma"] for row in rows_for_beta], [row["accuracy"] for row in rows_for_beta], marker="o", label=f"β={beta:g}")
        plt.xlabel("Gaussian noise sigma")
        plt.ylabel("accuracy")
        plt.title(f"{dataset}: robustness curves")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figure_dir / f"{dataset}_robustness.svg")
        plt.close()

    plt.figure(figsize=(6, 4))
    for dataset in sorted({row["dataset"] for row in metric_rows}):
        rows = _dataset_rows(metric_rows, dataset, "vib")
        plt.scatter([row["average_kl"] for row in rows], [row["test_accuracy"] for row in rows], label=dataset)
    plt.xlabel("average KL proxy")
    plt.ylabel("clean accuracy")
    plt.title("Information plane")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "information_plane.svg")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize VIB full experiment artifacts.")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/full"))
    parser.add_argument("--output-dir", type=Path, default=Path("report/results"))
    args = parser.parse_args()
    metric_rows = build_metric_rows(args.artifact_root)
    robustness_rows = build_robustness_rows(args.artifact_root)
    findings = derive_key_findings(metric_rows)
    _write_csv(args.output_dir / "metrics_summary.csv", metric_rows)
    _write_csv(args.output_dir / "robustness_summary.csv", robustness_rows)
    _write_findings(args.output_dir / "key_findings.md", findings)
    write_figures(metric_rows, robustness_rows, Path("report/figures"))
    print(f"Wrote {len(metric_rows)} metric rows and {len(robustness_rows)} robustness rows to {args.output_dir}")


if __name__ == "__main__":
    main()
