import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
}

PHASE_COLORS = {
    "under-regularized": "#8fb3d9",
    "useful-compression": "#0f8f7d",
    "robustness-specialized": "#c26a2e",
    "over-compressed": "#b94b4b",
    "unstable": "#8a8175",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _beta_label(value: str) -> str:
    beta = float(value)
    if beta == 0.0:
        return "0"
    if beta < 0.001:
        return f"{beta:.0e}"
    return f"{beta:g}"


def plot_clean_accuracy(metric_rows: list[dict[str, str]], output_path: Path) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metric_rows:
        grouped[row["dataset"]].append(row)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for axis, dataset in zip(axes, ["mnist", "fashion_mnist", "cifar10"]):
        rows = sorted(grouped[dataset], key=lambda row: (row["model"] != "cnn", float(row["beta"])))
        labels = ["CNN" if row["model"] == "cnn" else _beta_label(row["beta"]) for row in rows]
        values = [float(row["test_accuracy"]) * 100 for row in rows]
        colors = ["#8a8175" if row["model"] == "cnn" else "#0f8f7d" for row in rows]
        axis.bar(range(len(values)), values, color=colors, width=0.78)
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Clean accuracy (%)")
    fig.suptitle("PhaseLab clean accuracy across beta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_phase_label_counts(summary: dict[str, Any], output_path: Path) -> None:
    datasets = ["mnist", "fashion_mnist", "cifar10"]
    labels = ["under-regularized", "useful-compression", "robustness-specialized", "unstable", "over-compressed"]
    fig, axis = plt.subplots(figsize=(9, 4.2))
    bottoms = [0] * len(datasets)
    for label in labels:
        values = [summary["datasets"][dataset]["phase_label_counts"].get(label, 0) for dataset in datasets]
        axis.bar([DATASET_LABELS[d] for d in datasets], values, bottom=bottoms, label=label, color=PHASE_COLORS[label])
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set_ylabel("Experiment count")
    axis.set_title("Phase label distribution")
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_auc(phase_rows: list[dict[str, str]], output_path: Path) -> None:
    best: dict[tuple[str, str], tuple[str, float]] = {}
    for row in phase_rows:
        dataset = row["dataset"]
        experiment_id = row["experiment_id"]
        aucs = json.loads(row["robustness_auc"])
        for corruption, value in aucs.items():
            key = (dataset, corruption)
            value = float(value)
            if key not in best or value > best[key][1]:
                best[key] = (experiment_id, value)
    datasets = ["mnist", "fashion_mnist", "cifar10"]
    corruptions = ["gaussian", "salt_pepper", "blur", "contrast"]
    values = [[best[(dataset, corruption)][1] for corruption in corruptions] for dataset in datasets]
    fig, axis = plt.subplots(figsize=(8.5, 3.8))
    image = axis.imshow(values, cmap="YlGnBu", vmin=0, vmax=1)
    axis.set_xticks(range(len(corruptions)))
    axis.set_xticklabels(corruptions, rotation=20, ha="right")
    axis.set_yticks(range(len(datasets)))
    axis.set_yticklabels([DATASET_LABELS[d] for d in datasets])
    for i, dataset in enumerate(datasets):
        for j, corruption in enumerate(corruptions):
            axis.text(j, i, f"{values[i][j]:.3f}", ha="center", va="center", fontsize=9)
    axis.set_title("Best robustness AUC by dataset and corruption")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_phase_figures(result_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = _read_csv(result_root / "phase_metrics.csv")
    phase_rows = _read_csv(result_root / "phase_labels.csv")
    summary = _read_json(result_root / "phase_summary.json")
    plot_clean_accuracy(metric_rows, output_dir / "phaselab_clean_accuracy.png")
    plot_phase_label_counts(summary, output_dir / "phaselab_phase_counts.png")
    plot_robustness_auc(phase_rows, output_dir / "phaselab_robustness_auc.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures for VIB PhaseLab.")
    parser.add_argument("--result-root", type=Path, default=Path("report/phase_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("report/figures"))
    args = parser.parse_args()
    generate_phase_figures(args.result_root, args.output_dir)
    print(f"Wrote PhaseLab figures to {args.output_dir}")


if __name__ == "__main__":
    main()
