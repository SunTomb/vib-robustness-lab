import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _metric(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"


def _dataset_title(dataset: str) -> str:
    labels = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}
    return labels.get(dataset, dataset)


def _experiment_payload(phase_root: Path, experiment_id: str, index: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    experiment = next(item for item in index["experiments"] if item["id"] == experiment_id)
    experiment_dir = phase_root / experiment["path"]
    return experiment, _read_json(experiment_dir / "metrics.json"), _read_json(experiment_dir / "phase_indicators.json")


def generate_phaselab_report(phase_root: Path, result_root: Path, output_path: Path) -> None:
    index = _read_json(phase_root / "index.json")
    summary = _read_json(result_root / "phase_summary.json")
    lines = [
        "# VIB PhaseLab：表示压缩的鲁棒性相变与诊断系统",
        "",
        "## 摘要",
        "",
        (
            "本报告基于真实 `artifacts/phase` 离线实验结果，扩展原 VIB 复现实验为 PhaseLab："
            "在多数据集、多 β、多 corruption 条件下观察表示压缩从有效正则化到过度压缩的相变现象。"
            "文中 accuracy、KL proxy、robustness AUC、compression benefit index 和 phase label 均从实验 artifact 读取，未手工编造数值。"
        ),
        "",
        "## 1 研究问题与方法",
        "",
        (
            "PhaseLab 关注的问题是：VIB 的压缩何时是 useful compression，何时变成 destructive over-compression。"
            "实验以 KL(q(z|x)||p(z)) 作为 I(X;Z) 的变分上界 proxy，以 clean accuracy 和 corruption accuracy 作为任务信息与鲁棒性的经验 proxy。"
        ),
        "",
        "Phase indicators 使用同一数据集内的 CNN baseline 作为 normalized robustness AUC 参考，"
        "使用 VIB β=0 作为 KL collapse 参考，并用同数据集最佳 clean accuracy 判断 clean accuracy drop。",
        "",
        "## 2 实验矩阵",
        "",
        f"本轮 PhaseLab artifacts 覆盖 {len(index['datasets'])} 个数据集、{len(index['experiments'])} 个实验配置。",
        "数据集包括：" + "、".join(_dataset_title(dataset) for dataset in index["datasets"]) + "。",
        "每个实验在 Gaussian、salt-and-pepper、blur、contrast corruption 与多个 severity 上进行评测。",
        "",
        "## 3 结果概览",
        "",
    ]

    for dataset, dataset_summary in summary["datasets"].items():
        best_id = dataset_summary["best_clean_experiment"]
        best_exp, best_metrics, best_phase = _experiment_payload(phase_root, best_id, index)
        lines.extend(
            [
                f"### {_dataset_title(dataset)}",
                "",
                (
                    f"Clean accuracy 最佳实验为 `{best_id}`，模型为 `{best_exp['model']}`，β={float(best_exp['beta']):g}，"
                    f"test accuracy 为 {_percent(float(dataset_summary['best_clean_accuracy']))}，"
                    f"average KL proxy 为 {_metric(float(best_metrics.get('average_kl', 0.0)))}。"
                ),
                (
                    f"该实验的 phase label 为 `{best_phase.get('phase_label', 'unstable')}`，"
                    f"KL collapse score 为 {_metric(best_phase.get('kl_collapse_score'))}。"
                ),
                "",
                "各 corruption 的 robustness AUC 最优实验如下：",
                "",
            ]
        )
        for corruption, experiment_id in dataset_summary["best_robust_by_corruption"].items():
            exp, metrics, phase = _experiment_payload(phase_root, experiment_id, index)
            robustness_auc = phase.get("robustness_auc", {}).get(corruption)
            cbi = phase.get("compression_benefit_index", {}).get(corruption)
            lines.append(
                f"- `{corruption}`：`{experiment_id}`，模型 `{exp['model']}`，β={float(exp['beta']):g}，"
                f"robustness AUC {_metric(robustness_auc)}，CBI {_metric(cbi)}，clean accuracy {_percent(float(metrics['test_accuracy']))}。"
            )
        lines.extend(
            [
                "",
                f"Phase label 分布：`{dataset_summary['phase_label_counts']}`。",
                "",
            ]
        )

    lines.extend(
        [
            "## 4 PhaseLab 系统实现",
            "",
            "离线实验层生成 `metrics.json`、`robustness.json`、`latent_geometry.json`、`latent_drift.json` 和 `phase_indicators.json`。",
            "FastAPI 后端只读取 artifacts，不启动训练；React/Vite 前端显示 phase overview、β phase map、compression lens 和 diagnosis lab。",
            "这种 artifact-driven 结构使课程展示页面可复现、可部署，并与 GPU 训练解耦。",
            "",
            "## 5 局限性",
            "",
            "KL 仍是 variational proxy，不是精确互信息估计；phase label 是用于比较实验现象的诊断标签，不应解释为理论定理。",
            "CIFAR-10 结果受当前小型 CNN/VIB-CNN 结构与训练轮数限制，更适合作为相变诊断样例，而不是最优分类性能 benchmark。",
            "",
            "## 6 结论",
            "",
            "PhaseLab 将原先的 VIB 复现扩展为可诊断的实验系统：它不仅展示 β 是否影响 accuracy，"
            "还比较不同 corruption 下的鲁棒性 AUC、CBI、KL collapse 与 latent geometry。",
            "从系统角度看，本项目形成了 GPU 离线实验、静态 artifacts、后端 API、中文交互 dashboard 和报告之间的闭环。",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Chinese VIB PhaseLab report from measured artifacts.")
    parser.add_argument("--phase-root", type=Path, default=Path("artifacts/phase"))
    parser.add_argument("--result-root", type=Path, default=Path("report/phase_results"))
    parser.add_argument("--output", type=Path, default=Path("report/phaselab_report.md"))
    args = parser.parse_args()
    generate_phaselab_report(args.phase_root, args.result_root, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
