import json
from pathlib import Path

from scripts.generate_phaselab_report import generate_phaselab_report


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_generate_phaselab_report_uses_measured_summary(tmp_path: Path):
    phase_root = tmp_path / "artifacts" / "phase"
    result_root = tmp_path / "report" / "phase_results"
    report_path = tmp_path / "report" / "phaselab_report.md"
    _write_json(
        phase_root / "index.json",
        {
            "datasets": ["mnist"],
            "experiments": [
                {"id": "mnist_cnn_baseline", "dataset": "mnist", "model": "cnn", "beta": 0, "path": "mnist_cnn_baseline"},
                {"id": "mnist_vib_beta_0_1", "dataset": "mnist", "model": "vib", "beta": 0.1, "path": "mnist_vib_beta_0_1"},
            ],
        },
    )
    _write_json(
        result_root / "phase_summary.json",
        {
            "datasets": {
                "mnist": {
                    "best_clean_experiment": "mnist_vib_beta_0_1",
                    "best_clean_accuracy": 0.98,
                    "best_robust_by_corruption": {"gaussian": "mnist_cnn_baseline"},
                    "phase_label_counts": {"useful-compression": 1},
                }
            }
        },
    )
    _write_json(phase_root / "mnist_cnn_baseline" / "metrics.json", {"test_accuracy": 0.95, "average_kl": 0.0})
    _write_json(
        phase_root / "mnist_cnn_baseline" / "phase_indicators.json",
        {
            "robustness_auc": {"gaussian": 0.92},
            "compression_benefit_index": {"gaussian": 0.0},
            "kl_collapse_score": None,
            "phase_label": "under-regularized",
            "over_compression_flag": False,
        },
    )
    _write_json(phase_root / "mnist_vib_beta_0_1" / "metrics.json", {"test_accuracy": 0.98, "average_kl": 2.5})
    _write_json(
        phase_root / "mnist_vib_beta_0_1" / "phase_indicators.json",
        {
            "robustness_auc": {"gaussian": 0.9},
            "compression_benefit_index": {"gaussian": 0.2},
            "kl_collapse_score": 0.25,
            "phase_label": "useful-compression",
            "over_compression_flag": False,
        },
    )

    generate_phaselab_report(phase_root, result_root, report_path)

    report = report_path.read_text(encoding="utf-8")
    assert "mnist_vib_beta_0_1" in report
    assert "98.00%" in report
    assert "0.2500" in report
    assert "useful-compression" in report
