from vib_project.phase_metrics import (
    assign_phase_label,
    compute_compression_benefit_index,
    compute_kl_collapse_score,
    compute_robustness_auc,
)


def test_compute_robustness_auc_groups_by_corruption():
    rows = [
        {"corruption": "gaussian", "severity": 0.0, "accuracy": 0.9},
        {"corruption": "gaussian", "severity": 0.2, "accuracy": 0.7},
        {"corruption": "contrast", "severity": 0.0, "accuracy": 0.9},
        {"corruption": "contrast", "severity": 0.2, "accuracy": 0.8},
    ]
    assert compute_robustness_auc(rows) == {"gaussian": 0.8, "contrast": 0.85}


def test_compute_compression_benefit_index():
    vib = {"gaussian": 0.9}
    baseline = {"gaussian": 0.82}
    assert compute_compression_benefit_index(vib, baseline) == {"gaussian": 0.08}


def test_kl_collapse_score_handles_zero_reference():
    assert compute_kl_collapse_score(1.0, 0.0) is None
    assert compute_kl_collapse_score(2.0, 10.0) == 0.2


def test_assign_phase_label_over_compressed():
    label = assign_phase_label(
        average_kl=0.1,
        beta0_kl=10.0,
        test_accuracy=0.70,
        best_accuracy=0.90,
        cbi_by_corruption={"gaussian": -0.1},
    )
    assert label == "over-compressed"


def test_assign_phase_label_useful_compression():
    label = assign_phase_label(
        average_kl=4.0,
        beta0_kl=100.0,
        test_accuracy=0.89,
        best_accuracy=0.90,
        cbi_by_corruption={"gaussian": 0.02},
    )
    assert label == "useful-compression"


def test_assign_phase_label_unstable_when_cbi_has_mixed_signs():
    label = assign_phase_label(
        average_kl=4.0,
        beta0_kl=10.0,
        test_accuracy=0.84,
        best_accuracy=0.90,
        cbi_by_corruption={"gaussian": 0.03, "contrast": -0.02},
    )
    assert label == "unstable"
