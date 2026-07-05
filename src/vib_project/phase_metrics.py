from __future__ import annotations

from collections import defaultdict
from typing import Any

OVER_COMPRESSION_ACCURACY_DROP = 0.10
KL_COLLAPSE_THRESHOLD = 0.05
USEFUL_ACCURACY_DROP = 0.02
MEANINGFUL_KL_DROP = 0.5


def _rounded(value: float) -> float:
    return round(float(value), 12)


def compute_robustness_auc(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["corruption"])].append(float(row["accuracy"]))
    return {corruption: _rounded(sum(values) / len(values)) for corruption, values in grouped.items() if values}


def compute_normalized_robustness_auc(auc_by_corruption: dict[str, float], clean_accuracy: float) -> dict[str, float | None]:
    if clean_accuracy <= 0.0:
        return {corruption: None for corruption in auc_by_corruption}
    return {corruption: _rounded(value / clean_accuracy) for corruption, value in auc_by_corruption.items()}


def compute_compression_benefit_index(
    vib_normalized_auc: dict[str, float | None],
    baseline_normalized_auc: dict[str, float | None],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for corruption, vib_value in vib_normalized_auc.items():
        baseline_value = baseline_normalized_auc.get(corruption)
        if vib_value is None or baseline_value is None:
            result[corruption] = None
        else:
            result[corruption] = _rounded(vib_value - baseline_value)
    return result


def compute_kl_collapse_score(average_kl: float, beta0_kl: float) -> float | None:
    if beta0_kl <= 0.0:
        return None
    return _rounded(average_kl / beta0_kl)


def assign_phase_label(
    average_kl: float,
    beta0_kl: float,
    test_accuracy: float,
    best_accuracy: float,
    cbi_by_corruption: dict[str, float | None],
) -> str:
    kl_score = compute_kl_collapse_score(average_kl, beta0_kl)
    accuracy_drop = best_accuracy - test_accuracy
    valid_cbi = [value for value in cbi_by_corruption.values() if value is not None]
    best_cbi = max(valid_cbi) if valid_cbi else 0.0
    worst_cbi = min(valid_cbi) if valid_cbi else 0.0

    if kl_score is not None and accuracy_drop > OVER_COMPRESSION_ACCURACY_DROP and kl_score < KL_COLLAPSE_THRESHOLD:
        return "over-compressed"
    if kl_score is not None and kl_score > MEANINGFUL_KL_DROP:
        return "under-regularized"
    if accuracy_drop <= USEFUL_ACCURACY_DROP and best_cbi >= 0.0:
        return "useful-compression"
    if best_cbi > 0.0 and worst_cbi < 0.0:
        return "unstable"
    if best_cbi > 0.0:
        return "robustness-specialized"
    return "unstable"


def build_phase_indicators(
    robustness_rows: list[dict[str, Any]],
    clean_accuracy: float,
    baseline_normalized_auc: dict[str, float | None],
    average_kl: float,
    beta0_kl: float,
    best_accuracy: float,
) -> dict[str, Any]:
    robustness_auc = compute_robustness_auc(robustness_rows)
    normalized_auc = compute_normalized_robustness_auc(robustness_auc, clean_accuracy)
    cbi = compute_compression_benefit_index(normalized_auc, baseline_normalized_auc)
    kl_score = compute_kl_collapse_score(average_kl, beta0_kl)
    phase_label = assign_phase_label(average_kl, beta0_kl, clean_accuracy, best_accuracy, cbi)
    return {
        "robustness_auc": robustness_auc,
        "normalized_robustness_auc": normalized_auc,
        "compression_benefit_index": cbi,
        "kl_collapse_score": kl_score,
        "phase_label": phase_label,
        "over_compression_flag": phase_label == "over-compressed",
    }
