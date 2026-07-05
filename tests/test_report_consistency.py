from pathlib import Path


def test_phaselab_report_clean_best_text_matches_summary():
    tex = Path("report/phaselab_report.tex").read_text(encoding="utf-8")

    assert "MNIST & mnist\\_vib\\_beta\\_0\\_03" in tex
    assert "Fashion-MNIST & fashion\\_mnist\\_vib\\_beta\\_0\\_03" in tex
    assert "在 MNIST 上，普通 CNN baseline 的 clean accuracy 为 98.81\\%，train-test gap 为 0.76\\%。VIB-CNN 在 $\\beta=0.03$ 时取得最高 clean accuracy 99.39\\%" in tex
    assert "在 Fashion-MNIST 上，CNN baseline 的 clean accuracy 为 90.50\\%，train-test gap 为 2.82\\%。VIB-CNN 在 $\\beta=0.03$ 时取得最高 clean accuracy 92.29\\%" in tex
    assert "VIB-CNN 在 $\\beta=0.1$ 时取得最高 clean accuracy 99.38\\%" not in tex
    assert "VIB-CNN 在 $\\beta=0.01$ 时取得最高 clean accuracy 92.17\\%" not in tex
