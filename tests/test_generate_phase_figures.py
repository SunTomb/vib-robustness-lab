from scripts.generate_phase_figures import PHASE_COLORS


def test_phase_colors_include_baseline_label():
    assert PHASE_COLORS["baseline"] == "#4e5b55"
