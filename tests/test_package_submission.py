from pathlib import Path

from scripts.package_submission import build_submission_file_list


def test_build_submission_file_list_excludes_caches_and_node_modules(tmp_path: Path):
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    (tmp_path / "frontend" / "node_modules").mkdir(parents=True)
    (tmp_path / "frontend" / "node_modules" / "x.js").write_text("x", encoding="utf-8")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "x").write_text("x", encoding="utf-8")
    (tmp_path / "report" / "phaselab_report.md").parent.mkdir(parents=True)
    (tmp_path / "report" / "phaselab_report.md").write_text("report", encoding="utf-8")
    (tmp_path / "report" / "phaselab_report.tex").write_text("tex", encoding="utf-8")
    (tmp_path / "report" / "phaselab_report.pdf").write_text("pdf", encoding="utf-8")
    (tmp_path / "report" / "build").mkdir(parents=True)
    (tmp_path / "report" / "build" / "phaselab_report.aux").write_text("aux", encoding="utf-8")

    files = build_submission_file_list(tmp_path)

    assert "README.md" in files
    assert "report/phaselab_report.md" in files
    assert "report/phaselab_report.tex" in files
    assert "report/phaselab_report.pdf" in files
    assert not any("node_modules" in item for item in files)
    assert not any(".pytest_cache" in item for item in files)
    assert not any("report/build" in item for item in files)
