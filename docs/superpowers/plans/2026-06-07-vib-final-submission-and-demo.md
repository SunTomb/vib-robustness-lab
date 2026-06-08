# VIB Final Submission and Demo Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or inline task execution. Avoid long-running subagents in this project because previous large-context subagent runs triggered stream decoding interruptions. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the completed VIB experiment system into a clean course-submission-ready deliverable with reproducible instructions, screenshots, optional lightweight deployment path, and safe git/commit preparation.

**Architecture:** Keep `artifacts/full` and `report/results` as the evidence source of truth. Add only lightweight submission metadata, screenshots, README/report polish, and optional deployment notes; do not rerun full training unless a defect is discovered. Preserve the artifact-driven offline-training -> FastAPI -> React dashboard flow.

**Tech Stack:** Python 3.10+, pytest, FastAPI, React/Vite, Playwright/browser verification, Markdown, Bash/SSH for optional server work.

---

## File Structure

```text
vib-robustness-lab/
├── README.md                                      # modify: final quickstart, artifact/demo/report guide
├── CLAUDE.md                                      # modify only if final operating notes changed
├── docs/superpowers/plans/
│   └── 2026-06-07-vib-final-submission-and-demo.md
├── report/
│   ├── final_report.md                            # modify: add figure/screenshot references and final verification note
│   ├── figures/
│   │   ├── dashboard_mnist_overview.png           # generated screenshot
│   │   ├── dashboard_fashion_beta1.png            # generated screenshot
│   │   └── existing SVG result figures
│   └── submission_manifest.md                     # new: exact deliverables and provenance
├── scripts/
│   ├── validate_full_artifacts.py                 # existing
│   ├── summarize_results.py                       # existing
│   └── package_submission.py                      # optional new: create deterministic manifest/checklist, not zip by default
└── frontend/src/                                  # modify only if screenshot verification finds UI defects
```

Boundaries:

- Do not hand-edit `artifacts/full`.
- Do not invent new metrics; all report numbers must remain traceable to `artifacts/full` or `report/results`.
- Do not commit generated caches, virtual environments, `node_modules`, `frontend/dist`, `.claude`, or raw datasets.
- Do not deploy to the small cloud server unless explicitly requested after this plan; deployment remains optional.
- Do not use PowerShell.

---

## Task 1: Git and Artifact Policy Audit

**Files:**
- Read/inspect: `.gitignore`
- Read/inspect: `git status --short`
- Modify only if needed: `.gitignore`
- Create: `report/submission_manifest.md`

- [ ] **Step 1: Inspect current git state**

Run:

```bash
git status --short
```

Expected: source directories, report outputs, full artifacts, logs, tests, and plans are visible or intentionally ignored.

- [ ] **Step 2: Inspect ignore policy**

Run:

```bash
python - <<'PY'
from pathlib import Path
print(Path('.gitignore').read_text(encoding='utf-8'))
PY
```

Check whether `artifacts/full/` is ignored. If it is ignored, decide in the manifest whether the final submission will:

1. keep `artifacts/full` uncommitted and submit it separately as an archive, or
2. track a curated artifact subset in git, or
3. remove the ignore entry and commit full artifacts if file size is acceptable.

Preferred policy: keep full generated artifacts out of normal git history unless the course submission system requires a single repository archive containing results.

- [ ] **Step 3: Create submission manifest**

Create `report/submission_manifest.md`:

```markdown
# VIB Project Submission Manifest

## Evidence source

- Full experiment artifacts: `artifacts/full/`
- Metrics table: `report/results/metrics_summary.csv`
- Robustness table: `report/results/robustness_summary.csv`
- Figures: `report/figures/*.svg`
- Final report: `report/final_report.md`

## Experiment provenance

- Datasets: MNIST, Fashion-MNIST
- Models: CNN baseline, VIB-CNN
- β grid: 0, 1e-4, 1e-3, 1e-2, 1e-1, 1
- Noise grid: σ = 0.0, 0.1, 0.2, 0.3, 0.4
- Full artifact count: 14 experiments
- Robustness rows: 70 evaluations

## Submission policy

- Source code should be committed.
- Report files and derived CSV/SVG figures should be committed.
- `artifacts/full` should be included in the course submission archive or committed only if the user explicitly chooses that policy.
- Local caches, datasets, virtual environments, `node_modules`, and frontend build output should not be submitted.

## Verification commands

```bash
python -m pytest -v
python scripts/validate_full_artifacts.py artifacts/full
python scripts/summarize_results.py --artifact-root artifacts/full --output-dir report/results
npm --prefix frontend test -- src/lib/metrics.test.ts
npm --prefix frontend run build
```
```

- [ ] **Step 4: Verify manifest has no stale placeholders**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('report/submission_manifest.md').read_text(encoding='utf-8')
bad = [word for word in ['TODO', 'TBD', '填写'] if word in text]
print({'placeholders': bad})
raise SystemExit(1 if bad else 0)
PY
```

Expected: `{'placeholders': []}`.

---

## Task 2: README Final Quickstart Polish

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read current README**

Run:

```bash
python - <<'PY'
from pathlib import Path
print(Path('README.md').read_text(encoding='utf-8'))
PY
```

- [ ] **Step 2: Update README with final project workflow**

Ensure `README.md` includes these sections:

```markdown
## 快速验证

```bash
python -m pytest -v
python scripts/validate_full_artifacts.py artifacts/full
python scripts/summarize_results.py --artifact-root artifacts/full --output-dir report/results
npm --prefix frontend run build
```

## 查看交互式实验台

```bash
VIB_ARTIFACT_DIR=artifacts/full python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5177
```

然后打开 `http://127.0.0.1:5177`。

## 结果与报告

- 最终报告：`report/final_report.md`
- 指标表：`report/results/metrics_summary.csv`
- 鲁棒性表：`report/results/robustness_summary.csv`
- 报告图表：`report/figures/`

## 复现实验

小规模 smoke test：

```bash
bash scripts/run_tiny_smoke.sh
```

完整实验建议在实验室 GPU/NAS 环境运行，训练结果写入 `artifacts/full`。如果遇到 cuDNN 初始化问题，可设置：

```bash
VIB_DISABLE_CUDNN=1 bash scripts/run_core_experiments.sh
```
```

- [ ] **Step 3: Verify README commands are present**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('README.md').read_text(encoding='utf-8')
required = [
    'python scripts/validate_full_artifacts.py artifacts/full',
    'VIB_ARTIFACT_DIR=artifacts/full python -m uvicorn backend.app:app',
    'report/final_report.md',
    'bash scripts/run_tiny_smoke.sh',
]
missing = [item for item in required if item not in text]
print({'missing': missing})
raise SystemExit(1 if missing else 0)
PY
```

Expected: `{'missing': []}`.

---

## Task 3: Dashboard Screenshots for Report and Submission

**Files:**
- Create generated: `report/figures/dashboard_mnist_overview.png`
- Create generated: `report/figures/dashboard_fashion_beta1.png`
- Modify: `report/final_report.md`

- [ ] **Step 1: Start local backend and frontend**

Run backend:

```bash
VIB_ARTIFACT_DIR=artifacts/full python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Run frontend:

```bash
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5177
```

- [ ] **Step 2: Capture MNIST overview screenshot**

Use browser automation to open `http://127.0.0.1:5177`, keep MNIST selected, and save a full-page screenshot to:

```text
report/figures/dashboard_mnist_overview.png
```

Expected screenshot content:

- Chinese title: `变分信息瓶颈鲁棒泛化实验台`
- Dataset tabs: `MNIST`, `Fashion-MNIST`
- Metric cards
- Information plane
- Robustness chart
- Latent PCA panel

- [ ] **Step 3: Capture Fashion-MNIST β=1 screenshot**

In browser automation:

1. click `Fashion-MNIST`,
2. select `fashion_mnist_vib_beta_1`,
3. verify page text includes `60.14%` and `0.0210`,
4. save screenshot to:

```text
report/figures/dashboard_fashion_beta1.png
```

- [ ] **Step 4: Add screenshots to final report**

Modify `report/final_report.md` section `## 6 Web 交互系统` to include:

```markdown
![MNIST 实验台总览](figures/dashboard_mnist_overview.png)

![Fashion-MNIST β=1 过压缩示例](figures/dashboard_fashion_beta1.png)
```

- [ ] **Step 5: Verify screenshot files exist**

Run:

```bash
python - <<'PY'
from pathlib import Path
expected = [
    'report/figures/dashboard_mnist_overview.png',
    'report/figures/dashboard_fashion_beta1.png',
]
missing = [path for path in expected if not Path(path).exists()]
print({'missing': missing})
raise SystemExit(1 if missing else 0)
PY
```

Expected: `{'missing': []}`.

---

## Task 4: Optional Submission Packaging Script

**Files:**
- Create: `scripts/package_submission.py`
- Create: `tests/test_package_submission.py`

This task creates a manifest/checklist helper only. It should not zip or copy large artifacts by default, because artifact inclusion policy should remain explicit.

- [ ] **Step 1: Write failing test**

Create `tests/test_package_submission.py`:

```python
from pathlib import Path

from scripts.package_submission import build_submission_file_list


def test_build_submission_file_list_excludes_caches_and_node_modules(tmp_path: Path):
    (tmp_path / 'README.md').write_text('readme', encoding='utf-8')
    (tmp_path / 'frontend' / 'node_modules').mkdir(parents=True)
    (tmp_path / 'frontend' / 'node_modules' / 'x.js').write_text('x', encoding='utf-8')
    (tmp_path / '.pytest_cache').mkdir()
    (tmp_path / '.pytest_cache' / 'x').write_text('x', encoding='utf-8')
    (tmp_path / 'report' / 'final_report.md').parent.mkdir(parents=True)
    (tmp_path / 'report' / 'final_report.md').write_text('report', encoding='utf-8')

    files = build_submission_file_list(tmp_path)

    assert 'README.md' in files
    assert 'report/final_report.md' in files
    assert not any('node_modules' in item for item in files)
    assert not any('.pytest_cache' in item for item in files)
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/test_package_submission.py -v
```

Expected: FAIL because `scripts.package_submission` does not exist.

- [ ] **Step 3: Implement file list helper**

Create `scripts/package_submission.py`:

```python
import argparse
from pathlib import Path

EXCLUDED_PARTS = {
    '.git', '.claude', '.pytest_cache', '__pycache__',
    'node_modules', 'dist', '.venv', 'venv', 'data', 'artifacts/dev',
}


def _is_excluded(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDED_PARTS:
        return True
    normalized = path.as_posix()
    return normalized.startswith('artifacts/dev/')


def build_submission_file_list(root: Path) -> list[str]:
    files: list[str] = []
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _is_excluded(relative):
            continue
        files.append(relative.as_posix())
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description='Print VIB project submission file list.')
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    for item in build_submission_file_list(args.root):
        print(item)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test and inspect file list**

Run:

```bash
python -m pytest tests/test_package_submission.py -v
python scripts/package_submission.py --root . > report/submission_file_list.txt
```

Expected: test passes; generated file list excludes cache and dependency directories.

---

## Task 5: Final Report Polish and Figure Reference Verification

**Files:**
- Modify: `report/final_report.md`

- [ ] **Step 1: Verify report has core result values**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('report/final_report.md').read_text(encoding='utf-8')
required = ['99.38%', '92.17%', '60.14%', '0.0210', '79.92%', '28.81%']
missing = [item for item in required if item not in text]
print({'missing': missing})
raise SystemExit(1 if missing else 0)
PY
```

Expected: `{'missing': []}`.

- [ ] **Step 2: Add explicit figure references**

Ensure the report references these generated result figures in relevant sections:

```markdown
![MNIST β 与 clean accuracy](figures/mnist_beta_accuracy.svg)
![Fashion-MNIST β 与 clean accuracy](figures/fashion_mnist_beta_accuracy.svg)
![信息平面](figures/information_plane.svg)
![MNIST 噪声鲁棒性曲线](figures/mnist_robustness.svg)
![Fashion-MNIST 噪声鲁棒性曲线](figures/fashion_mnist_robustness.svg)
```

- [ ] **Step 3: Verify all referenced local figures exist**

Run:

```bash
python - <<'PY'
import re
from pathlib import Path
report = Path('report/final_report.md')
text = report.read_text(encoding='utf-8')
refs = re.findall(r'\]\((figures/[^)]+)\)', text)
missing = [ref for ref in refs if not (report.parent / ref).exists()]
print({'referenced_figures': refs, 'missing': missing})
raise SystemExit(1 if missing else 0)
PY
```

Expected: `missing` is empty.

---

## Task 6: Optional Lightweight Deployment Decision Record

**Files:**
- Create: `docs/deployment-notes.md`

This task documents the safe deployment path without actually deploying.

- [ ] **Step 1: Create deployment notes**

Create `docs/deployment-notes.md`:

```markdown
# Lightweight Deployment Notes

## Recommended deployment shape

Deploy only the artifact viewer:

- FastAPI backend reads precomputed `artifacts/full`.
- React/Vite frontend is built once and served statically or through a small web server.
- No GPU, training loop, raw dataset download, or checkpoint-heavy workflow is required on the cloud server.

## Not recommended on the small cloud server

- Full PyTorch/CUDA installation
- Full MNIST/Fashion-MNIST experiment sweeps
- Raw dataset/cache transfer unless explicitly needed
- Docker images that include GPU training dependencies

## Suggested production commands

Build frontend locally or on a capable machine:

```bash
npm --prefix frontend run build
```

Run backend against copied artifacts:

```bash
VIB_ARTIFACT_DIR=artifacts/full python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

## Artifact policy

If deploying remotely, copy only:

- `backend/`
- `frontend/dist/`
- `artifacts/full/`
- minimal Python dependency files

Do not copy raw datasets, local virtual environments, `node_modules`, logs unless needed for audit, or experiment checkpoints.
```

- [ ] **Step 2: Verify deployment notes avoid training requirement**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('docs/deployment-notes.md').read_text(encoding='utf-8')
required = ['artifact viewer', 'No GPU', 'VIB_ARTIFACT_DIR=artifacts/full']
missing = [item for item in required if item not in text]
print({'missing': missing})
raise SystemExit(1 if missing else 0)
PY
```

Expected: `{'missing': []}`.

---

## Task 7: Final Verification and Commit Readiness

**Files:**
- Modify only if verification reveals defects.

- [ ] **Step 1: Run full verification commands**

Run:

```bash
python -m pytest -v
python scripts/validate_full_artifacts.py artifacts/full
python scripts/summarize_results.py --artifact-root artifacts/full --output-dir report/results
npm --prefix frontend test -- src/lib/metrics.test.ts
npm --prefix frontend run build
```

Expected:

- Python tests pass.
- Full artifacts report 14 experiments and no missing files.
- Summary writes 14 metric rows and 70 robustness rows.
- Frontend tests pass.
- Frontend build succeeds; Vite/Recharts chunk warning is acceptable.

- [ ] **Step 2: Browser verify dashboard one final time**

With backend and frontend running:

```bash
VIB_ARTIFACT_DIR=artifacts/full python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5177
```

Verify:

- MNIST and Fashion-MNIST tabs are visible.
- MNIST baseline explanation says it is a CNN baseline, not a compressed VIB model.
- Fashion-MNIST β=1 shows 60.14% clean accuracy and KL proxy 0.0210.
- Charts render without console-breaking errors.

- [ ] **Step 3: Inspect git status and file sizes**

Run:

```bash
git status --short
python - <<'PY'
from pathlib import Path
for path in ['artifacts/full', 'report/figures', 'frontend/dist']:
    root = Path(path)
    if not root.exists():
        continue
    size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    print(path, round(size / 1024 / 1024, 2), 'MB')
PY
```

Expected: no local caches or dependency directories are staged; artifact/report sizes are known before deciding commit policy.

- [ ] **Step 4: Prepare commit recommendation without committing**

Report a staged-file recommendation grouped by category:

- Source code: `src/`, `backend/`, `frontend/src/`, config files
- Tests: `tests/`, `backend/tests/`, frontend test files
- Scripts: `scripts/`
- Docs/report: `README.md`, `CLAUDE.md`, `docs/`, `report/final_report.md`, `report/results/`, `report/figures/`
- Generated artifacts: `artifacts/full/` only if user explicitly wants them in git
- Logs: include only if useful as experiment provenance; otherwise keep out of commit

Do not run `git add`, `git commit`, or `git push` unless the user explicitly asks.

---

## Self-Review

Spec coverage:

- Final code/result state is audited by Task 1 and Task 7.
- README and reproducibility instructions are covered by Task 2.
- Dashboard screenshots and report insertion are covered by Task 3.
- Submission packaging support is covered by Task 4.
- Report figure references and value checks are covered by Task 5.
- Deployment is documented but not performed by Task 6.

Placeholder scan:

- No task leaves a `TODO`, `TBD`, or `填写` marker in final deliverables.
- Optional actions are explicitly documented as optional and not executed without user instruction.

Safety check:

- No PowerShell commands are used.
- No destructive git commands are included.
- No commit or deployment is performed without explicit user request.
- No training rerun is required unless verification reveals a defect.
