# Compact Dashboard Density Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the VIB PhaseLab dashboard substantially denser for 13–14 inch 1080p screens and report screenshots by replacing the large hero with a compact top bar and tightening typography, spacing, and cards.

**Architecture:** Keep the existing artifact-driven React component tree unchanged and make the first pass through CSS only. The implementation changes `frontend/src/style.css` density rules, then verifies the live dashboard in a browser and refreshes the report screenshots that document the web results.

**Tech Stack:** React, TypeScript, Vite, Recharts, CSS, FastAPI artifact API, Playwright/browser verification.

---

## File Structure

- Modify: `frontend/src/style.css`
  - Owns all visual density changes: top bar hero, global spacing, panels, PhaseLab cards, controls, and responsive behavior.
- Read-only reference: `frontend/src/App.tsx`
  - Confirms the existing order: compact hero, PhaseLab overview, phase map, compression lens, diagnosis lab, original VIB workspace.
- Read-only reference: `frontend/src/components/PhaseOverview.tsx`
  - Uses `.phase-overview`, `.panel`, `.panel-heading`, `.phase-overview-grid`, `.phase-overview-item`.
- Read-only reference: `frontend/src/components/PhaseMap.tsx`
  - Uses `.phase-map`, `.panel`, `.panel-heading`, `.phase-map-grid`, `.phase-cell`.
- Read-only reference: `frontend/src/components/CompressionLens.tsx`
  - Uses `.compression-lens`, `.panel`, `.phase-overview-grid`, `.phase-overview-item`, `.phase-list`.
- Read-only reference: `frontend/src/components/DiagnosisLab.tsx`
  - Uses `.diagnosis-lab`, `.panel`, `.phase-overview-grid`, `.phase-overview-item`, `.phase-list`.
- Modify after visual verification: `report/figures/dashboard_phaselab_overview.png`
  - Browser-captured overview screenshot after compact styling.
- Modify after visual verification: `report/figures/dashboard_phaselab_diagnosis.png`
  - Browser-captured diagnosis screenshot after compact styling.
- Optional modify only if screenshots are regenerated into PDF in the same pass: `report/phaselab_report.pdf`
  - Recompiled report PDF with refreshed screenshots.

## Task 1: Compact the Global Dashboard CSS

**Files:**
- Modify: `frontend/src/style.css`

- [ ] **Step 1: Inspect the current stylesheet sections before editing**

Read `frontend/src/style.css` and locate these selectors:

```css
main
.hero
.hero-copy
.eyebrow,
.section-kicker
.hero h1
.hero p:not(.eyebrow)
.hero-mark
.workspace
.control-strip
.metric-grid
.metric-card
.analysis-grid
.panel
.panel h2
.panel-heading
.muted
.phaselab-section
.phase-overview-grid,
.phase-map-grid
.phase-overview-item,
.phase-cell
.phase-overview-item strong,
.phase-cell strong
.phase-list
.dataset-tabs button
.interpretation-panel
@media (max-width: 900px)
```

Expected: all selectors already exist in `frontend/src/style.css`.

- [ ] **Step 2: Replace the top-level page and hero density rules**

In `frontend/src/style.css`, replace the existing `main`, `.hero`, `.hero-copy`, `.eyebrow/.section-kicker`, `.hero h1`, `.hero p:not(.eyebrow)`, `.hero-mark`, `.hero-mark::before/.after`, `.hero-mark b`, and `.hero-mark span` rules with this compact dashboard version:

```css
main {
  width: min(1280px, calc(100vw - 28px));
  margin: 0 auto;
  padding: 14px 0 32px;
}

.hero {
  position: sticky;
  top: 0;
  z-index: 10;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 18px;
  align-items: center;
  min-height: 72px;
  padding: 12px 0;
  border-bottom: 1px solid rgba(22, 33, 30, 0.14);
  background: rgba(247, 242, 231, 0.92);
  backdrop-filter: blur(14px);
  animation: rise-in 420ms ease-out both;
}

.hero-copy {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 14px;
  align-items: baseline;
  max-width: none;
}

.eyebrow,
.section-kicker {
  margin: 0 0 8px;
  color: #0f8f7d;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.13em;
  text-transform: uppercase;
}

.hero .eyebrow {
  margin: 0;
  white-space: nowrap;
}

.hero h1 {
  max-width: none;
  margin: 0;
  color: #13231f;
  font-size: clamp(22px, 2.8vw, 34px);
  line-height: 1.02;
  letter-spacing: -0.055em;
}

.hero p:not(.eyebrow) {
  max-width: 360px;
  margin: 0;
  color: #4e5b55;
  font-size: 13px;
  line-height: 1.45;
}

.hero-mark {
  position: relative;
  display: grid;
  place-items: center;
  width: 70px;
  height: 70px;
  justify-self: end;
  border: 1px solid rgba(15, 143, 125, 0.26);
  border-radius: 50%;
  background:
    linear-gradient(135deg, rgba(255, 255, 255, 0.52), rgba(15, 143, 125, 0.08)),
    repeating-linear-gradient(90deg, transparent 0 10px, rgba(22, 33, 30, 0.035) 10px 11px);
  color: #0f8f7d;
  box-shadow: 0 14px 34px rgba(47, 62, 56, 0.12);
}

.hero-mark::before,
.hero-mark::after {
  content: "";
  position: absolute;
  border-radius: 999px;
  border: 1px solid rgba(194, 106, 46, 0.3);
}

.hero-mark::before {
  inset: 10px;
}

.hero-mark::after {
  inset: 22px;
}

.hero-mark b {
  color: #c26a2e;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 30px;
  font-weight: 500;
  letter-spacing: -0.08em;
}

.hero-mark span {
  position: absolute;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 9px;
}

.hero-mark span:first-child {
  top: 9px;
  left: 8px;
}

.hero-mark span:last-child {
  right: 7px;
  bottom: 10px;
}
```

- [ ] **Step 3: Replace workspace, controls, metric, panel, and text density rules**

In the same file, update these rules to the following values. Keep unrelated declarations not shown here only if they are needed for existing behavior.

```css
.workspace {
  margin-top: 18px;
  animation: rise-in 520ms ease-out 60ms both;
}

.control-strip {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 18px;
  padding: 18px 0;
  border-bottom: 1px solid rgba(22, 33, 30, 0.14);
}

.control-strip h2 {
  margin: 0;
  font-size: clamp(22px, 3vw, 34px);
  letter-spacing: -0.045em;
}

.selector {
  display: grid;
  min-width: min(380px, 100%);
  gap: 7px;
  color: #4e5b55;
  font-size: 13px;
  font-weight: 700;
}

.selector select {
  width: 100%;
  padding: 9px 13px;
  border: 1px solid rgba(22, 33, 30, 0.18);
  border-radius: 999px;
  background: rgba(255, 252, 244, 0.82);
  color: #16211e;
  font: inherit;
  outline: none;
  box-shadow: 0 10px 24px rgba(47, 62, 56, 0.08);
  transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1px;
  margin-top: 14px;
  overflow: hidden;
  border: 1px solid rgba(22, 33, 30, 0.14);
  border-radius: 20px;
  background: rgba(22, 33, 30, 0.12);
}

.metric-card {
  min-height: 108px;
  padding: 16px;
  background: rgba(255, 252, 244, 0.74);
}

.metric-card span {
  display: block;
  color: inherit;
  opacity: 0.72;
  font-size: 12px;
  font-weight: 700;
}

.metric-card strong {
  display: block;
  margin-top: 20px;
  font-family: Georgia, "Times New Roman", serif;
  font-size: clamp(24px, 3vw, 36px);
  font-weight: 500;
  letter-spacing: -0.055em;
}

.analysis-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  margin-top: 16px;
}

.panel {
  min-width: 0;
  padding: 18px;
  border: 1px solid rgba(22, 33, 30, 0.14);
  border-radius: 20px;
  background: rgba(255, 252, 244, 0.7);
  box-shadow: 0 18px 44px rgba(47, 62, 56, 0.07);
  animation: rise-in 520ms ease-out 100ms both;
}

.panel h2 {
  margin: 0;
  font-size: 22px;
  letter-spacing: -0.035em;
}

.chart-panel .muted {
  min-height: 34px;
}

.latent-panel {
  margin-top: 16px;
}

.panel-heading {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 10px;
}

.panel-heading .muted {
  max-width: 400px;
  margin: 0;
}

.muted {
  color: #64746f;
  font-size: 13px;
  line-height: 1.5;
}
```

- [ ] **Step 4: Replace PhaseLab card and list density rules**

Use these values for the PhaseLab-specific blocks:

```css
.phaselab-section {
  display: grid;
  gap: 14px;
  margin-top: 16px;
}

.phase-overview-grid,
.phase-map-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 8px;
}

.phase-overview-item,
.phase-cell {
  padding: 12px;
  border: 1px solid rgba(22, 33, 30, 0.12);
  border-radius: 14px;
  background: rgba(255, 252, 244, 0.72);
  text-align: left;
}

.phase-overview-item span,
.phase-cell span {
  display: block;
  color: #64746f;
  font-size: 11px;
  font-weight: 700;
}

.phase-overview-item strong,
.phase-cell strong {
  display: block;
  margin-top: 6px;
  color: #16211e;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 21px;
  font-weight: 500;
}

.phase-overview-item p,
.phase-cell em {
  color: #4e5b55;
  font-size: 12px;
  line-height: 1.45;
}

.phase-list {
  margin-top: 10px;
  color: #4e5b55;
  font-size: 12px;
  line-height: 1.45;
}

.phase-list p {
  margin: 4px 0;
}

.experiment-controls {
  display: grid;
  gap: 8px;
}

.dataset-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}

.dataset-tabs button {
  padding: 7px 12px;
  border: 1px solid rgba(22, 33, 30, 0.18);
  border-radius: 999px;
  background: rgba(255, 252, 244, 0.72);
  color: #4e5b55;
  cursor: pointer;
  font-size: 12px;
  font-weight: 700;
  transition: background 160ms ease, border-color 160ms ease, color 160ms ease, transform 160ms ease;
}
```

- [ ] **Step 5: Replace interpretation panel density rules**

Use these values for the original VIB interpretation section:

```css
.interpretation-panel {
  margin-top: 16px;
  padding: 18px 0 4px;
  border-top: 1px solid rgba(22, 33, 30, 0.14);
}

.interpretation-panel h2 {
  margin: 0;
  font-size: 24px;
  letter-spacing: -0.035em;
}

.interpretation-panel p:not(.section-kicker) {
  max-width: 760px;
  color: #4e5b55;
  font-size: 13px;
  line-height: 1.55;
}

.interpretation-panel dl {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1px;
  overflow: hidden;
  border: 1px solid rgba(22, 33, 30, 0.12);
  border-radius: 14px;
  background: rgba(22, 33, 30, 0.1);
}

.interpretation-panel dl div {
  padding: 12px;
  background: rgba(255, 252, 244, 0.72);
}

.interpretation-panel dt {
  color: #64746f;
  font-size: 12px;
}

.interpretation-panel dd {
  margin: 6px 0 0;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 22px;
}
```

- [ ] **Step 6: Replace the mobile/tablet media query**

In the existing `@media (max-width: 900px)` block, use this compact responsive behavior:

```css
@media (max-width: 900px) {
  main {
    width: min(100vw - 20px, 1280px);
    padding-top: 10px;
  }

  .hero {
    grid-template-columns: 1fr auto;
    min-height: 62px;
    gap: 10px;
    padding: 9px 0;
  }

  .hero-copy {
    grid-template-columns: 1fr;
    gap: 3px;
    align-items: start;
  }

  .hero h1 {
    font-size: clamp(20px, 5vw, 28px);
  }

  .hero p:not(.eyebrow) {
    display: none;
  }

  .hero-mark {
    width: 52px;
    height: 52px;
    justify-self: end;
  }

  .hero-mark b {
    font-size: 23px;
  }

  .hero-mark span {
    display: none;
  }

  .control-strip,
  .panel-heading {
    align-items: stretch;
    flex-direction: column;
  }

  .interpretation-panel dl,
  .metric-grid,
  .analysis-grid {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 7: Run frontend tests**

Run:

```bash
npm --prefix frontend test -- src/lib/metrics.test.ts src/lib/phaseMetrics.test.ts
```

Expected: the existing Vitest suite passes. CSS-only changes should not affect metric logic.

- [ ] **Step 8: Run frontend build**

Run:

```bash
npm --prefix frontend run build
```

Expected: build exits 0. The existing Recharts/Vite chunk-size warning is acceptable.

- [ ] **Step 9: Review the CSS diff before moving to browser verification**

Run:

```bash
git diff -- frontend/src/style.css
```

Expected: the diff changes layout density only. There should be no API, TypeScript, or data-flow changes.

## Task 2: Browser-Verify the Compact Dashboard

**Files:**
- Modify after verification: `report/figures/dashboard_phaselab_overview.png`
- Modify after verification: `report/figures/dashboard_phaselab_diagnosis.png`
- Read-only reference: `frontend/src/App.tsx`

- [ ] **Step 1: Start the backend artifact API**

Run from the repository root with Bash:

```bash
VIB_ARTIFACT_DIR=artifacts/full VIB_PHASE_ARTIFACT_DIR=artifacts/phase \
  python -m uvicorn backend.app:app --host 127.0.0.1 --port 8010
```

Expected: Uvicorn starts on `http://127.0.0.1:8010` and serves both original and PhaseLab artifacts.

- [ ] **Step 2: Start the frontend dev server**

Run from the repository root with Bash:

```bash
VITE_API_PROXY_TARGET=http://127.0.0.1:8010 \
  npm --prefix frontend run dev -- --host 127.0.0.1
```

Expected: Vite reports a local URL. Use the actual port Vite prints; if `5173` is occupied it may choose a later port.

- [ ] **Step 3: Inspect the dashboard at a 13–14 inch 1080p-like viewport**

Use a browser viewport around `1366x768` or `1440x900`.

Expected visual checks:

```text
- The top hero behaves like a compact top bar, not a landing-page block.
- PhaseLab overview starts near the top of the first screen.
- At least PhaseLab overview and a meaningful part of Phase Map are visible without scrolling at 1366x768.
- At 1440x900, PhaseLab overview, Phase Map, and the start of Compression Lens are visible without excessive scrolling.
- Text remains legible; no key Chinese labels are clipped.
- The beta mark is small and does not steal layout space.
```

- [ ] **Step 4: Capture the compact overview screenshot**

Use the live frontend URL and capture a viewport screenshot to:

```text
report/figures/dashboard_phaselab_overview.png
```

Recommended viewport: `1440x900` for report readability while still representing compact laptop density.

Expected: the image shows the compact top bar plus PhaseLab overview and Phase Map much higher than before.

- [ ] **Step 5: Capture the compact diagnosis screenshot**

Interact with the page so Compression Lens and Diagnosis Lab are visible, then capture:

```text
report/figures/dashboard_phaselab_diagnosis.png
```

Recommended viewport: `1440x900`.

Expected: the image shows compact Compression Lens and Diagnosis Lab cards with reduced whitespace and readable diagnostic rows.

- [ ] **Step 6: Stop local backend and frontend servers**

Stop both background tasks that were started for verification.

Expected: no local dev servers remain running from this verification pass.

## Task 3: Refresh Report PDF if Screenshots Changed

**Files:**
- Modify: `report/phaselab_report.pdf`
- Read-only reference: `report/phaselab_report.tex`
- Read-only reference: `report/phaselab_report.md`

- [ ] **Step 1: Compile the LaTeX report with refreshed screenshots**

Run:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=report/build report/phaselab_report.tex
```

Expected: LaTeX exits 0 and writes `report/build/phaselab_report.pdf`.

- [ ] **Step 2: Copy the compiled PDF to the final report path**

Run:

```bash
cp report/build/phaselab_report.pdf report/phaselab_report.pdf
```

Expected: `report/phaselab_report.pdf` is updated.

- [ ] **Step 3: Check the LaTeX log for screenshot/layout warnings**

Run:

```bash
python - <<'PY'
from pathlib import Path
log = Path('report/build/phaselab_report.log').read_text(encoding='utf-8', errors='ignore')
print('overfull_count', log.count('Overfull '))
print('missing_file_warnings', log.count('LaTeX Warning: File'))
PY
```

Expected:

```text
overfull_count 0
missing_file_warnings 0
```

- [ ] **Step 4: Run report packaging tests**

Run:

```bash
python -m pytest tests/test_package_submission.py tests/test_generate_phaselab_report.py -v
```

Expected: both tests pass.

## Task 4: Final Verification and Handoff

**Files:**
- Review: `frontend/src/style.css`
- Review: `report/figures/dashboard_phaselab_overview.png`
- Review: `report/figures/dashboard_phaselab_diagnosis.png`
- Review if updated: `report/phaselab_report.pdf`
- Review existing unstaged handoff edit: `CLAUDE.md`

- [ ] **Step 1: Check all changed files**

Run:

```bash
git status --short
```

Expected changed paths include only relevant files:

```text
M CLAUDE.md
M frontend/src/style.css
M report/figures/dashboard_phaselab_overview.png
M report/figures/dashboard_phaselab_diagnosis.png
M report/phaselab_report.pdf
```

If `report/phaselab_report.pdf` was not refreshed, it may be absent from this list. If other files appear, inspect them before proceeding.

- [ ] **Step 2: Review the final diff/stat**

Run:

```bash
git diff --stat
```

Expected: CSS changes dominate; screenshot/PDF binary changes are present only if screenshots/PDF were refreshed.

- [ ] **Step 3: Run final frontend verification commands fresh**

Run:

```bash
npm --prefix frontend test -- src/lib/metrics.test.ts src/lib/phaseMetrics.test.ts
npm --prefix frontend run build
```

Expected: tests pass and build exits 0.

- [ ] **Step 4: Report evidence without committing unless explicitly requested**

State the actual verification evidence:

```text
- Browser viewport used for manual verification.
- Which PhaseLab regions fit in first screen after compacting.
- Frontend test result.
- Frontend build result.
- Whether screenshots and PDF were refreshed.
- Current git status.
```

Do not commit unless the user explicitly asks for a commit.

---

## Self-Review

- Spec coverage: The plan covers the approved top-bar hero, global CSS density reduction, 1080p/report screenshot verification, screenshot refresh, and optional PDF refresh.
- Placeholder scan: No task uses TBD, TODO, or unspecified implementation language. Every code-changing CSS step includes concrete replacement CSS.
- Type consistency: No TypeScript types or component props are changed; CSS selectors match the current component files read before writing this plan.
- Scope check: This is a single CSS-first UI density pass. It does not include PhaseLab structural reordering, new screenshot mode, or API/data changes.
