# VIB PhaseLab：表示压缩的鲁棒性相变与诊断系统

《信息论》课程大作业：从变分信息瓶颈（Variational Information Bottleneck, VIB）复现实验扩展为 **VIB PhaseLab**，分析神经网络表示压缩在 clean accuracy、多类型输入扰动鲁棒性、KL proxy 和 latent 几何结构之间的相变行为。

## 核心问题

瓶颈系数 β 何时带来有效压缩（useful compression），何时变成破坏任务信息的过度压缩（over-compression）？

本项目使用：

- `average_kl = KL(q(z|x)||p(z))` 作为 `I(X;Z)` 的变分上界 proxy。
- clean accuracy 和 corruption accuracy 作为任务性能与鲁棒性的经验指标。
- robustness AUC、compression benefit index、KL collapse score 和 phase label 描述 β 相变。
- latent geometry / latent drift 解释不同 β 下表示空间的结构变化。

## 最终产物

- 离线 PyTorch 实验包：`src/vib_project/`
- 真实实验 artifacts：
  - `artifacts/full/`：原始 MNIST/Fashion-MNIST Gaussian-noise VIB 复现实验。
  - `artifacts/phase/`：PhaseLab 全量实验，共 39 个实验配置、780 条 corruption robustness 评测。
- FastAPI 只读 artifact API：`backend/`
- React + TypeScript + Vite 中文 dashboard：`frontend/`
- 中文报告与汇总结果：
  - `report/phaselab_report.md`：最终 PhaseLab 课程报告 Markdown 版。
  - `report/phaselab_report.tex`：最终 PhaseLab 课程报告 LaTeX 版。
  - `report/phaselab_report.pdf`：已编译 PDF 版报告。
  - `report/results/`：原始实验汇总表。
  - `report/phase_results/`：PhaseLab metrics / robustness / phase labels / key findings。

## 项目结构

```text
src/vib_project/
  corruption.py           # Gaussian / salt-pepper / blur / contrast corruption
  phase_metrics.py        # robustness AUC, CBI, KL collapse, phase labels
  latent_diagnostics.py   # latent geometry and drift diagnostics
  phase_runner.py         # PhaseLab experiment orchestration
  cli.py                  # run / phase-run CLI entrypoints

artifacts/
  full/                   # 原始 VIB 复现实验 artifacts
  phase/                  # PhaseLab 全量真实实验 artifacts

backend/
  app.py                  # FastAPI artifact service

frontend/
  src/                    # 中文 dashboard and PhaseLab views

scripts/
  run_core_experiments.sh
  run_phase_experiments.sh
  validate_phase_artifacts.py
  summarize_phase_results.py
  generate_phaselab_report.py

report/
  phaselab_report.md
  phaselab_report.tex
  phaselab_report.pdf
  phase_results/
```

## 快速验证

从仓库根目录运行：

```bash
python -m pytest -v
python scripts/validate_full_artifacts.py artifacts/full
python scripts/validate_phase_artifacts.py artifacts/phase
python scripts/summarize_results.py --artifact-root artifacts/full --output-dir report/results
python scripts/summarize_phase_results.py --artifact-root artifacts/phase --output-dir report/phase_results
python scripts/generate_phaselab_report.py --phase-root artifacts/phase --result-root report/phase_results --output report/phaselab_report.md
npm --prefix frontend test -- src/lib/metrics.test.ts src/lib/phaseMetrics.test.ts
npm --prefix frontend run build
```

当前最终验证结果：

- Python / backend tests：`52 passed, 1 warning`
- Phase artifacts validator：3 个数据集、39 个实验配置、missing `[]`
- Phase summary：39 metric rows、780 robustness rows、39 phase rows
- Frontend tests：6 passed
- Frontend build：通过；Vite/Recharts chunk-size warning 可接受
- 浏览器验证：PhaseLab overview、phase map、Compression Lens、Diagnosis Lab 均能读取真实 `artifacts/phase`

## 查看交互式实验台

启动后端：

```bash
VIB_ARTIFACT_DIR=artifacts/full VIB_PHASE_ARTIFACT_DIR=artifacts/phase \
  python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

启动前端：

```bash
npm --prefix frontend run dev -- --host 127.0.0.1
```

如果 8000 端口被占用，可改用其他后端端口，并通过 Vite proxy 环境变量指定：

```bash
VIB_ARTIFACT_DIR=artifacts/full VIB_PHASE_ARTIFACT_DIR=artifacts/phase \
  python -m uvicorn backend.app:app --host 127.0.0.1 --port 8010

VITE_API_PROXY_TARGET=http://127.0.0.1:8010 \
  npm --prefix frontend run dev -- --host 127.0.0.1
```

页面包含：

- PhaseLab 鲁棒性相变总览
- β phase map
- Compression Lens
- Diagnosis Lab
- 原始 VIB 实验指标卡、信息平面、鲁棒性曲线和 latent PCA

## 复现实验

小规模 smoke test：

```bash
bash scripts/run_tiny_smoke.sh
bash scripts/run_phase_smoke.sh
```

单个 PhaseLab 实验示例：

```bash
python -m vib_project.cli phase-run \
  --dataset mnist \
  --model vib \
  --beta 0.001 \
  --epochs 1 \
  --limit-train 128 \
  --limit-test 64 \
  --output-dir artifacts/dev-phase \
  --corruptions gaussian salt_pepper blur contrast \
  --severities 0.0 0.1 0.2 0.3 0.4
```

原始完整实验：

```bash
bash scripts/run_core_experiments.sh
```

PhaseLab 全量实验：

```bash
bash scripts/run_phase_experiments.sh
```

如遇 cuDNN 初始化问题：

```bash
VIB_DISABLE_CUDNN=1 bash scripts/run_phase_experiments.sh
```

## PhaseLab 实验矩阵

`artifacts/phase` 覆盖：

- 数据集：MNIST、Fashion-MNIST、CIFAR-10
- 模型：CNN baseline、VIB-CNN
- β grid：`0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1`
- corruption：Gaussian、salt-and-pepper、blur、contrast
- severity：`0.0, 0.1, 0.2, 0.3, 0.4`

主要发现见 `report/phaselab_report.md` 和 `report/phase_results/key_findings.md`。

## 信息论解释边界

- VIB 目标对应 `I(Z;Y) - beta * I(Z;X)` 的压缩—预测折中。
- 本项目中的 `average_kl` 是 KL proxy / variational upper bound，不是精确互信息估计。
- clean accuracy、noisy/corruption accuracy 和 robustness AUC 是预测性能与鲁棒性的经验指标，不直接等同于 `I(Z;Y)`。
- phase label 是实验诊断标签，不是理论定理。

## 实验环境记录

PhaseLab 全量实验在实验室 GPU/NAS 环境运行：

```bash
ssh Tang-2-Wu
cd /NAS/yesh/vib-robustness-lab
source /data/wujcan/yesh/miniconda3/etc/profile.d/conda.sh
conda activate tn_env
export VIB_DISABLE_CUDNN=1 PYTHONPATH=src
CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase_experiments.sh
```

远端日志已同步到：`logs/phase_full_20260608_001753.log`。
