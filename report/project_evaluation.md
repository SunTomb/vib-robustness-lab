# VIB PhaseLab 项目评价意见

评价对象：`vib-robustness-lab`  
评价日期：2026-07-05  
评价范围：README、CLAUDE/AGENTS 指南、`src/vib_project/` 实验代码、`backend/` FastAPI API、`frontend/` React/Vite dashboard、`artifacts/full`、`artifacts/phase`、`report/phaselab_report.tex` 与 PDF。

## 一、项目整体评价

这个项目作为信息论课程大作业已经达到优秀水平。它不是停留在“复现 VIB + 展示几张结果图”，而是把 Variational Information Bottleneck 扩展成了一个完整的实验系统：离线 PyTorch 实验生成静态 artifacts，FastAPI 后端只读提供 artifact API，React/Vite 前端交互展示，中文报告再从真实结果中总结结论。

项目 framing “VIB PhaseLab：表示压缩的鲁棒性相变与诊断系统”总体成立。它的核心问题是：瓶颈系数 beta 何时带来 useful compression，何时变成 destructive over-compression。README 中明确说明 `average_kl = KL(q(z|x)||p(z))` 只是 `I(X;Z)` 的 variational upper-bound proxy，clean accuracy 和 corruption accuracy 只是任务性能与鲁棒性 proxy，phase label 是实验诊断标签而不是理论定理。这一点避免了常见的信息论过度解释问题。

项目形成了较完整闭环：

- `src/vib_project/` 负责实验、模型、训练、评测、诊断和 artifact 写出。
- `artifacts/phase/` 保存 3 个数据集、39 个实验配置、780 条 robustness 评测记录。
- `backend/app.py` 暴露 `/api/phase/index`、`/api/phase/summary`、`/api/phase/experiments/{id}` 等只读接口。
- `frontend/src/` 通过 API 读取真实 artifacts，显示 Phase overview、Phase Map、Compression Lens、Diagnosis Lab。
- `report/phaselab_report.md`、`report/phaselab_report.tex`、`report/phaselab_report.pdf` 从真实结果组织课程报告。

总体评价：这是一个有实验、有系统、有报告、有复现入口的完整课程项目，完成度和创新性都明显高于普通复现实验。

## 二、实验设计评价

实验矩阵合理。`artifacts/phase/index.json` 覆盖 MNIST、Fashion-MNIST、CIFAR-10 三个数据集；每个数据集包含 1 个 CNN baseline 和 12 个 VIB-CNN beta 配置；corruption 包括 Gaussian、salt-and-pepper、blur、contrast；severity 为 `0.0, 0.1, 0.2, 0.3, 0.4`。这与 README 和报告中声称的 39 个实验配置、780 条 robustness rows 一致。

指标设计也比较有说服力：

- `robustness_auc`：同一 corruption 下不同 severity accuracy 的平均值，直观且适合课程项目。
- `normalized_robustness_auc`：用 robustness AUC 除以 clean accuracy，避免 clean accuracy 本身过高导致 AUC 虚高。
- `compression_benefit_index`：VIB normalized robustness AUC 减去同数据集 CNN baseline normalized AUC，能体现 VIB 是否相对 baseline 更能保留扰动下性能。
- `kl_collapse_score`：用同数据集 VIB beta=0 的 KL 作为参考，适合识别过度压缩。
- `phase_label`：把 raw metrics 转换为 under-regularized、useful-compression、robustness-specialized、over-compressed、unstable 等诊断标签，便于报告和 dashboard 解释。

实际 artifacts 支持条件性结论。例如 `artifacts/phase/summary.json` 显示：

- MNIST 最佳 clean experiment 是 `mnist_vib_beta_0_03`，accuracy 为 `0.9939`。
- Fashion-MNIST 最佳 clean experiment 是 `fashion_mnist_vib_beta_0_03`，accuracy 为 `0.9229`。
- CIFAR-10 最佳 clean experiment 是 `cifar10_vib_beta_0_003`，accuracy 为 `0.7041`。
- Fashion-MNIST 的 Gaussian 和 salt-pepper 最佳 robustness 仍来自 CNN baseline。
- CIFAR-10 的 contrast 和 Gaussian 最佳 robustness 也来自 CNN baseline。

这说明项目没有强行得出 “VIB 总是更鲁棒” 的结论，而是展示了 dataset、beta、corruption 共同决定结果的条件性现象。

主要不足：

1. 当前实验主要是单 seed，没有重复实验、置信区间或显著性分析。作为课程项目可以接受，但研究可信度仍有提升空间。
2. CNN baseline 也被赋予了 `useful-compression` phase label，这在语义上不严谨。CNN 没有 VIB 压缩阶段，应单独标记为 `baseline` 或从 phase label 统计中排除。
3. `src/vib_project/phase_metrics.py` 中 `assign_phase_label` 的 unstable 判断顺序存在问题。`if best_cbi > 0.0: return "robustness-specialized"` 位于 `if best_cbi > 0.0 and worst_cbi < 0.0` 之前，导致后一条冲突判断基本不可达。
4. blur corruption 的 severity 映射较粗，当前实现中低强度和高强度分别对应 3x3 / 5x5 均值卷积，实际连续性弱于 Gaussian 或 contrast。

## 三、工程实现评价

`src/vib_project/` 的结构清楚、职责分离较好：

- `data.py` 支持 MNIST、Fashion-MNIST、CIFAR-10，并返回统一的 `DatasetBundle`。
- `models.py` 提供 `CNNClassifier` 和 `VIBClassifier`，VIB 输出 `logits, z, mu, logvar`。
- `losses.py` 实现 VIB loss 和标准正态 KL。
- `evaluate.py` 实现 clean/noise/corruption 评测，以及 confidence、entropy 统计。
- `corruption.py` 实现四类 corruption。
- `latent_diagnostics.py` 计算 class geometry 和 latent drift。
- `phase_runner.py` 编排训练、评测、PCA、confusion matrix、latent diagnostics 和 phase indicators。
- `phase_artifacts.py` 写出 phase artifact contract。

Artifact-driven 架构是项目最大工程优点。`artifacts/phase/<experiment_id>/` 下的 `metrics.json`、`robustness.json`、`latent_geometry.json`、`latent_drift.json`、`phase_indicators.json`、`latent_pca.json`、`confusion_matrix.json` 构成了离线实验、后端、前端、报告之间的清晰数据边界。

需要注意的是，单次 `phase-run` 生成的 phase indicators 缺少跨实验参考；真正可信的 CBI、KL collapse 和 phase labels 依赖 `scripts/summarize_phase_results.py` 统一刷新。这个设计可行，但应在文档和复现实验流程中强调：跑完全量实验后必须运行 summary 脚本。

FastAPI 后端符合“只读 artifact API”的目标。`backend/app.py` 没有在线训练逻辑，主要读取 index、summary 和单实验 artifact 文件。缺失文件返回 404，设计简洁。

React/Vite 前端确实是 artifact-driven：`frontend/src/api.ts` 读取 `/api/phase/*`，`App.tsx` 加载 `phaseIndex` 和 `phaseSummary` 后展示 PhaseLab 区域。不过前端交互深度仍低于设计文档的目标：

- `PhaseMap` 当前更像实验按钮网格，不是真正按 metric 着色的 phase heatmap。
- `CompressionLens` 当前展示单个 experiment 的 clean accuracy、KL proxy、KL collapse score 和 robustness AUC，不是跨 beta linked charts。
- `DiagnosisLab` 展示 latent geometry 和部分 drift rows，但还没有充分结合 latent PCA、confusion matrix 或按 corruption/class 筛选的诊断交互。

## 四、报告评价

中文报告整体结构清楚，课程特色明显。报告从信息瓶颈理论、VIB 目标、KL proxy、corruption robustness、phase indicators、latent diagnostics 逐步展开，随后给出 MNIST、Fashion-MNIST、CIFAR-10 的实验结果，再介绍 artifact-driven dashboard 和局限性。

报告优点：

- 摘要和引言明确说明项目从 VIB 复现扩展为 PhaseLab。
- 方法部分把 `I(Z;Y) - beta I(Z;X)`、VIB loss、`average_kl` proxy、CBI、KL collapse score 连接起来。
- 结果部分没有过度美化 VIB，承认 Fashion-MNIST 与 CIFAR-10 上部分 corruption 最优仍是 CNN baseline。
- 局限性部分明确指出 KL 不是精确互信息、phase label 不是理论定理、CIFAR-10 小模型不能代表最优 benchmark。
- PDF 已编译生成，约 10 页，图表和网页截图文件存在，能支撑课程提交。

主要需要修正的一点是报告局部数字一致性。`report/phaselab_report.tex` 中基础复现实验段落写道 MNIST 最高 clean 是 beta=0.1、Fashion-MNIST 最高 clean 是 beta=0.01；但同一报告后面的表格以及 `artifacts/phase/summary.json` 显示，MNIST 和 Fashion-MNIST 的最佳 clean 都是 beta=0.03，分别为 99.39% 和 92.29%。建议以 `artifacts/phase/summary.json` 为准统一修改。

## 五、下一步优化建议

### 必须修复

1. 修正 `src/vib_project/phase_metrics.py` 中 `assign_phase_label` 的判断顺序，让 `best_cbi > 0 and worst_cbi < 0` 的 unstable 情况能够先被识别。
2. CNN baseline 不应被标为 `useful-compression`。建议将 CNN 的 phase label 改为 `baseline`，或在 phase label counts 中只统计 VIB experiments。
3. 统一 `report/phaselab_report.tex` 中关于最佳 beta 的文字描述，使其与 `artifacts/phase/summary.json` 和表格一致。

### 强烈建议

1. 对关键 beta 区间补充 3 个随机种子的重复实验，至少报告 mean ± std。
2. 在报告中明确说明 phase label counts 是否包含 CNN baseline。
3. 前端将 Phase Map 改成真正的 metric heatmap，支持 clean accuracy、robustness AUC、normalized AUC、KL proxy、Fisher ratio、phase label 切换。
4. Compression Lens 改成同一数据集下跨 beta 的联动曲线，而不是单 experiment 摘要。
5. Diagnosis Lab 增加 latent PCA、confusion matrix、drift-by-class 的交互选择。

### 可选增强

1. 改进 blur severity 映射，使不同 severity 对应更连续的 kernel/sigma。
2. 在 summary 中增加 schema version 和 artifact generation timestamp。
3. 把 `mean_confidence` 与 `mean_entropy` 用进报告，分析压缩是否影响校准或不确定性。
4. 增加 per-corruption degradation slope，而不仅是平均 AUC。
5. 在 dashboard 中加入 artifact provenance，例如实验时间、seed、epochs、artifact path。

### 若继续扩展为研究项目

1. 增加更强 baseline，例如 ResNet-small、Dropout、weight decay、data augmentation。
2. 对 CIFAR-10 引入更标准的 corruption benchmark，例如 CIFAR-C 子集。
3. 做更系统的 seed 重复和统计显著性检验。
4. 探索 adaptive beta 或 target-KL controller，检验是否能自动避开 over-compression。
5. 扩展理论分析，但继续保持边界：KL 是 variational proxy，不应声称精确估计 `I(X;Z)`。

## 六、最终结论与评分

综合评价：本项目已经达到优秀课程项目水平。它的优势在于研究问题清楚、信息论 framing 较稳、实验矩阵真实、artifact-driven 工程闭环完整、报告表达成熟。主要短板集中在统计可靠性、phase label 语义和报告局部一致性。

评分建议：

- 课程项目完成度：9/10
- 信息论相关性：8.5/10
- 实验可信度：7.5/10
- 工程完整度：8.5/10
- 报告表达质量：8.5/10

总体等级：A- / 优秀。

提交前最值得处理的是三个小而关键的问题：修正 phase label 逻辑、单独处理 CNN baseline 标签、统一报告中的最佳 beta 描述。修完之后，项目的可信度和提交观感会更稳。
