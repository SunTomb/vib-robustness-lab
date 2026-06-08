# 信息论课程报告

## VIB PhaseLab：基于变分信息瓶颈的表示压缩、鲁棒性相变与诊断系统

**姓名**：叶盛豪 &emsp; **学号**：PB24000227  
**日期**：2026 年 6 月 8 日

---

## 摘要

信息瓶颈（Information Bottleneck, IB）理论将表示学习描述为“保留任务相关信息、压缩输入冗余信息”的优化问题。变分信息瓶颈（Variational Information Bottleneck, VIB）进一步将这一思想转化为可训练的神经网络目标，通过在交叉熵损失之外加入 KL 正则项，使模型学习随机潜变量表示，并用瓶颈系数 β 控制表示压缩强度。

本报告围绕“表示压缩何时提升泛化与鲁棒性，何时破坏任务信息”这一问题，完成了 VIB-CNN 的复现、扩展实验和交互式诊断系统构建。基础复现实验比较普通 CNN 与 VIB-CNN 在 MNIST 和 Fashion-MNIST 上的 clean accuracy、Gaussian noise 鲁棒性、KL proxy 与 latent PCA 结构。进一步地，本人设计并实现了 **VIB PhaseLab**：在 MNIST、Fashion-MNIST 和 CIFAR-10 三个数据集上，系统扫描 CNN baseline 与 VIB-CNN 的 12 个 β 配置，并在 Gaussian、salt-and-pepper、blur、contrast 四类 corruption 与五个 severity 上进行鲁棒性评测，共得到 39 个真实实验配置和 780 条 corruption robustness 评测记录。

实验结果表明，β 对表示压缩具有显著控制作用，但压缩并不自动等价于鲁棒性提升。MNIST 上，β=0.03 的 VIB-CNN 达到最高 clean accuracy 99.39%，并在 salt-and-pepper corruption 上取得最佳 robustness AUC 0.9474；Fashion-MNIST 上，β=0.03 取得最高 clean accuracy 92.29%，但 Gaussian 与 salt-and-pepper 的最佳鲁棒性仍来自 CNN baseline；CIFAR-10 上，β=0.003 取得最高 clean accuracy 70.41%，而不同 corruption 的最优模型分散在 CNN baseline 与不同 β 的 VIB-CNN 之间。上述现象说明，VIB 更适合被视为一种可诊断、可调节的压缩正则化机制，而不是通用鲁棒性保证。

除离线实验外，本项目还实现了 artifact-driven 的 FastAPI + React/Vite 中文交互式 Dashboard，支持 Phase overview、β phase map、Compression Lens 和 Diagnosis Lab 等视图。所有报告数值均来自真实实验 artifacts，未手工编造。

**关键词**：信息瓶颈；变分信息瓶颈；表示压缩；鲁棒泛化；相变分析；VIB PhaseLab

---

## 1. 引言

深度神经网络的中间表示通常同时包含任务相关信息和输入细节信息。对于图像分类任务而言，模型既需要保留类别判别所需的结构信息，又可能无意中记住背景、纹理、噪声和训练集偶然模式。信息论提供了一种分析这一现象的视角：如果将输入记为 $X$，标签记为 $Y$，中间表示记为 $Z$，那么好的表示应当在保留关于 $Y$ 的信息的同时，尽可能压缩关于 $X$ 的冗余信息。

信息瓶颈目标可写为：

$$
\max I(Z;Y) - \beta I(Z;X),
$$

其中 $I(Z;Y)$ 衡量表示中的任务相关信息，$I(Z;X)$ 衡量表示中保留的输入信息，$\beta$ 控制压缩强度。直观上，较小的 $\beta$ 允许模型保留更多输入细节，较大的 $\beta$ 则强迫模型学习更紧凑的表示。然而，在实际神经网络中，压缩与泛化之间的关系并不总是单调的：适度压缩可能去除噪声和冗余，提高泛化；过度压缩则可能抹去标签相关信息，导致欠拟合。

本文关注的问题是：**VIB 的压缩何时是 useful compression，何时变成 destructive over-compression？** 为回答这一问题，本项目不只复现 VIB-CNN，而是构建了一个完整的 PhaseLab 实验系统：

1. 从信息瓶颈目标出发，实现普通 CNN 与 VIB-CNN 的可复现实验管线；
2. 在 MNIST、Fashion-MNIST 和 CIFAR-10 上扫描多组 β，比较 clean accuracy、KL proxy 和多类型 corruption 鲁棒性；
3. 设计 robustness AUC、compression benefit index、KL collapse score 与 phase label，刻画表示压缩的阶段变化；
4. 计算 latent geometry 与 latent drift，分析压缩对表示空间结构的影响；
5. 构建 FastAPI + React/Vite 的中文 Dashboard，使实验结果可以交互式展示和复核。

本项目的主要贡献包括：

- 完整复现 VIB-CNN 在图像分类中的表示压缩思想，并生成可复查的离线 artifacts；
- 将原本单一 Gaussian noise 评测扩展为四类 corruption、多 severity、多数据集的 PhaseLab 实验矩阵；
- 提出并实现 phase indicators，用实验标签区分压缩不足、有效压缩、鲁棒特化、过度压缩和不稳定状态；
- 引入 latent geometry 和 latent drift 诊断，解释不同 β 下表示空间的结构变化；
- 构建 artifact-driven 的 Web 系统，使训练、展示和报告三者解耦；
- 在实验室 GPU/NAS 环境完成真实全量实验，并将所有报告结论绑定到真实 artifacts。

---

## 2. 相关工作

**信息瓶颈理论。** 信息瓶颈理论最早将表示学习表述为压缩输入信息并保留任务相关信息的优化问题。对于监督学习任务，理想表示 $Z$ 应当舍弃与标签无关的输入扰动，同时保留足够的类别判别信息。这一理论为理解深度神经网络的泛化能力提供了信息论解释框架。

**变分信息瓶颈。** Alemi 等人提出的 Variational Information Bottleneck 将 IB 目标转化为可优化的变分形式。其核心做法是令编码器输出潜变量分布 $q_\phi(z|x)$，并用先验 $p(z)$ 约束潜变量，使 $KL(q_\phi(z|x)||p(z))$ 成为 $I(X;Z)$ 的可计算上界。这样，神经网络训练目标可以写成普通分类损失与 KL 正则项的加权和。

**鲁棒泛化与输入扰动。** 模型在 clean test set 上表现良好并不意味着它对输入扰动稳健。Gaussian noise、salt-and-pepper noise、blur 和 contrast corruption 分别破坏像素值、局部像素、空间细节和整体强度分布。不同 corruption 对模型依赖的图像特征提出了不同挑战，因此多 corruption 评测比单一噪声更能揭示表示的鲁棒性结构。

**表示诊断。** 除 accuracy 外，中间表示的几何结构也能反映模型是否学到有意义的类别分离。类内方差、类间距离、Fisher ratio、silhouette score 和 corruption 下 latent drift 可以从不同角度描述表示空间的紧凑性、分离度和扰动敏感性。这些指标不能替代预测性能，但能够帮助解释为什么某些 β 表现为有效压缩，而另一些 β 导致过度压缩或不稳定。

---

## 3. 方法

### 3.1 CNN baseline

普通 CNN baseline 使用卷积编码器将输入图像映射为 latent representation，再通过线性分类器输出类别 logits。训练目标为标准交叉熵：

$$
\mathcal{L}_{\text{CNN}} = CE(\hat{y}, y).
$$

该模型不包含显式信息瓶颈，因此可作为同结构容量下的无压缩基线。对于 PhaseLab 指标，CNN baseline 的 normalized robustness AUC 被作为同数据集的 CBI 参考。

### 3.2 VIB-CNN

VIB-CNN 使用与 CNN baseline 相近的卷积骨干，但编码器不直接输出确定性 latent，而是输出均值 $\mu(x)$ 与对数方差 $\log\sigma^2(x)$，定义潜变量分布：

$$
q_\phi(z|x)=\mathcal{N}(\mu(x), \operatorname{diag}(\sigma^2(x))).
$$

训练时使用重参数化技巧采样：

$$
z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I).
$$

分类器基于 $z$ 输出 logits。损失函数为：

$$
\mathcal{L}_{\text{VIB}} = CE(\hat{y}, y) + \beta KL(q_\phi(z|x)||p(z)),
$$

其中 $p(z)=\mathcal{N}(0,I)$。本文报告中的 `average_kl` 是测试集上的平均 KL proxy，用作 $I(X;Z)$ 的变分上界近似，而不是精确互信息估计。

### 3.3 多 corruption 鲁棒性评测

PhaseLab 覆盖四类输入扰动：

| Corruption | 参数含义 | 作用 |
|---|---:|---|
| Gaussian | $\sigma \in [0,0.4]$ | 向输入加入高斯噪声 |
| Salt-and-pepper | probability | 随机将像素置为 0 或 1 |
| Blur | severity | 使用均值卷积削弱局部细节 |
| Contrast | factor | 改变图像相对 0.5 的对比度 |

每类 corruption 使用 severity $\{0.0,0.1,0.2,0.3,0.4\}$。对于每个实验配置，PhaseLab 记录每个 corruption/severity 下的 accuracy、mean confidence 和 mean entropy。

### 3.4 Phase indicators

为避免只用 clean accuracy 判断模型好坏，本项目设计了四类 phase indicators。

**Robustness AUC.** 对同一 corruption 下不同 severity 的 accuracy 求平均：

$$
AUC_c = \frac{1}{|S|}\sum_{s\in S} Acc(c,s).
$$

**Normalized robustness AUC.** 为排除 clean accuracy 差异的影响，将 robustness AUC 除以 clean accuracy：

$$
\widetilde{AUC}_c = \frac{AUC_c}{Acc_{clean}}.
$$

**Compression Benefit Index (CBI).** 使用同一数据集的 CNN baseline 作为参考：

$$
CBI_c = \widetilde{AUC}_{c,\text{VIB}} - \widetilde{AUC}_{c,\text{CNN}}.
$$

当 CBI 为正时，表示在该 corruption 上相对 CNN baseline 具有更好的归一化鲁棒性。

**KL collapse score.** 使用同一数据集 VIB β=0 的 KL 作为参考：

$$
\text{KL-collapse} = \frac{KL_\beta}{KL_{\beta=0}}.
$$

当该值过低且 accuracy 大幅下降时，说明潜变量接近先验但任务信息也被破坏，可视为过度压缩。

**Phase label.** 系统根据 clean accuracy drop、CBI 和 KL collapse score 给每个实验配置赋予诊断标签：`under-regularized`、`useful-compression`、`robustness-specialized`、`over-compressed` 或 `unstable`。该标签是实验诊断工具，不是理论定理。

### 3.5 Latent diagnostics

PhaseLab 进一步计算 latent 表示的几何指标：

- 类内方差：衡量同类样本在 latent 空间中的紧凑程度；
- 类间距离：衡量不同类别中心之间的平均距离；
- Fisher ratio：类间距离与类内方差的比例；
- Silhouette score：无监督聚类分离度；
- Latent drift：同一批样本在 clean 和 corrupted 输入下 latent 表示的平均位移。

这些指标用于解释 accuracy 与 robustness AUC 背后的表示结构。

---

## 4. 实验设置

### 4.1 硬件与软件环境

全量 PhaseLab 实验在实验室 GPU/NAS 环境中完成：

| 配置项 | 详情 |
|---|---|
| 服务器 | Tang-2-Wu |
| GPU | NVIDIA A40 48GB（使用单卡） |
| Python 环境 | `/data/wujcan/yesh/miniconda3/envs/tn_env` |
| PyTorch / Torchvision | 2.7.1+cu126 / 0.22.1+cu126 |
| 运行目录 | `/NAS/yesh/vib-robustness-lab` |
| 关键环境变量 | `VIB_DISABLE_CUDNN=1`, `PYTHONPATH=src` |
| 前端 | React + TypeScript + Vite + Recharts |
| 后端 | FastAPI |

实际运行命令为：

```bash
source /data/wujcan/yesh/miniconda3/etc/profile.d/conda.sh
conda activate tn_env
cd /NAS/yesh/vib-robustness-lab
export VIB_DISABLE_CUDNN=1 PYTHONPATH=src
CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase_experiments.sh
```

远端运行日志已同步至 `logs/phase_full_20260608_001753.log`。

### 4.2 数据集与实验矩阵

PhaseLab 覆盖三个数据集：MNIST、Fashion-MNIST 和 CIFAR-10。MNIST 与 Fashion-MNIST 是灰度图像分类任务，CIFAR-10 是彩色自然图像分类任务，三者由易到难形成了较好的实验梯度。

实验矩阵如下：

| 维度 | 配置 |
|---|---|
| 数据集 | MNIST, Fashion-MNIST, CIFAR-10 |
| 模型 | CNN baseline, VIB-CNN |
| β grid | 0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1 |
| Corruption | Gaussian, salt-and-pepper, blur, contrast |
| Severity | 0.0, 0.1, 0.2, 0.3, 0.4 |
| Epoch | 10 |
| Latent dimension | 32 |

每个数据集包含 1 个 CNN baseline 和 12 个 VIB-CNN 配置，因此总计 $3\times 13=39$ 个实验配置。每个配置在 4 类 corruption 和 5 个 severity 上评测，因此总计 $39\times 4\times 5=780$ 条 robustness 记录。

### 4.3 Artifact-driven 实验流程

项目采用静态 artifacts 作为稳定边界。离线训练代码只负责生成 JSON 文件；后端和前端只读取这些文件，不在线启动训练。PhaseLab 每个实验目录包含：

```text
metrics.json
robustness.json
latent_geometry.json
latent_drift.json
phase_indicators.json
latent_pca.json
confusion_matrix.json
```

这种设计使实验结果可以被验证、复现和部署，也避免 Web Demo 依赖 GPU。

---

## 5. 基础 VIB 复现实验

在扩展 PhaseLab 之前，本项目首先完成了 MNIST 与 Fashion-MNIST 上的基础 VIB-CNN 复现实验。该阶段使用 CNN baseline 和 β ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1, 1} 的 VIB-CNN，并使用 Gaussian noise σ ∈ {0.0, 0.1, 0.2, 0.3, 0.4} 评测鲁棒性。

### 5.1 MNIST 结果

MNIST 上，普通 CNN baseline 的 clean accuracy 为 98.81%，train-test gap 为 0.76%。VIB-CNN 在 β=0.1 时取得最高 clean accuracy 99.38%，比 CNN baseline 高 0.57 个百分点；此时 KL proxy 为 4.7760，明显低于 β=0 时的 872.7278，说明适度信息瓶颈能够在保持分类性能的同时显著压缩表示。

当 β=1 时，MNIST clean accuracy 降至 64.10%，KL proxy 降至 0.0433。这是典型的过压缩现象：潜变量几乎被推向先验分布，但类别相关信息也被大量抹去。

### 5.2 Fashion-MNIST 结果

Fashion-MNIST 上，CNN baseline 的 clean accuracy 为 90.50%，train-test gap 为 2.82%。VIB-CNN 在 β=0.01 时取得最高 clean accuracy 92.17%，比 CNN baseline 高 1.67 个百分点；此时 KL proxy 为 11.5234，相比 β=0 的 316.1828 已经大幅下降。

当 β=1 时，Fashion-MNIST clean accuracy 降至 60.14%，KL proxy 降至 0.0210。这说明过大 β 在两个数据集上都会造成任务信息丢失。

### 5.3 基础实验结论

基础复现实验验证了 VIB 的核心现象：β 能显著控制 KL proxy，适度 β 可能提升 clean generalization，而过大 β 会导致过度压缩。然而，单一 Gaussian noise 评测无法充分说明鲁棒性结构，因此需要进一步扩展到多 corruption、多数据集、多 β 的 PhaseLab。

---

## 6. PhaseLab 全量实验结果

### 6.1 总体结果

PhaseLab 全量实验覆盖 3 个数据集、39 个实验配置和 780 条 robustness 记录。每个数据集的 clean accuracy 最佳配置如下：

| 数据集 | 最佳 clean 实验 | 模型 | β | Clean accuracy | Average KL proxy | Phase label | KL collapse score |
|---|---|---|---:|---:|---:|---|---:|
| MNIST | `mnist_vib_beta_0_03` | VIB-CNN | 0.03 | 99.39% | 7.0490 | useful-compression | 0.0081 |
| Fashion-MNIST | `fashion_mnist_vib_beta_0_03` | VIB-CNN | 0.03 | 92.29% | 7.5336 | unstable | 0.0238 |
| CIFAR-10 | `cifar10_vib_beta_0_003` | VIB-CNN | 0.003 | 70.41% | 25.8218 | useful-compression | 0.1398 |

三个数据集的最佳 clean accuracy 均由 VIB-CNN 取得，说明适当 β 的信息瓶颈确实能发挥正则化作用。不过，最佳 clean 配置并不一定是所有 corruption 下最鲁棒的配置，尤其在 Fashion-MNIST 与 CIFAR-10 上表现明显。

### 6.2 MNIST：有效压缩较稳定

MNIST 的 phase label 分布为：`useful-compression` 7 个、`under-regularized` 1 个、`unstable` 4 个、`over-compressed` 1 个。说明在简单手写数字任务中，较宽范围的 β 都能落入有效压缩区间，只有极大 β 会导致过压缩。

MNIST 各 corruption 的 robustness AUC 最优配置如下：

| Corruption | 最优实验 | β | Robustness AUC | CBI | Clean accuracy |
|---|---|---:|---:|---:|---:|
| blur | `mnist_vib_beta_0_0003` | 0.0003 | 0.9799 | 0.0081 | 99.15% |
| contrast | `mnist_vib_beta_3em05` | 0.00003 | 0.9899 | 0.0023 | 99.17% |
| gaussian | `mnist_vib_beta_0` | 0 | 0.9394 | 0.0107 | 98.96% |
| salt_pepper | `mnist_vib_beta_0_03` | 0.03 | 0.9474 | 0.0025 | 99.39% |

可以看到，MNIST 上四类 corruption 的最优鲁棒性均由 VIB-CNN 获得，且 CBI 均为正。这说明在 MNIST 这类低复杂度任务中，VIB 压缩较容易去除冗余输入变化，同时保留类别结构。

### 6.3 Fashion-MNIST：压缩收益更依赖扰动类型

Fashion-MNIST 的 phase label 分布为：`useful-compression` 3 个、`under-regularized` 2 个、`unstable` 7 个、`over-compressed` 1 个。与 MNIST 相比，Fashion-MNIST 的不稳定配置更多，说明服饰图像的类别边界更依赖细粒度纹理和形状，压缩强度稍有不当就可能损害某些 corruption 下的表现。

Fashion-MNIST 各 corruption 的 robustness AUC 最优配置如下：

| Corruption | 最优实验 | β | Robustness AUC | CBI | Clean accuracy |
|---|---|---:|---:|---:|---:|
| blur | `fashion_mnist_vib_beta_1em05` | 0.00001 | 0.8067 | 0.0062 | 91.50% |
| contrast | `fashion_mnist_vib_beta_0_003` | 0.003 | 0.7408 | 0.0171 | 91.98% |
| gaussian | `fashion_mnist_cnn_baseline` | 0 | 0.6302 | 0.0000 | 90.50% |
| salt_pepper | `fashion_mnist_cnn_baseline` | 0 | 0.6859 | 0.0000 | 90.50% |

VIB 在 blur 和 contrast 上取得最优，但 Gaussian 与 salt-and-pepper 的最优结果仍来自 CNN baseline。这说明 VIB 对不同扰动类型的影响并不一致：它可能改善对平滑和对比度变化的适应性，但未必能抵抗像素级噪声破坏。

### 6.4 CIFAR-10：复杂任务上的混合结论

CIFAR-10 的 phase label 分布为：`useful-compression` 7 个、`under-regularized` 4 个、`unstable` 1 个、`over-compressed` 1 个。CIFAR-10 图像复杂度更高，小型 CNN/VIB-CNN 的绝对 accuracy 不高，但仍能观察到 β 对压缩和鲁棒性结构的影响。

CIFAR-10 各 corruption 的 robustness AUC 最优配置如下：

| Corruption | 最优实验 | β | Robustness AUC | CBI | Clean accuracy |
|---|---|---:|---:|---:|---:|
| blur | `cifar10_vib_beta_1em05` | 0.00001 | 0.4889 | 0.0384 | 69.59% |
| contrast | `cifar10_cnn_baseline` | 0 | 0.6625 | 0.0000 | 70.20% |
| gaussian | `cifar10_cnn_baseline` | 0 | 0.4116 | 0.0000 | 70.20% |
| salt_pepper | `cifar10_vib_beta_0_01` | 0.01 | 0.4645 | 0.0031 | 70.04% |

CIFAR-10 上，VIB 在 blur 和 salt-and-pepper 上略有优势，但 contrast 和 Gaussian 最优仍是 CNN baseline。这说明在复杂自然图像上，表示压缩的效果受模型容量、数据分布和扰动类型共同制约。VIB 能提供可调节的正则化，但不能替代更强的模型结构或数据增强策略。

### 6.5 过度压缩现象

三个数据集均出现 1 个 `over-compressed` 配置，通常对应较大的 β。过度压缩的共同特征是 KL collapse score 极低，同时 clean accuracy 明显低于同数据集最佳配置。这验证了信息瓶颈理论中的基本张力：如果只追求降低 $I(X;Z)$ proxy，而不保留足够的 $I(Z;Y)$，模型就会失去判别能力。

---

## 7. 表示空间诊断

PhaseLab 不仅记录预测性能，还记录 latent geometry 和 drift。以 CIFAR-10 CNN baseline 的浏览器验证结果为例，其 latent geometry 指标为：类内方差 8.1162、类间距离 11.8992、Fisher ratio 1.4661、silhouette score 0.0614。silhouette score 较低，说明在小型 CNN 结构下 CIFAR-10 latent 空间仍存在较强类别重叠。

在 corruption 下，Diagnosis Lab 会展示各类别的 latent drift。例如 Gaussian severity 0.1 下，不同类别的 mean drift 大多在 5 到 8 之间，说明即使预测准确率没有立即崩溃，输入扰动也已经在 latent 空间造成明显位移。该现象说明只看 accuracy 容易忽略表示层面的敏感性，而 latent drift 可以作为鲁棒性分析的补充指标。

从系统角度看，latent diagnostics 的意义在于把 VIB 的信息论压缩目标与可视化表示结构连接起来：当 KL proxy 降低、类内方差收缩、类间距离保持时，可以解释为有效压缩；当 KL proxy 接近 0 且 accuracy 大幅下降时，则说明潜变量不再携带足够标签信息。

---

## 8. 交互式系统实现

### 8.1 后端：FastAPI Artifact Service

后端位于 `backend/`，只提供 artifact 读取服务，不执行训练。主要接口包括：

| Endpoint | 功能 |
|---|---|
| `/api/index` | 读取原始 VIB 实验索引 |
| `/api/experiments/{id}` | 读取原始实验 metrics、curves 和 latent PCA |
| `/api/phase/index` | 读取 PhaseLab 实验索引 |
| `/api/phase/summary` | 读取 PhaseLab summary |
| `/api/phase/experiments/{id}` | 读取单个 PhaseLab 实验详情 |
| `/api/phase/diagnostics` | 读取可选诊断扩展 |

后端通过环境变量指定 artifact 路径：

```bash
VIB_ARTIFACT_DIR=artifacts/full VIB_PHASE_ARTIFACT_DIR=artifacts/phase \
  python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

### 8.2 前端：React/Vite 中文 Dashboard

前端位于 `frontend/`，使用 React、TypeScript、Vite 和 Recharts。页面采用中文科研工作台风格，而不是通用深色 dashboard。主要模块包括：

1. **Phase overview**：展示每个数据集的最佳 clean 实验和 phase label 分布；
2. **Phase map**：列出所有数据集和 β 配置，点击后进入诊断视图；
3. **Compression Lens**：显示 clean accuracy、KL proxy、KL collapse score 和 robustness AUC；
4. **Diagnosis Lab**：显示 latent geometry 和 latent drift；
5. **原始 VIB 实验台**：展示基础复现实验的指标卡、信息平面、鲁棒性曲线和 latent PCA。

本地浏览器验证中，前端成功读取真实 `artifacts/phase`，显示 CIFAR-10、Fashion-MNIST 和 MNIST 的 PhaseLab 结果，并能进入 Compression Lens 与 Diagnosis Lab。

### 8.3 Artifact-driven 设计的优势

本项目没有把 Web Demo 做成在线训练平台，而是采用“离线训练—静态 artifacts—只读 API—前端展示”的设计。这种方式有三点优势：

1. **可复现**：报告、网页和后端读取同一批 JSON artifacts；
2. **轻量部署**：展示系统不依赖 GPU，也不需要下载原始数据集；
3. **可信报告**：所有表格和结论都可以追溯到 `artifacts/full` 或 `artifacts/phase`。

---

## 9. 工程实践与问题处理

本项目在实现过程中遇到并解决了多类工程问题。

**环境与 GPU。** 本地主要用于开发和测试，完整实验在实验室 Tang-2-Wu A40 GPU 上运行。由于 A40 环境曾出现 cuDNN 初始化问题，训练脚本支持 `VIB_DISABLE_CUDNN=1`，最终全量 PhaseLab 实验使用该设置完成。

**远端运行与监控。** 长连接 SSH Monitor 在本次实验中不稳定，曾多次因连接中断退出。实际训练进程并未失败，因此后续改为一次性 SSH 检查和定时唤醒，更适合长时间实验监控。

**Artifact 合约。** PhaseLab 初始实现中，每个实验独立计算 phase indicators，导致 CBI 和 KL collapse 缺乏跨实验参考。后续在 `scripts/summarize_phase_results.py` 中增加全局刷新逻辑：CBI 使用同数据集 CNN baseline，KL collapse 使用同数据集 VIB β=0，best accuracy 使用同数据集最佳 clean accuracy。

**前端代理。** 本地验证时发现 8000 端口被占用，而 Vite proxy 固定指向 8000，导致页面卡在读取状态。后续将 `frontend/vite.config.ts` 改为支持 `VITE_API_PROXY_TARGET`，使前端可以连接任意后端端口。

**报告生成。** 为避免手工改写数值，本项目提供 `scripts/generate_phaselab_report.py` 从 artifacts 和 summary 自动生成报告。后续如重新跑实验，应先运行 summary exporter，再生成报告。

---

## 10. 最终验证

最终验证命令包括：

```bash
python -m pytest -v
python scripts/validate_phase_artifacts.py artifacts/phase
python scripts/summarize_phase_results.py --artifact-root artifacts/phase --output-dir report/phase_results
python scripts/generate_phaselab_report.py --phase-root artifacts/phase --result-root report/phase_results --output report/phaselab_report.md
npm --prefix frontend test -- src/lib/metrics.test.ts src/lib/phaseMetrics.test.ts
npm --prefix frontend run build
```

验证结果如下：

| 验证项 | 结果 |
|---|---|
| Python / backend tests | 52 passed, 1 warning |
| Phase artifacts validator | 3 datasets, 39 experiments, missing `[]` |
| Phase summary exporter | 39 metric rows, 780 robustness rows, 39 phase rows |
| Frontend tests | 6 passed |
| Frontend build | passed，存在可接受的 Recharts chunk-size warning |
| 浏览器验证 | PhaseLab overview、phase map、Compression Lens、Diagnosis Lab 均可读取真实数据 |

---

## 11. 局限性

本项目仍有以下局限：

1. **KL 只是 proxy。** `average_kl` 是 $I(X;Z)$ 的变分上界近似，不是精确互信息估计。报告中所有信息论解释都应理解为 proxy 层面的实验分析。
2. **模型结构较小。** CIFAR-10 上使用的小型 CNN/VIB-CNN 只能作为课程实验和相变诊断样例，不能代表最优图像分类性能。
3. **Phase label 是启发式。** `useful-compression`、`over-compressed` 等标签基于经验阈值和相对指标，不应解释为严格理论判定。
4. **未覆盖对抗攻击。** 本项目研究的是常见 corruption 鲁棒性，不涉及 adversarial attack 或安全攻击场景。
5. **Adaptive β 仍是扩展方向。** 项目实现了 warmup / target-KL helper，但未将 adaptive β 作为全量训练策略系统比较。

---

## 12. 结论

本文从信息瓶颈理论出发，复现并扩展了 VIB-CNN 在图像分类表示压缩中的应用。实验表明，β 能显著控制 KL proxy，适度压缩可以提升 clean accuracy，并在部分 corruption 上改善归一化鲁棒性；但压缩并不自动带来全面鲁棒性提升，且过大 β 会造成任务信息丢失。

VIB PhaseLab 的核心价值在于把单一复现实验扩展为可诊断的实验系统：它不仅回答“VIB 是否有效”，还进一步分析“在哪个数据集、哪个 β、哪类扰动下有效，以及是否出现过度压缩”。MNIST 上，VIB 的有效压缩区间较宽，四类 corruption 的最优鲁棒性均来自 VIB；Fashion-MNIST 和 CIFAR-10 上，最优配置随 corruption 类型变化，说明复杂任务中的压缩—鲁棒性关系更加细致。

从课程项目角度看，本项目形成了完整闭环：实验室 GPU 离线训练生成 artifacts，FastAPI 后端只读服务提供数据，React/Vite 前端交互式展示，中文报告从真实结果中归纳结论。这一闭环保证了实验结果的可复查性，也使信息论理论、神经网络实验和工程系统实现结合在一起。

---

## 参考文献

[1] Tishby, N., Pereira, F. C., & Bialek, W. The Information Bottleneck Method. 1999.

[2] Alemi, A. A., Fischer, I., Dillon, J. V., & Murphy, K. Deep Variational Information Bottleneck. ICLR, 2017.

[3] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE, 1998.

[4] Xiao, H., Rasul, K., & Vollgraf, R. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. 2017.

[5] Krizhevsky, A. Learning Multiple Layers of Features from Tiny Images. 2009.

[6] Hendrycks, D., & Dietterich, T. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR, 2019.
