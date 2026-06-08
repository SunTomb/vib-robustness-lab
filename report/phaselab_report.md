# VIB PhaseLab：表示压缩的鲁棒性相变与诊断系统

## 摘要

本报告基于真实 `artifacts/phase` 离线实验结果，扩展原 VIB 复现实验为 PhaseLab：在多数据集、多 β、多 corruption 条件下观察表示压缩从有效正则化到过度压缩的相变现象。文中 accuracy、KL proxy、robustness AUC、compression benefit index 和 phase label 均从实验 artifact 读取，未手工编造数值。

## 1 研究问题与方法

PhaseLab 关注的问题是：VIB 的压缩何时是 useful compression，何时变成 destructive over-compression。实验以 KL(q(z|x)||p(z)) 作为 I(X;Z) 的变分上界 proxy，以 clean accuracy 和 corruption accuracy 作为任务信息与鲁棒性的经验 proxy。

Phase indicators 使用同一数据集内的 CNN baseline 作为 normalized robustness AUC 参考，使用 VIB β=0 作为 KL collapse 参考，并用同数据集最佳 clean accuracy 判断 clean accuracy drop。

## 2 实验矩阵

本轮 PhaseLab artifacts 覆盖 3 个数据集、39 个实验配置。
数据集包括：MNIST、Fashion-MNIST、CIFAR-10。
每个实验在 Gaussian、salt-and-pepper、blur、contrast corruption 与多个 severity 上进行评测。

## 3 结果概览

### CIFAR-10

Clean accuracy 最佳实验为 `cifar10_vib_beta_0_003`，模型为 `vib`，β=0.003，test accuracy 为 70.41%，average KL proxy 为 25.8218。
该实验的 phase label 为 `useful-compression`，KL collapse score 为 0.1398。

各 corruption 的 robustness AUC 最优实验如下：

- `blur`：`cifar10_vib_beta_1em05`，模型 `vib`，β=1e-05，robustness AUC 0.4889，CBI 0.0384，clean accuracy 69.59%。
- `contrast`：`cifar10_cnn_baseline`，模型 `cnn`，β=0，robustness AUC 0.6625，CBI 0.0000，clean accuracy 70.20%。
- `gaussian`：`cifar10_cnn_baseline`，模型 `cnn`，β=0，robustness AUC 0.4116，CBI 0.0000，clean accuracy 70.20%。
- `salt_pepper`：`cifar10_vib_beta_0_01`，模型 `vib`，β=0.01，robustness AUC 0.4645，CBI 0.0031，clean accuracy 70.04%。

Phase label 分布：`{'useful-compression': 7, 'under-regularized': 4, 'unstable': 1, 'over-compressed': 1}`。

### Fashion-MNIST

Clean accuracy 最佳实验为 `fashion_mnist_vib_beta_0_03`，模型为 `vib`，β=0.03，test accuracy 为 92.29%，average KL proxy 为 7.5336。
该实验的 phase label 为 `unstable`，KL collapse score 为 0.0238。

各 corruption 的 robustness AUC 最优实验如下：

- `blur`：`fashion_mnist_vib_beta_1em05`，模型 `vib`，β=1e-05，robustness AUC 0.8067，CBI 0.0062，clean accuracy 91.50%。
- `contrast`：`fashion_mnist_vib_beta_0_003`，模型 `vib`，β=0.003，robustness AUC 0.7408，CBI 0.0171，clean accuracy 91.98%。
- `gaussian`：`fashion_mnist_cnn_baseline`，模型 `cnn`，β=0，robustness AUC 0.6302，CBI 0.0000，clean accuracy 90.50%。
- `salt_pepper`：`fashion_mnist_cnn_baseline`，模型 `cnn`，β=0，robustness AUC 0.6859，CBI 0.0000，clean accuracy 90.50%。

Phase label 分布：`{'useful-compression': 3, 'under-regularized': 2, 'unstable': 7, 'over-compressed': 1}`。

### MNIST

Clean accuracy 最佳实验为 `mnist_vib_beta_0_03`，模型为 `vib`，β=0.03，test accuracy 为 99.39%，average KL proxy 为 7.0490。
该实验的 phase label 为 `useful-compression`，KL collapse score 为 0.0081。

各 corruption 的 robustness AUC 最优实验如下：

- `blur`：`mnist_vib_beta_0_0003`，模型 `vib`，β=0.0003，robustness AUC 0.9799，CBI 0.0081，clean accuracy 99.15%。
- `contrast`：`mnist_vib_beta_3em05`，模型 `vib`，β=3e-05，robustness AUC 0.9899，CBI 0.0023，clean accuracy 99.17%。
- `gaussian`：`mnist_vib_beta_0`，模型 `vib`，β=0，robustness AUC 0.9394，CBI 0.0107，clean accuracy 98.96%。
- `salt_pepper`：`mnist_vib_beta_0_03`，模型 `vib`，β=0.03，robustness AUC 0.9474，CBI 0.0025，clean accuracy 99.39%。

Phase label 分布：`{'useful-compression': 7, 'under-regularized': 1, 'unstable': 4, 'over-compressed': 1}`。

## 4 PhaseLab 系统实现

离线实验层生成 `metrics.json`、`robustness.json`、`latent_geometry.json`、`latent_drift.json` 和 `phase_indicators.json`。
FastAPI 后端只读取 artifacts，不启动训练；React/Vite 前端显示 phase overview、β phase map、compression lens 和 diagnosis lab。
这种 artifact-driven 结构使课程展示页面可复现、可部署，并与 GPU 训练解耦。

## 5 局限性

KL 仍是 variational proxy，不是精确互信息估计；phase label 是用于比较实验现象的诊断标签，不应解释为理论定理。
CIFAR-10 结果受当前小型 CNN/VIB-CNN 结构与训练轮数限制，更适合作为相变诊断样例，而不是最优分类性能 benchmark。

## 6 结论

PhaseLab 将原先的 VIB 复现扩展为可诊断的实验系统：它不仅展示 β 是否影响 accuracy，还比较不同 corruption 下的鲁棒性 AUC、CBI、KL collapse 与 latent geometry。
从系统角度看，本项目形成了 GPU 离线实验、静态 artifacts、后端 API、中文交互 dashboard 和报告之间的闭环。
