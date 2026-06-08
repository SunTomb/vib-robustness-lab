# 基于变分信息瓶颈的神经网络表示压缩与鲁棒泛化分析

## 摘要

说明项目复现 VIB，研究 beta 对表示压缩、分类准确率和噪声鲁棒性的影响，并实现交互式 Web demo。

## 1 引言

- 信息论关注不确定性、编码、压缩与传输。
- 深度学习表示可以理解为从输入 X 到表示 Z 再到标签 Y 的信息处理过程。
- 本文问题：适度压缩是否能提升鲁棒泛化？

## 2 信息论背景

- 熵 H(X)。
- 互信息 I(X;Z)、I(Z;Y)。
- 信息瓶颈目标 max I(Z;Y) - beta I(X;Z)。
- KL proxy 与变分上界。

## 3 方法

- CNN baseline。
- VIB-CNN。
- Reparameterization trick。
- Loss = CE + beta * KL。

## 4 实验设置

- 数据集：MNIST、Fashion-MNIST，进阶 CIFAR-10。
- beta 网格。
- Gaussian noise sigma 网格。
- 指标：clean accuracy、noisy accuracy、train-test gap、average KL、latent PCA。

## 5 结果与分析

- CNN vs VIB。
- beta vs KL proxy。
- beta vs clean accuracy。
- beta vs noisy accuracy。
- information plane。
- latent space 可视化。

## 6 Web 交互系统

- 系统架构。
- Artifact 驱动设计。
- Dashboard、Information Plane、Latent Space、Robustness Lab。

## 7 局限性

- KL 是 proxy，不是精确互信息。
- MNIST/Fashion-MNIST 简化。
- CIFAR-10 可能需要更强模型。
- Adaptive beta 是扩展方向。

## 8 结论

总结信息瓶颈如何连接表示压缩、任务预测与鲁棒泛化。
