# 基于变分信息瓶颈的神经网络表示压缩与鲁棒泛化分析设计文档

## 1. 项目定位

本项目是《信息论》课程大作业，主题为 **信息论 × 深度学习表示学习**。项目复现经典变分信息瓶颈（Variational Information Bottleneck, VIB）的核心思想，并研究表示压缩对图像分类鲁棒泛化的影响。

项目最终交付包括：

1. 可复现实验代码；
2. 普通 CNN 与 VIB-CNN 的对比实验；
3. MNIST 与 Fashion-MNIST 上不同 `β` 的 clean/noisy accuracy、KL proxy、信息平面与 latent 可视化；
4. 一个可交互网页，用于展示 `β` 如何改变压缩强度、分类能力和鲁棒性；
5. 中文课程报告，围绕熵、互信息、信息瓶颈、率失真和泛化进行解释。

推荐题目：

> 基于变分信息瓶颈的神经网络表示压缩与鲁棒泛化分析

## 2. 核心研究问题

项目围绕四个问题组织：

1. **表示压缩**：随着 `β` 增大，VIB 是否会降低表示 `Z` 中关于输入 `X` 的信息量？
2. **任务预测**：压缩是否会降低 `Z` 对标签 `Y` 的预测能力？
3. **鲁棒泛化**：适度压缩是否能减少输入噪声、风格扰动或过拟合带来的性能下降？
4. **可解释展示**：能否把 `β`、KL proxy、accuracy、latent 分布和噪声鲁棒性组织成一个清晰的 Web demo？

## 3. 信息论主线

VIB 将神经网络表示学习写成信息瓶颈问题：

\[
\max I(Z;Y) - \beta I(Z;X)
\]

其中：

- `X` 是输入图像；
- `Y` 是类别标签；
- `Z` 是模型中间表示；
- `I(Z;Y)` 衡量表示对任务标签的保留；
- `I(Z;X)` 衡量表示从输入中保留了多少信息；
- `β` 控制压缩强度。

在可训练神经网络中，严格互信息通常难以直接估计。本项目采用 VIB 的变分形式，用分类交叉熵近似预测项，用

\[
KL(q_\phi(z|x) \| p(z))
\]

作为 `I(X;Z)` 的可优化上界或 proxy。报告中必须明确：项目展示的是 **KL proxy / variational upper bound**，不是精确互信息估计。

训练目标为：

\[
\mathcal{L} = \mathcal{L}_{CE}(\hat{y}, y) + \beta \cdot KL(q_\phi(z|x) \| p(z))
\]

`β` 越大，表示压缩越强；过小 `β` 可能保留输入噪声，过大 `β` 可能丢失任务信息。项目要寻找中间区域是否存在更好的鲁棒泛化折中。

## 4. 范围

### 4.1 必做范围

- 普通 CNN baseline；
- VIB-CNN baseline；
- MNIST 与 Fashion-MNIST；
- 多个固定 `β`：建议 `0, 1e-4, 1e-3, 1e-2, 1e-1, 1`；
- clean accuracy；
- Gaussian noise accuracy；
- train-test gap；
- KL proxy；
- information plane：横轴 KL proxy，纵轴 accuracy 或 estimated `I(Z;Y)` proxy；
- latent t-SNE 或 PCA 可视化；
- 静态 artifact 驱动的 Web demo。

### 4.2 进阶范围

- CIFAR-10；
- salt-and-pepper noise、blur、contrast corruption；
- adaptive `β` 策略；
- 单张图片上传推理；
- 将 Web demo 部署到云服务器或个人域名。

### 4.3 不做范围

- 不做严格互信息精确估计；
- 不做 LLM 或 RAG 主线；
- 不把对抗攻击作为主线；
- 不做在线训练平台；
- 不做用户系统、计费、长期数据库或权限管理。

## 5. 总体架构

项目分为四层。

### 5.1 离线实验层

离线实验层负责训练、评估和产出可信结果。

职责：

- 加载数据集；
- 训练普通 CNN 与 VIB-CNN；
- 评估 clean/noisy/corruption accuracy；
- 计算 KL proxy；
- 导出 latent 表示；
- 生成 JSON artifacts 和图像 artifacts。

离线层是项目可信度核心。网页不在线训练，只展示离线结果和可选单张推理。

### 5.2 后端服务层

后端使用 FastAPI。

职责：

- 读取 `artifacts/` 下的实验结果；
- 提供 REST API 给前端；
- 可选加载 checkpoint，支持样例图片或上传图片推理；
- 提供健康检查接口。

后端不承担训练任务，避免部署环境被 GPU 训练或大数据下载拖垮。

### 5.3 前端展示层

前端使用 React + TypeScript + Vite。

职责：

- 展示项目问题、VIB 公式和信息论解释；
- 展示不同 `β` 的 accuracy、KL proxy、鲁棒性曲线；
- 展示 information plane；
- 展示 latent 分布；
- 支持选择数据集、模型、噪声强度和 `β`；
- 可选支持图片上传推理。

### 5.4 报告产物层

报告使用中文撰写，叙事主线为：

> `β` 控制表示压缩强度；过小 `β` 保留过多输入细节，可能不鲁棒；过大 `β` 损失任务信息；适中 `β` 在压缩和预测之间取得更好的折中。

报告必须包含：

- 课程概念对应关系；
- VIB 公式推导；
- 实验设置；
- 表格与图；
- Web demo 截图；
- 复现与改进说明；
- 局限性。

## 6. 实验设计

### 6.1 数据集

MVP 使用：

- MNIST：低风险、结果清晰，用于验证 VIB 实现；
- Fashion-MNIST：比 MNIST 更难，用于观察泛化和鲁棒性差异。

进阶使用：

- CIFAR-10：更接近真实图像任务，但实验波动更大。

### 6.2 模型

#### 普通 CNN

作为非瓶颈 baseline。结构建议：

- 两层卷积；
- ReLU；
- MaxPool；
- 全连接 latent；
- 分类头。

#### VIB-CNN

结构建议：

- CNN encoder；
- 输出 `mu` 与 `logvar`；
- 使用 reparameterization trick 采样 `z`；
- classifier 从 `z` 预测类别；
- loss = CE + `β * KL`。

### 6.3 `β` 网格

建议固定网格：

```text
0, 1e-4, 1e-3, 1e-2, 1e-1, 1
```

其中 `β = 0` 等价于随机 latent 结构下的无 KL 正则模型，用于观察无压缩时的性能。

### 6.4 鲁棒性测试

MVP 使用 Gaussian noise：

```text
sigma = 0.0, 0.1, 0.2, 0.3, 0.4
```

进阶加入：

- salt-and-pepper；
- blur；
- contrast。

### 6.5 指标

每个 dataset/model/β 组合输出：

- train accuracy；
- test accuracy；
- noisy accuracy by sigma；
- train-test gap；
- average CE；
- average KL；
- average total loss；
- latent embedding samples；
- confusion matrix。

### 6.6 可视化

必须生成：

- `β` vs clean accuracy；
- `β` vs noisy accuracy；
- `β` vs average KL；
- information plane：KL proxy vs accuracy；
- latent 2D projection；
- per-class latent 分布。

## 7. Artifact 格式

所有离线结果统一输出到：

```text
artifacts/
  index.json
  mnist/
    cnn_baseline/
      metrics.json
      curves.json
      latent_pca.json
      confusion_matrix.json
    vib_beta_0.001/
      metrics.json
      curves.json
      latent_pca.json
      confusion_matrix.json
  fashion_mnist/
    ...
```

`index.json` 记录所有可用实验：

```json
{
  "datasets": ["mnist", "fashion_mnist"],
  "experiments": [
    {
      "id": "mnist_vib_beta_0.001",
      "dataset": "mnist",
      "model": "vib",
      "beta": 0.001,
      "path": "mnist/vib_beta_0.001"
    }
  ]
}
```

单个 `metrics.json`：

```json
{
  "dataset": "mnist",
  "model": "vib",
  "beta": 0.001,
  "train_accuracy": 0.992,
  "test_accuracy": 0.985,
  "train_test_gap": 0.007,
  "average_kl": 12.4,
  "average_ce": 0.052,
  "noise_accuracy": {
    "0.0": 0.985,
    "0.1": 0.971,
    "0.2": 0.944,
    "0.3": 0.901,
    "0.4": 0.842
  }
}
```

实际数值由实验生成，不能在报告中使用占位结果。

## 8. Web Demo 设计

### 8.1 页面结构

1. **Project Overview**
   - 项目问题；
   - VIB 公式；
   - `β` 的含义。

2. **Experiment Dashboard**
   - dataset 选择；
   - model/β 选择；
   - clean accuracy、KL、robustness cards；
   - 曲线图。

3. **Information Plane**
   - 横轴 KL proxy；
   - 纵轴 accuracy；
   - 每个点对应一个 `β`；
   - 点击点后更新右侧解释。

4. **Latent Space**
   - PCA/t-SNE 2D 点图；
   - 颜色表示类别；
   - 对比不同 `β` 的聚类变化。

5. **Robustness Lab**
   - 选择 noise sigma；
   - 展示 noisy accuracy；
   - 展示样例图像扰动前后预测。

### 8.2 交互原则

Web demo 展示离线 artifacts，不让用户触发训练。

交互必须服务信息论解释：

- 调大 `β` = 更强压缩；
- KL proxy 下降 = 表示保留输入信息更少；
- accuracy 下降或稳定 = 任务信息是否保留；
- noisy accuracy 改善或恶化 = 鲁棒泛化折中。

## 9. 可选增强：Adaptive β

当固定 β 实验闭环后，可以加入 adaptive β。

思路：给定目标 KL 区间或目标 accuracy 下限，训练中动态调整 `β`：

- 如果 KL 高于目标，增大 `β`；
- 如果 KL 低于目标且 accuracy 下降明显，减小 `β`。

报告中将其定位为课程项目改进方法，而不是保证必然优于固定 β 的新算法。评价重点是：它是否能自动找到接近固定 β 网格中较优点的压缩-预测折中。

## 10. 测试与验证

### 10.1 单元测试

- VIB reparameterization 输出形状正确；
- KL 计算非负；
- Gaussian noise 保持像素范围 `[0, 1]`；
- artifact schema 可被后端读取；
- API 返回结构稳定；
- 前端数据转换函数正确。

### 10.2 集成测试

- 训练一个 tiny subset，能输出 metrics；
- 后端能读取 sample artifacts；
- 前端能显示 sample artifacts；
- full artifact 生成后，dashboard 无缺失图表。

### 10.3 手动验证

- 运行 Web demo；
- 检查 dataset/model/β 切换；
- 检查信息平面点点击；
- 检查 latent 图颜色与类别；
- 检查噪声强度变化是否更新展示。

## 11. 验收标准

MVP 验收：

- MNIST 与 Fashion-MNIST 均完成 CNN baseline 与至少 5 个 VIB β 实验；
- 每个实验有 metrics、noise accuracy、KL proxy；
- 生成 information plane 和 latent projection；
- FastAPI 能读取 artifacts；
- React 前端能展示 dashboard、information plane、latent space、robustness lab；
- 报告中明确解释 VIB 与课程信息论概念的对应关系。

优秀版本验收：

- 加入 CIFAR-10 或 adaptive β；
- Web demo 可部署；
- 报告中有消融、局限性和失败实验分析；
- 能说明为什么某些 β 提升鲁棒性而另一些 β 损害预测能力。

## 12. 风险与应对

### 风险 1：CIFAR-10 效果不清楚

应对：MNIST 和 Fashion-MNIST 是保底主线，CIFAR-10 只作为进阶。

### 风险 2：互信息估计不严谨

应对：全文使用 KL proxy / variational upper bound 表述，不声称精确估计 `I(X;Z)`。

### 风险 3：adaptive β 效果不优

应对：将 adaptive β 定位为可选增强，评价其是否减少调参成本，而不是必须超过所有固定 β。

### 风险 4：网页开发占用过多时间

应对：网页只读取 artifacts，不做在线训练、不做账户系统、不做复杂后端状态。

### 风险 5：实验数量过大

应对：优先完成 MNIST/Fashion-MNIST × CNN/VIB × β 网格；CIFAR-10 与 adaptive β 在主线稳定后再做。

## 13. 推荐时间安排

三周以上时间建议如下：

- 第 1 周：完成数据、模型、训练、MNIST 跑通；
- 第 2 周：完成 Fashion-MNIST、鲁棒性评估、artifact 导出、核心图表；
- 第 3 周：完成 FastAPI、React demo、报告初稿；
- 第 4 周或机动时间：CIFAR-10、adaptive β、部署、报告润色。
