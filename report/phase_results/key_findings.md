# VIB PhaseLab 关键发现

## cifar10

- 最佳 clean experiment：`cifar10_vib_beta_0_003`，accuracy `0.7041`。
- 各 corruption 最佳 robust experiment：`{'blur': 'cifar10_vib_beta_1em05', 'contrast': 'cifar10_cnn_baseline', 'gaussian': 'cifar10_cnn_baseline', 'salt_pepper': 'cifar10_vib_beta_0_01'}`。
- phase label 计数：`{'useful-compression': 7, 'under-regularized': 4, 'unstable': 1, 'over-compressed': 1}`。

## fashion_mnist

- 最佳 clean experiment：`fashion_mnist_vib_beta_0_03`，accuracy `0.9229`。
- 各 corruption 最佳 robust experiment：`{'blur': 'fashion_mnist_vib_beta_1em05', 'contrast': 'fashion_mnist_vib_beta_0_003', 'gaussian': 'fashion_mnist_cnn_baseline', 'salt_pepper': 'fashion_mnist_cnn_baseline'}`。
- phase label 计数：`{'useful-compression': 3, 'under-regularized': 2, 'unstable': 7, 'over-compressed': 1}`。

## mnist

- 最佳 clean experiment：`mnist_vib_beta_0_03`，accuracy `0.9939`。
- 各 corruption 最佳 robust experiment：`{'blur': 'mnist_vib_beta_0_0003', 'contrast': 'mnist_vib_beta_3em05', 'gaussian': 'mnist_vib_beta_0', 'salt_pepper': 'mnist_vib_beta_0_03'}`。
- phase label 计数：`{'useful-compression': 7, 'under-regularized': 1, 'unstable': 4, 'over-compressed': 1}`。
