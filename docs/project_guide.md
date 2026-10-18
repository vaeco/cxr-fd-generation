# 架构与FD实现

## 数据与模型

现存实现使用No Finding和Pleural Effusion两类正位胸片，数据清单按患者划分。生成器将 `[B,128]` 噪声与同维类别嵌入拼接，经全连接得到4×4特征，逐级上采样为 `[B,1,128,128]`。投影判别器将图像特征与条件嵌入做内积，并与无条件评分相加。

判别器使用hinge目标，生成器使用负生成评分并按分支加入FD辅助损失。更新判别器时生成图detach；更新生成器时需保留判别器对生成图的输入梯度。

## 医学特征与FD

`src/models.py::CXRFeatureClassifier`基于ResNet-18，提取256维投影特征。`freeze_module`关闭参数更新并切换eval状态，但生成图的特征提取仍保留输入梯度。

```text
G → generated image → frozen feature encoder → mean/covariance → FD → backward to G
```

距离采用两个高斯特征近似的Fréchet距离：

```text
FD = ||mu_g-mu_r||² + tr(Sigma_g+Sigma_r
     -2*(Sigma_r^(1/2)*Sigma_g*Sigma_r^(1/2))^(1/2))
```

`src/fd_loss.py`对统计使用双精度、对称化与小对角扰动，利用特征值计算矩阵平方根的迹。真实参考统计可预先计算；当前生成特征不能整体detach或放在no_grad中。

## FeatureQueue

队列维护历史特征、特征和与外积和。计算当前损失时，先减去即将淘汰的旧项，再加当前可微特征，得到队列替换后的均值和协方差。历史特征为常数，梯度只回传当前生成批。`backward/step`之后调用`enqueue`正式更新缓存。

队列扩大统计支持，但存在生成器分布变化引起的历史滞后，不等于扩大当前反传batch。当前实现还包含FD stop-gradient归一化、固定lambda与warmup。Core5完整稳定化策略及EMA不在这套源码内。

## 执行顺序

1. `scripts/prepare_manifest.py`：清单、类别平衡与患者划分。
2. `scripts/train_classifier.py`：训练医学编码器。
3. `scripts/compute_real_stats.py`：保存真实特征均值、协方差和样本数。
4. `scripts/train_gan.py`：共同预训练、Base与FD分支、检查点恢复。
5. `scripts/evaluate.py`：固定评价设置并输出分布及条件指标。

具体参数见各入口的`--help`和 [运行手册](../README_RUNBOOK_CN.md)。不同协议的数据规模、训练预算和特征空间不能混合比较。
