# CXR FD Generation

CheXpert胸部X光条件生成与表征空间Fréchet损失适配。提供二分类条件GAN、可微历史特征队列，以及数据准备、医学特征训练、分布统计和生成评价流程。

## 实现范围

- `No Finding`与`Pleural Effusion`二分类数据清单，按患者互斥划分。
- 128×128单通道条件GAN与投影判别器。
- 冻结医学编码器、类别条件FD与可微FeatureQueue统计。
- 共同预训练检查点初始化的Base/FD分支对照。
- 环境自检、日志、检查点恢复及图像与分布评价入口。

Core5七维条件、EMA统计、完整梯度平衡及FD-SIM后训练的最终脚本和权重未分发。它们的历史结果与可运行的二分类实现分开列于 [实验协议](docs/experiments.md)。本仓库不包含PIXEL完整扩散训练或病灶定位实现。

## 安装与运行

先安装适配设备的PyTorch和torchvision，再安装依赖：

```bash
python -m pip install -r requirements.txt
python -m scripts.prepare_manifest --help
python -m scripts.train_classifier --help
python -m scripts.compute_real_stats --help
python -m scripts.train_gan --help
python -m scripts.evaluate --help
```

数据需按CheXpert访问条款自行准备，仓库不分发影像、患者级数据清单或训练权重。完整二分类配置步骤见 [运行手册](README_RUNBOOK_CN.md)，路径占位符须替换为本地目录。

## 方法

生成图像通过冻结的医学特征网络后，比较生成与真实特征的均值和协方差。历史队列保存detach特征，当前批特征保持可微，反传后再更新缓存。FD作为辅助目标与对抗损失共同训练生成器。实现与梯度路径见 [架构说明](docs/project_guide.md)。

## 历史Core5结果

| 宏平均指标 | Base28 | Queue2048 |
|---|---:|---:|
| FID ↓ | 59.4950 | 57.4315 |
| CXR-FD ↓ | 24.6504 | 14.6423 |
| 严格五病种Exact Match ↑ | 8.29% | 11.40% |

分支从Common20各续训8轮，医学评价采用独立DenseNet-121。该表是历史协议结果，不是当前二分类入口的复现报告。分布距离改善不等于临床有效性，部分病种条件指标仍有下降。详情见 [实验协议与结果](docs/experiments.md)。

## 文档与来源

- [架构与FD实现](docs/project_guide.md)
- [二分类运行手册](README_RUNBOOK_CN.md)
- [实验协议与结果](docs/experiments.md)
- [FD-Loss](https://github.com/Jiawei-Yang/FD-Loss)，Representation Fréchet Loss for Visual Generation
- PIXEL论文：[10.1038/s41551-025-01456-y](https://doi.org/10.1038/s41551-025-01456-y)

## 许可证

代码使用 [MIT License](LICENSE)。方法来源与数据、模型的独立授权见 [第三方声明](THIRD_PARTY_NOTICE.md)。
