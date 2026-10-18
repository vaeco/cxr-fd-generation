# CXR FD Generation

将表征空间 Fréchet 距离用于 CheXpert 条件图像生成的研究项目整理。

## 当前源码能做什么

本包是早期二分类条件 GAN + 可微 FeatureQueue FD 的参考实现，输出 128×128 单通道图像。保留数据清单、分类器、真实统计、生成训练与评测入口。

最终 Core5 多标签、EMA、梯度平衡及 FD-SIM 医学适配脚本/权重缺失，当前代码不能直接复现这些历史结果。相关实验协议与数值在项目档案中单列，绝不把原 FD-loss 仓库当作个人最终源码。

## 准备与入口

```bash
pip install -r requirements.txt
python -m scripts.prepare_manifest --help
python -m scripts.train_classifier --help
python -m scripts.compute_real_stats --help
python -m scripts.train_gan --help
python -m scripts.evaluate --help
```

请先安装匹配的 torch/torchvision。数据需要自行按官方访问要求准备；仓库不包含 CheXpert 数据、病人级清单、模型权重或服务器凭据。`README_RUNBOOK_CN.md` 是早期二分类的历史运行手册，路径为占位符，不代表最终实验复现脚本。

## 参考方法

- FD-loss: https://github.com/Jiawei-Yang/FD-Loss ，arXiv:2604.28190。
- PIXEL: DOI 10.1038/s41551-025-01456-y。使用其研究方向与数据背景，不声明完整复现其扩散与定位系统。

## 本次代码修正

把 FeatureQueue 中两个不合法的 `tensor.to(other_tensor, dtype=...)` 调用改为显式 `device=..., dtype=...`。只做语法与代码审查；未重训、未重新验证历史性能。


## 阅读与面试

- [项目档案](docs/project_guide.md)
- [核心问题与递进追问](docs/interview_qa.md)

## 本次整理的验证边界

采用历史材料，不重跑训练。Python 语法检查不能证明模型效果或历史指标已复现。缺失资源明确保留；源码整理和历史实验结果分别记录。

补充静态修正：为导入 src 的脚本增加项目根目录的模块搜索路径，使直接运行与 -m 入口一致；未运行这些训练脚本。
