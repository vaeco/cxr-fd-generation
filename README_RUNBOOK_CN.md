# CheXpert二分类FD实验运行手册

本手册对应当前二分类条件GAN和FeatureQueue实现，不对应Core5最终多标签实验。数据与检查点需单独准备，所有路径占位符应替换为实际目录。

## 1. 环境与资源

参考环境为Linux、Conda、Python3.11.9、PyTorch2.6.0和CUDA12.4。训练默认使用128×128灰度图与batch64，可按设备资源调整。数据、缓存、日志和检查点建议放在容量充足的数据盘；具体显存与耗时应通过实际配置测量。

## 2. 实验对照

最终只保留两组，不运行“当前 batch FD”等已知不稳定对照：

| 最终实验 | 初始化 | 后训练轮数 | 生成器损失 | 用途 |
|---|---|---:|---|---|
| Base | 同一个 `common_pretrain_final.pt` | 10 | 条件 GAN 对抗损失 | 原始条件生成器 |
| Base+C-FD | 同一个 `common_pretrain_final.pt` | 10 | 条件 GAN 对抗损失 + class-conditional queue FD | 检验 FD-loss 是否改善医学图像生成 |

特征分类器和 50 轮 common pretrain 是两组共同的准备阶段，不是额外实验组。两条分支使用同一数据、架构、batch、优化器状态、随机种子、父检查点和训练轮数，唯一有意差异是 C-FD 项。

类别定义：

- 类别 0：`No Finding == 1` 的正位胸片；
- 类别 1：`Pleural Effusion == 1` 的正位胸片；
- 不使用侧位片；病种的不确定标签 `-1` 从这两个二元类别中排除；
- 按患者划分 train/val/test，患者交集必须为 0；每个 split 内两类数量平衡。

医学特征空间由训练集上训练的 ResNet-18 二分类器定义，取其 256 维投影。FD 组为两类分别维护 4096 个生成特征的队列，并分别与真实训练特征统计量比较：

\[
L_{C\text{-}FD}=\frac{1}{2}\sum_{c\in\{0,1\}}
\frac{FD_c}{\operatorname{stopgrad}(FD_c)+0.01},
\qquad
L_G=L_{adv}+\lambda(t)L_{C\text{-}FD}.
\]

`λ(t)` 在分支前 1000 step 从 0 线性升到 0.10。特征编码器被冻结，但生成图送入编码器时没有 `detach()` 或 `no_grad()`，因此 FD 梯度能回到生成器；当前特征在反向传播完成后才写入队列。两类分开计算，避免健康和患病分布被混成一个高斯分布。

该实现使用CheXpert数据研究FD损失在条件GAN中的作用，不包含PIXEL完整扩散训练与病灶定位流程。FD-Loss与PIXEL作为方法参考，不是运行依赖。

## 3. 目录布局

固定使用以下布局，不要把大数据放到系统盘：

```text
/path/to/workspace/
├── CXR_FD/                         # 项目根目录
│   ├── README_RUNBOOK_CN.md
│   ├── requirements.txt
│   ├── configs/
│   │   ├── paths.env.example
│   │   └── paths.env               # 第 6 步生成
│   ├── src/
│   │   ├── dataset.py              # 数据读取、增强、平衡 batch
│   │   ├── models.py               # 条件 G/D 和医学特征分类器
│   │   ├── fd_loss.py              # 可微 FD、归一化和分类队列
│   │   └── utils.py                # 日志、随机种子、检查点、图像网格
│   ├── scripts/
│   │   ├── bootstrap_env.sh        # 创建完整 Conda 环境
│   │   ├── check_env.py            # GPU/依赖/数据检查
│   │   ├── self_test.py            # G/D 形状与 FD 梯度自测
│   │   ├── prepare_manifest.py      # 患者级划分与两类平衡
│   │   ├── visualize_data.py        # 数据样例和分布图
│   │   ├── train_classifier.py      # 医学特征编码器
│   │   ├── compute_real_stats.py    # 两类真实特征统计量
│   │   ├── train_gan.py             # common/Base/C-FD 共用训练入口
│   │   ├── evaluate.py              # 固定种子公平评价
│   │   └── capture_report.sh        # 汇总关键日志和产物清单
│   ├── data/chexpert_binary/        # manifest.csv 和数据划分摘要
│   ├── artifacts/                   # 数据可视化
│   ├── runs/
│   │   ├── feature_encoder/
│   │   ├── real_stats/
│   │   ├── common/
│   │   ├── base/
│   │   ├── cfd/
│   │   └── evaluation/
│   └── reports/                     # 实验日志、环境和哈希
├── datasets/
│   └── CheXpert-v1.0-small/
│       ├── train.csv
│       ├── valid.csv
│       └── train/patient.../study.../*.jpg
└── reference/                       # 可选，仅供论文/源码查阅
    ├── FD-Loss-main.zip
    ├── PIXEL-master.zip
    └── papers/
```

真正运行只需要本项目和 `CheXpert-v1.0-small`。原始 FD-Loss、PIXEL 压缩包和两篇 PDF 不参与运行，建议仅放进 `reference/` 供引用。CheXpert 受数据协议约束，仓库不包含影像；必须使用按官方条款取得的数据。

## 4. 环境与项目安装

以下命令均在服务器终端中执行。建议复制一段、确认无报错后再执行下一段。

### 4.1 先记录硬件并安装系统工具

```bash
nvidia-smi
df -h
free -h

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git unzip tmux htop tree ripgrep libgl1 libglib2.0-0
```

`nvidia-smi` 必须显示 RTX 4090 和约 24564 MiB 显存。CUDA Driver 版本可以高于 12.4；不要求 `nvcc` 等于 12.4，因为 PyTorch wheel 自带 CUDA runtime。

### 4.2 获取代码

```bash
git clone https://github.com/vaeco/cxr-fd-generation.git /path/to/workspace/CXR_FD
cd /path/to/workspace/CXR_FD
```

`/path/to/workspace`表示本地工作目录。后续路径与`configs/paths.env`保持一致。

### 4.3 一次性创建全部 Python 环境

```bash
cd /path/to/workspace/CXR_FD
eval "$(conda shell.bash hook)"
bash scripts/bootstrap_env.sh 2>&1 | tee bootstrap_env.log
conda activate cxrfd
```

脚本等价于以下完整安装指令；如果脚本中途网络失败，可以从失败位置手动重跑：

```bash
eval "$(conda shell.bash hook)"
conda create -n cxrfd python=3.11.9 -y
conda activate cxrfd
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r /path/to/workspace/CXR_FD/requirements.txt
python -m pip check
```

不需要另外安装 TensorFlow、JAX、diffusers、xformers 或完整 PIXEL requirements。

### 4.4 预下载两个评价模型的权重

这一步放在正式训练前，防止评估时因网络临时中断卡住：

```bash
cd /path/to/workspace/CXR_FD
conda activate cxrfd
python - <<'PY'
from torchvision.models import ResNet18_Weights, resnet18
from torchmetrics.image.fid import FrechetInceptionDistance

resnet18(weights=ResNet18_Weights.DEFAULT)
FrechetInceptionDistance(feature=2048, normalize=True)
print("[MODEL_CACHE_PASS] ResNet18 and Inception weights are ready")
PY
```

终端应打印 `[MODEL_CACHE_PASS]`。若下载失败，先检查服务器是否开放 PyTorch/GitHub 模型下载；不要在权重未缓存时开始长训练。

## 5. 放置 CheXpert-v1.0-small 数据

将数据压缩包上传到 `/path/to/workspace/`。假设文件名为 `CheXpert-v1.0-small.zip`：

```bash
mkdir -p /path/to/workspace/datasets
unzip -q /path/to/workspace/CheXpert-v1.0-small.zip \
  -d /path/to/workspace/datasets
find /path/to/workspace/datasets -maxdepth 3 -name train.csv -print
```

期望唯一输出：

```text
/path/to/workspace/datasets/CheXpert-v1.0-small/train.csv
```

继续检查：

```bash
test -f /path/to/workspace/datasets/CheXpert-v1.0-small/train.csv
test -d /path/to/workspace/datasets/CheXpert-v1.0-small/train
find /path/to/workspace/datasets/CheXpert-v1.0-small/train \
  -type f -name '*.jpg' | head
du -sh /path/to/workspace/datasets/CheXpert-v1.0-small
```

如果 `train.csv` 在另一层目录，不要盲目移动数据；把下一节 `configs/paths.env` 中的 `DATA=` 改成实际 `train.csv` 所在目录。

## 6. 固定路径、日志和长任务保护

```bash
cd /path/to/workspace/CXR_FD
cp -n configs/paths.env.example configs/paths.env
source configs/paths.env
export PYTHONUNBUFFERED=1
mkdir -p "$RUNS" "$META" "$PROJECT/artifacts" "$PROJECT/reports"
mkdir -p "$RUNS"/{env,feature_encoder,real_stats,common,base,cfd,evaluation}
```

后续每次重新登录都先执行：

```bash
eval "$(conda shell.bash hook)"
conda activate cxrfd
cd /path/to/workspace/CXR_FD
source configs/paths.env
export PYTHONUNBUFFERED=1
```

正式训练应放在 `tmux` 里，断开 SSH 不会终止任务：

```bash
tmux new -s cxrfd
```

进入 tmux 后再次执行上面的四行初始化。按 `Ctrl+B`，松开后按 `D` 可退出；重新连接用：

```bash
tmux attach -t cxrfd
```

另开一个 SSH 窗口实时查看 GPU：

```bash
watch -n 2 nvidia-smi
```

同时记录一分钟一次的 GPU 利用率，便于实验报告：

```bash
nohup nvidia-smi \
  --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv -l 60 > "$RUNS/gpu_monitor.csv" 2>&1 &
echo $! > "$RUNS/gpu_monitor.pid"
```

## 7. 阶段 A：环境与 FD 梯度自检

```bash
python scripts/check_env.py --data-root "$DATA" \
  2>&1 | tee "$RUNS/env/check_env_terminal.log"

python scripts/self_test.py \
  2>&1 | tee "$RUNS/env/self_test.log"
```

立即提取日志记录内容：

```bash
grep -E '\[(ENV|GPU|PACKAGES|DATA_CHECK|ENV_PASS)\]' \
  "$RUNS/env/check_env_terminal.log"
grep 'SELF_TEST_PASS' "$RUNS/env/self_test.log"
```

必须同时看到 `[ENV_PASS]` 与 `[SELF_TEST_PASS]`。后者证明生成器/判别器形状正确，并且 FD 对输入的梯度非零且有限。如果这里失败，不要开始训练。

## 8. 阶段 B：生成患者无泄漏的数据清单

```bash
python scripts/prepare_manifest.py \
  --data-root "$DATA" \
  --output-dir "$META" \
  --disease 'Pleural Effusion' \
  --max-train-per-class 12000 \
  --max-val-per-class 2000 \
  --max-test-per-class 2000 \
  --seed 2026 \
  2>&1 | tee "$META/prepare_manifest_terminal.log"
```

提取关键日志：

```bash
grep -E '\[(FILTER|LEAKAGE_CHECK|MANIFEST_READY)\]' \
  "$META/prepare_manifest_terminal.log"

python - <<'PY'
import os
import pandas as pd

frame = pd.read_csv(os.environ["MANIFEST"])
print("[MANIFEST_TABLE]")
print(pd.crosstab(frame["split"], frame["class_name"]))
for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
    a = set(frame.loc[frame.split == left, "patient_id"])
    b = set(frame.loc[frame.split == right, "patient_id"])
    print(f"[PATIENT_OVERLAP] {left}-{right}={len(a & b)}")
print("[PATH_EXISTS]", frame["path"].map(os.path.isfile).all())
PY
```

日志记录必须包含三个 patient overlap 都为 0、`PATH_EXISTS True`，并显示每个 split 的两类图像数量相等。`manifest.csv` 保存的是绝对路径；迁移服务器后要重新生成，不能直接沿用旧服务器的 manifest。

## 9. 阶段 C：数据可视化与人工检查

```bash
python scripts/visualize_data.py \
  --manifest "$MANIFEST" \
  --output-dir "$PROJECT/artifacts" \
  --seed 2026 \
  2>&1 | tee "$PROJECT/artifacts/visualize_terminal.log"

ls -lh "$PROJECT/artifacts"/*.png
grep 'DATA_VIS_READY' "$PROJECT/artifacts/visualize_terminal.log"
```

必须人工查看：

- `artifacts/data_examples.png`：两行分别为健康和胸腔积液样例；确认是正位胸片、方向正常、没有全黑图或纯文字图像；
- `artifacts/data_distribution.png`：确认三个 split 平衡，两个类别强度分布没有离谱断层。

下载图片到本地可以用平台文件管理器，或在本地终端执行（替换端口/IP）：

```bash
scp -P SSH端口 root@服务器IP:/path/to/workspace/CXR_FD/artifacts/data_examples.png .
scp -P SSH端口 root@服务器IP:/path/to/workspace/CXR_FD/artifacts/data_distribution.png .
```

## 10. 阶段 D：训练医学特征编码器

这是 FD 的“医学距离尺”，不是生成对照组。执行：

```bash
python -u scripts/train_classifier.py \
  --manifest "$MANIFEST" \
  --output-dir "$RUNS/feature_encoder" \
  --epochs 8 \
  --batch-size 64 \
  --workers 8 \
  --feature-dim 256 \
  --lr 1e-4 \
  --seed 2026 \
  2>&1 | tee "$RUNS/feature_encoder/terminal.log"
```

完成后展示：

```bash
grep -E '\[(CLASSIFIER_EPOCH|CLASSIFIER_BEST|CLASSIFIER_FINAL)\]' \
  "$RUNS/feature_encoder/terminal.log" | tail -n 15
ls -lh "$FEATURE" "$RUNS/feature_encoder/classifier_curves.png"
```

继续实验的最低质量门槛建议为 test AUC ≥ 0.65；若低于此值，先检查标签、数据路径和样例，不要让无效特征空间指导生成器。AUC 并非越高越能保证生成质量，但明显高于随机水平是必要条件。实验记录使用 `CLASSIFIER_FINAL` 日志和 `classifier_curves.png`。

可选实时查看 TensorBoard：

```bash
tensorboard --logdir "$RUNS" --host 127.0.0.1 --port 6006
```

然后在本地做 SSH 转发 `ssh -L 6006:127.0.0.1:6006 -P SSH端口 root@服务器IP`，浏览器打开 `http://127.0.0.1:6006`。

## 11. 阶段 E：计算两类真实训练特征统计量

```bash
python -u scripts/compute_real_stats.py \
  --manifest "$MANIFEST" \
  --feature-ckpt "$FEATURE" \
  --output-dir "$STATS" \
  --batch-size 128 \
  --workers 8 \
  --max-per-class 12000 \
  --seed 2026 \
  2>&1 | tee "$RUNS/real_stats/terminal.log"
```

提取日志并保存输入哈希：

```bash
grep -E '\[(REF_STATS|REF_STATS_READY)\]' "$RUNS/real_stats/terminal.log"
ls -lh "$STATS"/class_0.npz "$STATS"/class_1.npz
sha256sum "$FEATURE" "$STATS"/class_0.npz "$STATS"/class_1.npz \
  | tee "$PROJECT/reports/fd_inputs.sha256"
```

两个统计文件都应显示 `count=12000`（若原始可用数据不足，则是实际的平衡上限），最小特征值可以有接近 0 的浮点误差，但不应出现 NaN。

## 12. 阶段 F：共同预训练 50 轮

这一步得到两组完全相同的父模型：

```bash
python -u scripts/train_gan.py \
  --mode pretrain \
  --manifest "$MANIFEST" \
  --output-dir "$RUNS/common" \
  --epochs 50 \
  --batch-size 64 \
  --workers 8 \
  --image-size 128 \
  --z-dim 128 \
  --base-channels 64 \
  --lr-g 2e-4 \
  --lr-d 2e-4 \
  --seed 2026 \
  --save-every 5 \
  --sample-every 1 \
  2>&1 | tee "$RUNS/common/terminal.log"
```

重要日志与图片：

```bash
grep -E '\[(GAN_START|EPOCH_SUMMARY|CHECKPOINT|TRAINING_DONE)\]' \
  "$RUNS/common/terminal.log" | tail -n 20
ls -lh "$COMMON" "$RUNS/common/training_curves.png"
ls "$RUNS/common/samples" | head
ls "$RUNS/common/samples" | tail
sha256sum "$COMMON" | tee "$PROJECT/reports/common_before_branches.sha256"
```

人工查看 `samples/epoch_0001.png`、`epoch_0025.png`、`epoch_0050.png`。每张图前两行对应类别 0、后两行对应类别 1；早期模糊是正常的，后期应逐渐出现胸廓、肺野和心影结构。如果 50 轮仍接近纯噪声或全部相同，先停止分支实验并排查，不能用坏父模型得出 FD 结论。

## 13. 最终实验 1：Base

从共同检查点继续 10 轮，损失保持原始条件 GAN：

```bash
python -u scripts/train_gan.py \
  --mode base \
  --manifest "$MANIFEST" \
  --output-dir "$RUNS/base" \
  --resume "$COMMON" \
  --epochs 10 \
  --batch-size 64 \
  --workers 8 \
  --image-size 128 \
  --z-dim 128 \
  --base-channels 64 \
  --lr-g 2e-4 \
  --lr-d 2e-4 \
  --seed 2026 \
  --save-every 5 \
  --sample-every 1 \
  2>&1 | tee "$RUNS/base/terminal.log"
```

完成后检查以下日志：

```bash
grep -E '\[(GAN_START|EPOCH_SUMMARY|TRAINING_DONE)\]' \
  "$RUNS/base/terminal.log" | tail -n 15
ls -lh "$BASE" "$RUNS/base/training_curves.png"
ls -lh "$RUNS/base/samples"/*.png | tail
```

## 14. 最终实验 2：Base + class-conditional queue FD

务必仍从 `$COMMON` 开始，不能从 `$BASE` 开始：

```bash
python -u scripts/train_gan.py \
  --mode fd \
  --manifest "$MANIFEST" \
  --output-dir "$RUNS/cfd" \
  --resume "$COMMON" \
  --feature-ckpt "$FEATURE" \
  --stats-dir "$STATS" \
  --epochs 10 \
  --batch-size 64 \
  --workers 8 \
  --image-size 128 \
  --z-dim 128 \
  --base-channels 64 \
  --lr-g 2e-4 \
  --lr-d 2e-4 \
  --queue-size 4096 \
  --lambda-fd 0.10 \
  --fd-warmup-steps 1000 \
  --fd-offset 0.01 \
  --seed 2026 \
  --save-every 5 \
  --sample-every 1 \
  2>&1 | tee "$RUNS/cfd/terminal.log"
```

立即提取证明“确实加了正确 FD”的日志：

```bash
grep -E '\[(QUEUE_INIT|FD_CONFIG|EPOCH_SUMMARY|TRAINING_DONE)\]' \
  "$RUNS/cfd/terminal.log" | tail -n 25
ls -lh "$CFD" "$RUNS/cfd/training_curves.png"
ls -lh "$RUNS/cfd/samples"/*.png | tail
```

日志记录中应明确出现：两个 `QUEUE_INIT`（class 0 和 1）、`queue_size=4096`、`feature_dim=256`、`class_conditional=True`、`lambda_fd=0.1`。`EPOCH_SUMMARY` 会同时给出 normalized FD、两个类别的 raw FD 和生成器梯度范数。raw FD 不要求逐轮严格单调，但不能为 NaN/Inf，且整体趋势可用于描述。

## 15. 分支公平性审计

先确认共同检查点没有被修改：

```bash
sha256sum -c "$PROJECT/reports/common_before_branches.sha256"
```

再执行自动审计：

```bash
python - <<'PY' | tee /path/to/workspace/CXR_FD/reports/branch_fairness.txt
import os
import torch

base = torch.load(os.environ["BASE"], map_location="cpu", weights_only=False)
cfd = torch.load(os.environ["CFD"], map_location="cpu", weights_only=False)
checks = {
    "same_parent": base["parent_checkpoint"] == cfd["parent_checkpoint"],
    "same_architecture": base["config"] == cfd["config"],
    "same_final_epoch": base["epoch"] == cfd["epoch"],
    "same_seed": base["args"]["seed"] == cfd["args"]["seed"],
    "same_batch": base["args"]["batch_size"] == cfd["args"]["batch_size"],
    "base_mode": base["mode"] == "base",
    "cfd_mode": cfd["mode"] == "fd",
}
for key, value in checks.items():
    print(f"[FAIRNESS] {key}={value}")
assert all(checks.values()), checks
print("[FAIRNESS_PASS] Base and Base+C-FD are directly comparable")
PY
```

所有项必须为 `True`。

## 16. 固定种子最终评价和可视化

评价使用未参与训练的 test 患者，分别报告两类；FID/KID 使用独立 Inception 特征，CXR-FD 使用医学编码器特征，condition accuracy 检查生成图是否符合条件。不要只用 CXR-FD 宣称成功，因为它与训练损失共享特征空间。

先用每类 256 张做一次流程检查。它不是新实验组，也不进入最终表格：

```bash
python -u scripts/evaluate.py \
  --manifest "$MANIFEST" \
  --base-ckpt "$BASE" \
  --cfd-ckpt "$CFD" \
  --feature-ckpt "$FEATURE" \
  --output-dir "$RUNS/evaluation_smoke" \
  --num-per-class 256 \
  --batch-size 32 \
  --workers 8 \
  --seed 2026 \
  2>&1 | tee "$RUNS/evaluation_smoke/terminal.log"
```

出现 `[EVALUATION_DONE]` 后再跑最终 2000 张/类：

```bash
python -u scripts/evaluate.py \
  --manifest "$MANIFEST" \
  --base-ckpt "$BASE" \
  --cfd-ckpt "$CFD" \
  --feature-ckpt "$FEATURE" \
  --output-dir "$RUNS/evaluation" \
  --num-per-class 2000 \
  --batch-size 64 \
  --workers 8 \
  --kid-subsets 50 \
  --kid-subset-size 50 \
  --seed 2026 \
  2>&1 | tee "$RUNS/evaluation/terminal.log"
```

提取最终表格和日志：

```bash
grep -E '\[(FAIRNESS_CHECK|EVAL_CLASS|EVAL_FINAL|EVALUATION_DONE)\]' \
  "$RUNS/evaluation/terminal.log"

python - <<'PY'
import os
import pandas as pd

root = os.path.join(os.environ["RUNS"], "evaluation")
print("[METRICS_BY_CLASS]")
print(pd.read_csv(os.path.join(root, "metrics_by_class.csv")).to_string(index=False))
print("\n[MACRO_AVERAGE]")
print(pd.read_csv(os.path.join(root, "metrics_macro_average.csv")).to_string(index=False))
PY

ls -lh "$RUNS/evaluation"/*.png "$RUNS/evaluation"/*.csv
```

最终实验报告应至少插入：

- `fixed_noise_comparison.png`：相同噪声下四行 Base/C-FD × 健康/患病图像；
- `metric_comparison.png`：FID、KID、CXR-FD、条件准确率柱状图；
- `metrics_by_class.csv`：逐类结果；
- `metrics_macro_average.csv`：两类宏平均；
- `delta_summary.json`：C-FD 相对 Base 的变化百分比。

判读规则：FID/KID/CXR-FD 越低越好，condition accuracy 越高越好。最有说服力的结论是 C-FD 的独立 FID/KID 与医学 CXR-FD 同时下降，且条件准确率不下降；若只有 CXR-FD 下降而 FID/KID 变差，应写成“优化了训练特征空间中的匹配，但独立生成质量证据不足”，不能写全面优于 Base。一次训练种子属于单训练种子的探索性实验；若以后用于论文，应额外重复多个训练种子并报告均值和置信区间。

## 17. 一键整理实验报告证据

训练与评价全部结束后：

```bash
cd /path/to/workspace/CXR_FD
source configs/paths.env
bash scripts/capture_report.sh
sha256sum "$BASE" "$CFD" \
  | tee "$PROJECT/reports/final_checkpoints.sha256"

tail -n 50 "$PROJECT/reports/key_log_markers.txt"
cat "$PROJECT/reports/branch_fairness.txt"
cat "$RUNS/evaluation/metrics_macro_average.csv"
```

停止 GPU 监控进程：

```bash
test -f "$RUNS/gpu_monitor.pid" && kill "$(cat "$RUNS/gpu_monitor.pid")"
```

建议的实验记录顺序：

1. `nvidia-smi`、`ENV_PASS`、`SELF_TEST_PASS`；
2. manifest 两类计数、三个患者交集为 0；
3. `data_examples.png`；
4. 分类器 `CLASSIFIER_FINAL` 与 `classifier_curves.png`；
5. 两类 `REF_STATS`；
6. common 第 1/25/50 轮样图；
7. Base 的 `GAN_START`、末轮日志、曲线、样图；
8. C-FD 的两个 `QUEUE_INIT`、`FD_CONFIG`、末轮日志、曲线、样图；
9. `FAIRNESS_PASS`；
10. 四行定性对比图、指标柱状图和最终 CSV 表。

## 18. 常见问题及安全处理

### CUDA out of memory

先把训练命令的 `--batch-size 64` 改为 48。Base 和 C-FD 两个分支必须都用 48；如果 Base 已经用 64 跑完，应删除“比较结论”并用 48 重新跑 Base，不能拿不同 batch 的结果硬比较。评价 batch 可单独降到 32，不改变指标定义。不要先改队列方法，也不要切换到 batch-only FD。

### DataLoader worker 报错或共享内存不足

把所有命令的 `--workers 8` 改为 4，极端情况下改为 0；这只影响速度，不影响实验定义。Base/C-FD 最好保持相同 worker 配置。

### 分类器预训练权重下载失败

重新执行 4.4；网络完全不可用时可给分类器命令加 `--no-pretrained`，但通常需要增加训练轮数，且必须重新检查 AUC。不要混用不同分类器生成统计量和 FD 训练。

### FD 出现 NaN/Inf

立即停止，依次检查 `class_0.npz/class_1.npz` 是否存在、分类器 checkpoint 是否对应、`REF_STATS` 是否包含 NaN、队列是否两个类别都成功初始化。确认输入无误后，可把 C-FD 的 `--lambda-fd` 降到 0.05 并从共同检查点重新开始；同时要保留失败日志说明调整。不要让脚本跳过 NaN 继续训练。

### 训练中断

common 可用 `runs/common/checkpoints/latest.pt` 作为 `--resume`，但命令的 `--epochs` 表示“再增加多少轮”；先用 checkpoint 中的 epoch 计算剩余轮数。Base/C-FD 中断也可恢复各自 `latest.pt`。恢复 C-FD 时脚本会同时恢复队列。为保持分支对照一致，若只跑了很短时间便中断，优先从共同检查点重新跑该分支。

### 生成图看起来不错但指标不一致

定性图和量化指标都要报告。FID 在灰度胸片上并不完美，所以同时报告 KID、医学 CXR-FD 和 condition accuracy；不能挑一个最有利指标。不要用 test 数据参与模型选择或重新训练。

## 19. 最终验收清单

只有以下全部满足，才算“跑通”：

- `[ENV_PASS]`、`[SELF_TEST_PASS]`；
- train/val/test 患者交集均为 0，每个 split 两类平衡；
- 分类器 checkpoint 存在且 test AUC 达到基本有效水平；
- `class_0.npz` 和 `class_1.npz` 均存在且有限；
- common checkpoint 完成 50 轮；
- Base 与 C-FD 都从同一个 common checkpoint继续到相同 final epoch；
- C-FD 日志有两个分类队列、FD 配置、有限 FD 和非零生成器梯度；
- `[FAIRNESS_PASS]`；
- 最终评价使用相同 seed、相同样本数和 test split；
- `metrics_by_class.csv`、`metrics_macro_average.csv`、两张最终对比图、关键日志均已保存。

完成以上检查后，可在相同起点、预算和评价协议下比较Base与Base+C-FD，并同时报告分布指标、条件一致性和样图。
