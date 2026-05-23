# LightCLIP 项目交接文档

本文档用于把 LightCLIP 项目交接给后续实验人员，使其能够在本地或服务器上继续完成训练、评估、导出和结果整理。

仓库地址：

```text
https://github.com/Emmet-Ding/LightCLIP
```

## 1. 项目目标

本项目目标是实现一个轻量化 CLIP 蒸馏流程：

1. 使用 OpenCLIP ViT-B/32 作为 Teacher。
2. 将 Teacher 的图像特征、文本特征和跨模态相似度结构蒸馏到轻量 Student CLIP。
3. Student 当前采用 ResNet-18 图像编码器 + Tiny Transformer 文本编码器。
4. 完成训练后，需要评估 Student zero-shot accuracy、Teacher baseline、推理速度、FPS，并导出 ONNX。
5. 后续主要实验应在服务器上继续完成和验证。

## 2. 当前已经完成的工作

### 2.1 工程代码

已经实现并上传到 GitHub：

```text
configs/                 训练配置
datasets/                数据集读取和 CLIP 图像预处理
losses/                  蒸馏损失和 CLIP 对比损失
models/                  Student CLIP 模型
tools/                   数据构建、Teacher cache、训练、评估、benchmark、ONNX 导出脚本
requirements.txt         固定依赖版本
README.md                中文运行说明
```

核心脚本：

```text
tools/build_metadata_from_imagefolder.py
tools/precompute_teacher_features.py
tools/train_distill.py
tools/eval_zeroshot.py
tools/eval_teacher_zeroshot.py
tools/benchmark_latency.py
tools/export_onnx.py
tools/create_smoke_data.py
```

`tools/train_distill.py` 已支持：

```text
--config      指定配置文件
--resume      从 checkpoint 续训
--device      auto / cuda / cpu
--epochs      临时覆盖训练总 epoch，用于测试或阶段性续训
```

### 2.2 GitHub LFS 文件

以下文件已经通过 Git LFS 上传：

```text
weights/openclip_vit_b_32_openai.pt
data/lightclip_dataset.tar.gz
```

含义：

```text
weights/openclip_vit_b_32_openai.pt    OpenCLIP ViT-B/32 Teacher 权重
data/lightclip_dataset.tar.gz          数据集归档，包含 images/ 和 metadata.csv
```

当前仓库还直接跟踪：

```text
data/metadata.csv
```

该 CSV 当前包含 60000 行，列为：

```text
image_path,text,label
```

### 2.3 依赖版本

`requirements.txt` 已固定到本地验证过的 Python 3.10 + CUDA 12.1 环境：

```text
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
open_clip_torch==3.3.0
timm==1.0.27
transformers==5.8.1
pillow==12.2.0
pandas==2.3.3
numpy==2.2.6
tqdm==4.67.3
scikit-learn==1.7.2
matplotlib==3.10.9
PyYAML==6.0.3
psutil==7.2.2
onnx==1.21.0
onnxscript==0.7.0
onnxruntime==1.23.2
```

如果服务器不是 CUDA 12.1，需要替换 PyTorch 三件套和 wheel 源，其余依赖可优先保持一致。

## 3. 当前实验做到哪里

### 3.1 Teacher cache

本地已经生成过 Teacher cache：

```text
data/teacher_cache/teacher_features.pt
```

本地文件大小约 249 MB，包含：

```text
image_features: (60000, 512)
text_features:  (60000, 512)
model_name:     ViT-B-32
```

该文件没有上传到 GitHub，因为它是可再生成缓存。服务器上应重新生成。

### 3.2 训练日志

当前最新训练日志：

```text
outputs/logs/train_log.csv
```

日志显示训练已经到 `epoch 30`：

```text
best val epoch: 25
best val loss:  3.159343873977661
epoch 30 val loss: 3.160017431259156
```

最后 10 个 epoch 的趋势：

```text
epoch 21 val_loss 3.173937
epoch 22 val_loss 3.180169
epoch 23 val_loss 3.178972
epoch 24 val_loss 3.188604
epoch 25 val_loss 3.159344
epoch 26 val_loss 3.161639
epoch 27 val_loss 3.166293
epoch 28 val_loss 3.166047
epoch 29 val_loss 3.165704
epoch 30 val_loss 3.160017
```

结论：

1. 训练 loss 仍在缓慢下降。
2. 验证 loss 在 epoch 25 后基本平台化。
3. 继续增加 epoch 的优先级低于完成 zero-shot、Teacher baseline、速度和 ONNX 评估。

### 3.3 当前 checkpoint 风险

当前本地 checkpoint 与最新训练日志不匹配：

```text
outputs/checkpoints/student_best.pt -> epoch 6, metric 2.914185934702555
outputs/checkpoints/student_last.pt -> epoch 9, metric 3.2309686603546144
outputs/logs/train_log.csv          -> epoch 30
```

这意味着：

1. 不能用当前本地 `student_best.pt` 代表最新 30 epoch 训练结果。
2. 必须先找到与最新日志对应的 checkpoint。
3. 理论上最新训练对应的 best checkpoint 应该是 epoch 25，last checkpoint 应该是 epoch 30。
4. 如果训练是在服务器或其他目录完成的，应从该环境取回对应的 `student_best.pt` 和 `student_last.pt`。
5. 如果对应 checkpoint 丢失，仅凭 `train_log.csv` 无法恢复模型权重，只能重新训练或从已有较旧 checkpoint 继续训练。

## 4. 下一步优先事项

按优先级执行：

1. 找到与最新 `train_log.csv` 对应的 checkpoint。
2. 用最新 best checkpoint 跑 Student zero-shot。
3. 跑 Teacher baseline。
4. 计算 Student / Teacher accuracy retention。
5. 跑 latency / FPS benchmark。
6. 导出 ONNX。
7. 汇总完整实验表格。
8. 如结果合理，再考虑消融实验或替换更合适的领域 Teacher。

不建议下一步直接继续训练更多 epoch。当前日志显示验证 loss 已经平台化，继续训练的信息量低于评估。

## 5. 服务器需要准备或上传的文件

### 5.1 推荐方式：直接从 GitHub clone

服务器只需要能访问 GitHub 和 Git LFS：

```bash
git clone https://github.com/Emmet-Ding/LightCLIP.git
cd LightCLIP
git lfs install
git lfs pull
```

`git lfs pull` 后应得到：

```text
weights/openclip_vit_b_32_openai.pt
data/lightclip_dataset.tar.gz
```

然后解压数据：

```bash
tar -xzf data/lightclip_dataset.tar.gz -C data
```

解压后应存在：

```text
data/images/
data/metadata.csv
```

### 5.2 如果不能从 GitHub 拉取，需要手动上传

至少上传以下文件或目录：

```text
configs/
datasets/
losses/
models/
tools/
requirements.txt
README.md
docs/LightCLIP_项目交接文档.md
weights/openclip_vit_b_32_openai.pt
data/lightclip_dataset.tar.gz
data/metadata.csv
```

如果已经在某台机器上完成了最新 30 epoch 训练，还必须上传：

```text
outputs/logs/train_log.csv
outputs/checkpoints/student_best.pt
outputs/checkpoints/student_last.pt
```

其中，当前日志对应的 checkpoint 应满足：

```text
student_best.pt  应该对应 epoch 25
student_last.pt  应该对应 epoch 30
```

如果要避免服务器重新生成 Teacher cache，也可以上传：

```text
data/teacher_cache/teacher_features.pt
```

但该文件是可再生成缓存，不是必须上传项。

## 6. 从零运行完整流程

以下命令假设服务器为 Linux，并且 CUDA 环境与 `requirements.txt` 的 CUDA 12.1 PyTorch wheel 兼容。

### 6.1 克隆和准备 LFS 文件

```bash
git clone https://github.com/Emmet-Ding/LightCLIP.git
cd LightCLIP
git lfs install
git lfs pull
```

检查 LFS 文件：

```bash
ls -lh weights/openclip_vit_b_32_openai.pt
ls -lh data/lightclip_dataset.tar.gz
```

解压数据：

```bash
tar -xzf data/lightclip_dataset.tar.gz -C data
```

检查数据：

```bash
find data/images -type f | wc -l
python -c "import pandas as pd; df=pd.read_csv('data/metadata.csv'); print(len(df)); print(df.head())"
```

期望：

```text
metadata.csv 行数: 60000
```

### 6.2 创建 Python 环境

```bash
conda create -n lightclip_distill python=3.10 -y
conda activate lightclip_distill
pip install -r requirements.txt
```

检查 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

如果 CUDA 不匹配，先替换 PyTorch 安装命令。例如服务器需要其他 CUDA 版本时，按 PyTorch 官方命令安装对应版本，再安装其余依赖。

### 6.3 生成 Teacher cache

```bash
python tools/precompute_teacher_features.py \
  --metadata_csv data/metadata.csv \
  --out data/teacher_cache/teacher_features.pt \
  --model_name ViT-B-32 \
  --weights_path weights/openclip_vit_b_32_openai.pt \
  --batch_size 64
```

检查 cache：

```bash
python -c "import torch; x=torch.load('data/teacher_cache/teacher_features.pt', map_location='cpu'); print(x['image_features'].shape); print(x['text_features'].shape); print(x['model_name'])"
```

期望：

```text
torch.Size([60000, 512])
torch.Size([60000, 512])
ViT-B-32
```

如果 GPU 预计算不稳定，可用 CPU 生成：

```bash
CUDA_VISIBLE_DEVICES="" python tools/precompute_teacher_features.py \
  --metadata_csv data/metadata.csv \
  --out data/teacher_cache/teacher_features.pt \
  --model_name ViT-B-32 \
  --weights_path weights/openclip_vit_b_32_openai.pt \
  --batch_size 128
```

### 6.4 训练 Student

当前默认配置：

```text
configs/distill_resnet18.yaml
```

从头训练：

```bash
python tools/train_distill.py --config configs/distill_resnet18.yaml --device cuda
```

只试跑 1 个 epoch：

```bash
python tools/train_distill.py --config configs/distill_resnet18.yaml --device cuda --epochs 1
```

从 checkpoint 续训：

```bash
python tools/train_distill.py \
  --config configs/distill_resnet18.yaml \
  --device cuda \
  --resume outputs/checkpoints/student_last.pt
```

续训到指定 epoch：

```bash
python tools/train_distill.py \
  --config configs/distill_resnet18.yaml \
  --device cuda \
  --resume outputs/checkpoints/student_last.pt \
  --epochs 30
```

训练输出：

```text
outputs/checkpoints/student_best.pt
outputs/checkpoints/student_last.pt
outputs/logs/train_log.csv
```

检查 checkpoint：

```bash
python -c "import torch; \
for p in ['outputs/checkpoints/student_best.pt','outputs/checkpoints/student_last.pt']: \
    ckpt=torch.load(p,map_location='cpu'); print(p, ckpt.get('epoch'), ckpt.get('metric'))"
```

训练完成后，`student_best.pt` 的 epoch 应该和 `train_log.csv` 中最低 val loss 的 epoch 一致。

## 7. 如果沿用当前最新日志，应该怎么处理

当前日志显示 best epoch 是 25。因此应优先寻找：

```text
outputs/checkpoints/student_best.pt
```

且该 checkpoint 应显示：

```text
epoch = 25
metric ~= 3.159343873977661
```

如果当前机器没有这个文件，但训练机器上有，应将该 checkpoint 拷贝回来：

```bash
scp user@server:/path/to/LightCLIP/outputs/checkpoints/student_best.pt outputs/checkpoints/student_best.pt
scp user@server:/path/to/LightCLIP/outputs/checkpoints/student_last.pt outputs/checkpoints/student_last.pt
scp user@server:/path/to/LightCLIP/outputs/logs/train_log.csv outputs/logs/train_log.csv
```

如果 checkpoint 丢失，应从最近可用 checkpoint 重新训练。当前本地 `student_last.pt` 仅到 epoch 9，不能代表最新 30 epoch 日志。

## 8. 评估流程

以下评估必须在 checkpoint 与日志匹配后执行。

### 8.1 Student zero-shot

```bash
python tools/eval_zeroshot.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --metadata_csv data/metadata.csv \
  --prompt "a photo of a {}" \
  --batch_size 64 \
  --num_workers 0
```

需要记录输出中的 accuracy。

### 8.2 Teacher baseline

```bash
python tools/eval_teacher_zeroshot.py \
  --metadata_csv data/metadata.csv \
  --model_name ViT-B-32 \
  --weights_path weights/openclip_vit_b_32_openai.pt \
  --prompt "a photo of a {}" \
  --batch_size 64 \
  --num_workers 0
```

需要记录 Teacher accuracy。

### 8.3 精度保持率

计算：

```text
accuracy_retention = student_accuracy / teacher_accuracy
```

如果 Teacher accuracy 很低，需要谨慎解释 retention。原因可能是通用 CLIP Teacher 与数据集或 prompt 不匹配。

## 9. 速度测试和 ONNX 导出

### 9.1 Latency / FPS

```bash
python tools/benchmark_latency.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --batch_size 1 \
  --warmup 20 \
  --iters 200
```

如需 batch 推理：

```bash
python tools/benchmark_latency.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --batch_size 32 \
  --warmup 20 \
  --iters 200
```

### 9.2 ONNX 导出

```bash
python tools/export_onnx.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --out_dir outputs/onnx \
  --opset 17
```

导出后应得到：

```text
outputs/onnx/student_image_encoder.onnx
outputs/onnx/student_text_encoder.onnx
```

## 10. 结果整理模板

建议后续实验人员整理如下表格：

```text
数据集:
类别数:
样本数:
prompt:
Teacher:
Student:
训练 epoch:
best epoch:
best val loss:
Teacher zero-shot accuracy:
Student zero-shot accuracy:
Accuracy retention:
Student 参数量:
checkpoint 大小:
batch_size=1 latency:
batch_size=1 FPS:
batch_size=32 latency:
batch_size=32 FPS:
ONNX image encoder 是否导出成功:
ONNX text encoder 是否导出成功:
Python 版本:
torch 版本:
CUDA 版本:
GPU 型号:
```

## 11. 当前主要风险和判断

### 11.1 checkpoint 与日志不匹配

这是当前最重要的交接风险。最新日志到 epoch 30，但本地 checkpoint 只到 epoch 6/9。后续必须先解决该问题，否则评估结果不对应最新训练日志。

### 11.2 当前 loss 不是最终论文结果

`train_log.csv` 里的 loss 只能说明蒸馏训练过程，不能替代 zero-shot accuracy、latency、FPS 或 ONNX 验证。

### 11.3 通用 Teacher 的领域适配风险

OpenCLIP ViT-B/32 是通用自然图像 Teacher。如果后续用于医学图像或电磁相关图像，需要重新审视 prompt 和 Teacher 适配性。必要时应替换为领域 CLIP Teacher 或做领域 prompt 设计。

### 11.4 Windows 本地训练稳定性

本地曾出现 CUDA illegal memory access、illegal instruction 和进程卡住。当前配置已做保守调整：

```yaml
train:
  batch_size: 48
  num_workers: 0
  amp: false
  pin_memory: false
```

服务器 Linux CUDA 环境通常更适合完成正式实验。

## 12. 建议的后续实验顺序

1. 在服务器上 clone 仓库并 `git lfs pull`。
2. 解压 `data/lightclip_dataset.tar.gz`。
3. 生成 `data/teacher_cache/teacher_features.pt`。
4. 若已有最新 checkpoint，先检查其 epoch 和 metric 是否对应日志。
5. 若没有最新 checkpoint，从头训练或从可用 checkpoint 续训到 30 epoch。
6. 运行 Student zero-shot。
7. 运行 Teacher baseline。
8. 运行 latency/FPS。
9. 导出 ONNX。
10. 汇总结果表格。
11. 再决定是否继续消融实验。

建议消融实验：

```text
1. Student only: L_clip
2. Feature distillation: L_img + L_txt + L_clip
3. Similarity distillation: L_sim + L_clip
4. Full method: L_img + L_txt + 2 * L_sim + L_clip
```

## 13. 最短交接命令清单

如果接手者只需要最快继续：

```bash
git clone https://github.com/Emmet-Ding/LightCLIP.git
cd LightCLIP
git lfs install
git lfs pull
tar -xzf data/lightclip_dataset.tar.gz -C data

conda create -n lightclip_distill python=3.10 -y
conda activate lightclip_distill
pip install -r requirements.txt

python tools/precompute_teacher_features.py \
  --metadata_csv data/metadata.csv \
  --out data/teacher_cache/teacher_features.pt \
  --model_name ViT-B-32 \
  --weights_path weights/openclip_vit_b_32_openai.pt \
  --batch_size 64

python tools/train_distill.py --config configs/distill_resnet18.yaml --device cuda

python tools/eval_teacher_zeroshot.py \
  --metadata_csv data/metadata.csv \
  --model_name ViT-B-32 \
  --weights_path weights/openclip_vit_b_32_openai.pt \
  --prompt "a photo of a {}" \
  --batch_size 64 \
  --num_workers 0

python tools/eval_zeroshot.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --metadata_csv data/metadata.csv \
  --prompt "a photo of a {}" \
  --batch_size 64 \
  --num_workers 0

python tools/benchmark_latency.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --batch_size 1 \
  --warmup 20 \
  --iters 200

python tools/export_onnx.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --out_dir outputs/onnx \
  --opset 17
```

