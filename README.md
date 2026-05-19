# LightCLIP

LightCLIP 是一个轻量化 CLIP 蒸馏工程。项目使用冻结的 OpenCLIP ViT-B/32 作为 Teacher，将图像特征、文本特征和跨模态相似度结构蒸馏到较小的 Student CLIP 中。部署阶段只需要保留 Student 模型。

当前仓库已经包含完整实验流程：

- 从 ImageFolder 数据目录生成 `metadata.csv`
- 使用 ViT-B/32 Teacher 预计算图像/文本特征
- 训练 ResNet-18 + Tiny Transformer Student CLIP
- 支持中断后从 checkpoint 续训
- 评估 Teacher 和 Student zero-shot accuracy
- 测试 Student latency/FPS
- 导出 Student 图像编码器和文本编码器 ONNX
- 提供 smoke test 验证工程链路

## 目录结构

```text
configs/                 训练配置
datasets/                图文数据集与 CLIP 图像预处理
losses/                  特征蒸馏、相似度蒸馏、CLIP 对比损失
models/                  Student CLIP 模型
tools/                   数据、训练、评估、benchmark、ONNX 工具脚本
weights/                 OpenCLIP ViT-B/32 Teacher 权重，使用 Git LFS 管理
data/                    本地数据目录，不提交真实数据
outputs/                 checkpoint、日志和导出文件，不提交生成结果
requirements.txt         已验证依赖版本
```

## 0. 克隆仓库

```bash
git clone https://github.com/Emmet-Ding/LightCLIP.git
cd LightCLIP
```

如果机器上没有 Git LFS，先安装并启用：

```bash
git lfs install
```

拉取 Teacher 权重：

```bash
git lfs pull
```

确认权重文件存在：

```bash
ls weights/openclip_vit_b_32_openai.pt
```

Windows PowerShell 可以使用：

```powershell
Get-Item weights\openclip_vit_b_32_openai.pt
```

如果没有执行 `git lfs pull`，`weights/openclip_vit_b_32_openai.pt` 可能只是 LFS 指针文件，后续 Teacher 加载会失败。

## 1. 创建环境

推荐 Python 3.10。当前 `requirements.txt` 固定的是本地已验证的 CUDA 12.1 版本：

```text
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
```

创建并激活环境：

```bash
conda create -n lightclip_distill python=3.10 -y
conda activate lightclip_distill
```

安装依赖：

```bash
pip install -r requirements.txt
```

检查 PyTorch 和 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

如果服务器不是 CUDA 12.1 环境，需要先按服务器 CUDA 版本替换 PyTorch 三件套。例如 CPU-only 机器或其他 CUDA 版本，不应直接使用当前 `+cu121` 版本。其余依赖版本可以继续使用 `requirements.txt` 中的版本。

## 2. 准备数据

默认数据格式是 ImageFolder：

```text
data/images/
  class_1/
    000001.jpg
    000002.jpg
  class_2/
    000001.jpg
    000002.jpg
```

本地已使用 CIFAR-10 风格目录测试过：

```text
data/images/
  airplane/
  automobile/
  bird/
  cat/
  deer/
  dog/
  frog/
  horse/
  ship/
  truck/
```

仓库通过 Git LFS 提供了一个数据集归档：

```text
data/lightclip_dataset.tar.gz
```

归档内容包含：

```text
images/
metadata.csv
```

如果使用仓库随附数据集，clone 后先执行：

```bash
git lfs pull
tar -xzf data/lightclip_dataset.tar.gz -C data
```

Windows PowerShell 可以使用：

```powershell
git lfs pull
tar -xzf data\lightclip_dataset.tar.gz -C data
```

解压后应得到：

```text
data/images/
data/metadata.csv
```

当前 GitHub 只跟踪 `data/lightclip_dataset.tar.gz` 和 `data/metadata.csv`。解压后的 `data/images/`、Teacher cache 和训练输出仍被 `.gitignore` 排除，避免把大量派生文件直接写进普通 Git 历史。

## 3. 生成 metadata.csv

如果使用仓库随附的 `data/lightclip_dataset.tar.gz`，解压后已经包含 `data/metadata.csv`，可以直接跳到第 4 步。

从 `data/images/` 生成训练用 CSV：

```bash
python tools/build_metadata_from_imagefolder.py \
  --image_root data/images \
  --out_csv data/metadata.csv \
  --prompt "a photo of a {}"
```

Windows PowerShell 可以写成单行：

```powershell
python tools\build_metadata_from_imagefolder.py --image_root data\images --out_csv data\metadata.csv --prompt "a photo of a {}"
```

生成的 `data/metadata.csv` 包含三列：

```text
image_path,text,label
```

检查前几行：

```bash
python -c "import pandas as pd; df=pd.read_csv('data/metadata.csv'); print(df.head()); print(len(df))"
```

医学图像任务可以替换 prompt，例如：

```bash
python tools/build_metadata_from_imagefolder.py \
  --image_root data/images \
  --out_csv data/metadata.csv \
  --prompt "an ultrasound image of {}"
```

prompt 必须和后续 Teacher/Student zero-shot 评估保持一致，否则评估不可直接比较。

## 4. 预计算 Teacher 特征

使用本地 ViT-B/32 权重预计算 Teacher 图像和文本特征：

```bash
python tools/precompute_teacher_features.py \
  --metadata_csv data/metadata.csv \
  --out data/teacher_cache/teacher_features.pt \
  --model_name ViT-B-32 \
  --weights_path weights/openclip_vit_b_32_openai.pt \
  --batch_size 64
```

Windows PowerShell 单行：

```powershell
python tools\precompute_teacher_features.py --metadata_csv data\metadata.csv --out data\teacher_cache\teacher_features.pt --model_name ViT-B-32 --weights_path weights\openclip_vit_b_32_openai.pt --batch_size 64
```

如果本地 GPU 在 Teacher 预计算时不稳定，可以强制 CPU 执行：

```bash
CUDA_VISIBLE_DEVICES="" python tools/precompute_teacher_features.py \
  --metadata_csv data/metadata.csv \
  --out data/teacher_cache/teacher_features.pt \
  --model_name ViT-B-32 \
  --weights_path weights/openclip_vit_b_32_openai.pt \
  --batch_size 128
```

Windows PowerShell：

```powershell
$env:CUDA_VISIBLE_DEVICES=''
python tools\precompute_teacher_features.py --metadata_csv data\metadata.csv --out data\teacher_cache\teacher_features.pt --model_name ViT-B-32 --weights_path weights\openclip_vit_b_32_openai.pt --batch_size 128
Remove-Item Env:\CUDA_VISIBLE_DEVICES
```

检查 Teacher cache：

```bash
python -c "import torch; x=torch.load('data/teacher_cache/teacher_features.pt', map_location='cpu'); print(x['image_features'].shape); print(x['text_features'].shape); print(x['model_name'])"
```

期望输出的第一维等于 `metadata.csv` 行数。

## 5. 检查训练配置

默认配置文件：

```text
configs/distill_resnet18.yaml
```

关键字段：

```yaml
data:
  metadata_csv: data/metadata.csv
  teacher_cache: data/teacher_cache/teacher_features.pt

teacher:
  model_name: ViT-B-32
  weights_path: weights/openclip_vit_b_32_openai.pt

train:
  epochs: 30
  batch_size: 48
  num_workers: 0
  amp: false
  pin_memory: false
  cudnn_enabled: true
  disable_tqdm: true
```

当前配置是按本地 RTX 2060 稳定性调整后的版本。服务器显存更大时，可以提高 `batch_size`、启用 `amp: true`、增大 `num_workers`。修改后应先跑 1 个 epoch 验证稳定性。

## 6. 训练 Student

从头训练：

```bash
python tools/train_distill.py --config configs/distill_resnet18.yaml
```

指定设备：

```bash
python tools/train_distill.py --config configs/distill_resnet18.yaml --device cuda
```

CPU 训练：

```bash
python tools/train_distill.py --config configs/distill_resnet18.yaml --device cpu
```

只跑到指定 epoch，用于测试配置是否稳定：

```bash
python tools/train_distill.py --config configs/distill_resnet18.yaml --epochs 1
```

训练输出：

```text
outputs/checkpoints/student_best.pt
outputs/checkpoints/student_last.pt
outputs/logs/train_log.csv
```

查看训练日志：

```bash
cat outputs/logs/train_log.csv
```

Windows PowerShell：

```powershell
Get-Content outputs\logs\train_log.csv
```

## 7. 中断后续训

如果训练中断，从最近 checkpoint 继续：

```bash
python tools/train_distill.py \
  --config configs/distill_resnet18.yaml \
  --resume outputs/checkpoints/student_last.pt
```

Windows PowerShell 单行：

```powershell
python tools\train_distill.py --config configs\distill_resnet18.yaml --resume outputs\checkpoints\student_last.pt
```

续训到指定 epoch：

```bash
python tools/train_distill.py \
  --config configs/distill_resnet18.yaml \
  --resume outputs/checkpoints/student_last.pt \
  --epochs 10
```

如果训练过程中出现 CUDA illegal memory access、illegal instruction 或进程卡住，优先尝试：

```yaml
train:
  batch_size: 32
  num_workers: 0
  amp: false
  pin_memory: false
```

如果仍不稳定，建议在服务器或 Linux CUDA 环境中继续实验；Windows WDDM + 桌面 GPU 长时间训练存在额外不稳定变量。

## 8. 评估 Student zero-shot

使用最佳 checkpoint：

```bash
python tools/eval_zeroshot.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --metadata_csv data/metadata.csv \
  --prompt "a photo of a {}" \
  --batch_size 64 \
  --num_workers 0
```

Windows PowerShell 单行：

```powershell
python tools\eval_zeroshot.py --ckpt outputs\checkpoints\student_best.pt --metadata_csv data\metadata.csv --prompt "a photo of a {}" --batch_size 64 --num_workers 0
```

这里的 prompt 应和生成 `metadata.csv` 时的 prompt 逻辑一致。

## 9. 评估 Teacher baseline

Teacher baseline 用于计算 Student 的精度保持率：

```bash
python tools/eval_teacher_zeroshot.py \
  --metadata_csv data/metadata.csv \
  --model_name ViT-B-32 \
  --weights_path weights/openclip_vit_b_32_openai.pt \
  --prompt "a photo of a {}" \
  --batch_size 64 \
  --num_workers 0
```

Windows PowerShell 单行：

```powershell
python tools\eval_teacher_zeroshot.py --metadata_csv data\metadata.csv --model_name ViT-B-32 --weights_path weights\openclip_vit_b_32_openai.pt --prompt "a photo of a {}" --batch_size 64 --num_workers 0
```

建议同时记录：

```text
Student accuracy
Teacher accuracy
Student / Teacher accuracy retention
```

## 10. 测试推理速度

测试 Student latency 和 FPS：

```bash
python tools/benchmark_latency.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --batch_size 1 \
  --warmup 20 \
  --iters 200
```

Windows PowerShell 单行：

```powershell
python tools\benchmark_latency.py --ckpt outputs\checkpoints\student_best.pt --batch_size 1 --warmup 20 --iters 200
```

如果要测试 batch 推理：

```bash
python tools/benchmark_latency.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --batch_size 32 \
  --warmup 20 \
  --iters 200
```

## 11. 导出 ONNX

导出 Student 图像编码器和文本编码器：

```bash
python tools/export_onnx.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --out_dir outputs/onnx \
  --opset 17
```

Windows PowerShell 单行：

```powershell
python tools\export_onnx.py --ckpt outputs\checkpoints\student_best.pt --out_dir outputs\onnx --opset 17
```

导出结果：

```text
outputs/onnx/student_image_encoder.onnx
outputs/onnx/student_text_encoder.onnx
```

ONNX 文件可用于后续部署测试。导出后建议用 `onnxruntime` 在目标服务器上再做一次加载验证。

## 12. Smoke test

如果只是验证代码链路，不想先准备真实数据，可以运行 smoke test。该流程会生成合成 RGB 图像和随机 Teacher 特征，并训练一个极小 Student 一轮。

```bash
python tools/create_smoke_data.py --config configs/smoke_resnet18.yaml
python tools/train_distill.py --config configs/smoke_resnet18.yaml
python tools/eval_zeroshot.py \
  --ckpt outputs/smoke/checkpoints/student_best.pt \
  --metadata_csv data/smoke/metadata.csv \
  --batch_size 2 \
  --num_workers 0
python tools/benchmark_latency.py \
  --ckpt outputs/smoke/checkpoints/student_best.pt \
  --iters 2 \
  --warmup 1
python tools/export_onnx.py \
  --ckpt outputs/smoke/checkpoints/student_best.pt \
  --out_dir outputs/smoke/onnx
```

smoke test 只验证工程能否跑通，不能作为精度实验结果。

## 13. 服务器完整运行顺序

在服务器上建议按下面顺序执行：

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

python tools/train_distill.py --config configs/distill_resnet18.yaml

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

如果训练被中断：

```bash
python tools/train_distill.py \
  --config configs/distill_resnet18.yaml \
  --resume outputs/checkpoints/student_last.pt
```

## 14. 需要记录的实验结果

正式实验至少记录：

```text
数据集名称与类别数
训练/验证划分
prompt 模板
Teacher zero-shot accuracy
Student zero-shot accuracy
Student / Teacher accuracy retention
Student 参数量
checkpoint 大小
latency
FPS
ONNX 导出是否成功
训练环境：OS、Python、torch、CUDA、GPU 型号
```

建议进一步做消融实验：

```text
1. Student only: L_clip
2. Feature distillation: L_img + L_txt + L_clip
3. Similarity distillation: L_sim + L_clip
4. Full method: L_img + L_txt + 2 * L_sim + L_clip
```

医学图像场景下，OpenCLIP ViT-B/32 仍然是通用自然图像 Teacher。若用于甲状腺超声等任务，应谨慎解释 zero-shot 结果，并优先设计领域 prompt 或替换医学 CLIP Teacher。
