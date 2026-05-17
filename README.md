# LightCLIP

LightCLIP 是一个轻量化 CLIP 蒸馏工程。该项目使用冻结的 OpenCLIP ViT-B/32 作为 Teacher，将图像特征、文本特征和跨模态相似度结构蒸馏到较小的 Student CLIP 中。部署阶段只需要保留 Student 模型。

当前仓库已经包含完整工程流程：

- 从 ImageFolder 生成 `metadata.csv`
- 使用 Teacher 预计算图像/文本特征
- 训练 ResNet-18 + Tiny Transformer Student CLIP
- 评估 Student zero-shot accuracy
- 评估 Teacher baseline
- 测试 latency/FPS
- 导出 ONNX 图像编码器和文本编码器
- 提供本地 smoke test 验证代码链路

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
```

## 环境安装

推荐使用 Python 3.10：

```bash
conda create -n lightclip_distill python=3.10 -y
conda activate lightclip_distill
pip install -r requirements.txt
```

如果服务器需要指定 CUDA 版本，建议先按 PyTorch 官方命令安装对应 CUDA wheel，再执行：

```bash
pip install -r requirements.txt
```

## Teacher 权重

OpenCLIP ViT-B/32 OpenAI 权重已通过 Git LFS 上传到：

```text
weights/openclip_vit_b_32_openai.pt
```

服务器 clone 后需要执行：

```bash
git lfs pull
```

如果没有执行 `git lfs pull`，该文件可能只是 LFS 指针文件，无法作为真实权重加载。

该 `.pt` 权重来自 OpenAI 公开 checkpoint URL，是 TorchScript archive。PyTorch 2.6+ 默认 `weights_only=True` 时会拒绝通过 `torch.load` 加载此类文件，因此 Teacher 脚本在使用 `--weights_path` 时会显式传入 `weights_only=False`。

## 数据格式

默认使用 ImageFolder 形式：

```text
data/images/
  cat/
    0001.jpg
    0002.jpg
  dog/
    0001.jpg
    0002.jpg
```

生成 `metadata.csv`：

```bash
python tools/build_metadata_from_imagefolder.py \
  --image_root data/images \
  --out_csv data/metadata.csv \
  --prompt "a photo of a {}"
```

生成的 CSV 包含三列：

```text
image_path,text,label
```

医学图像任务可以改用领域 prompt，例如：

```bash
python tools/build_metadata_from_imagefolder.py \
  --image_root data/images \
  --out_csv data/metadata.csv \
  --prompt "an ultrasound image of {}"
```

## 主流程

### 1. 预计算 Teacher 特征

```bash
python tools/precompute_teacher_features.py \
  --metadata_csv data/metadata.csv \
  --out data/teacher_cache/teacher_features.pt \
  --model_name ViT-B-32 \
  --weights_path weights/openclip_vit_b_32_openai.pt \
  --batch_size 64
```

### 2. 训练 Student

```bash
python tools/train_distill.py --config configs/distill_resnet18.yaml
```

训练输出：

```text
outputs/checkpoints/student_best.pt
outputs/checkpoints/student_last.pt
outputs/logs/train_log.csv
```

### 3. 评估 Student zero-shot

```bash
python tools/eval_zeroshot.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --metadata_csv data/metadata.csv \
  --prompt "a photo of a {}"
```

### 4. 评估 Teacher baseline

```bash
python tools/eval_teacher_zeroshot.py \
  --metadata_csv data/metadata.csv \
  --model_name ViT-B-32 \
  --weights_path weights/openclip_vit_b_32_openai.pt \
  --prompt "a photo of a {}"
```

### 5. 测试推理速度

```bash
python tools/benchmark_latency.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --batch_size 1 \
  --iters 200
```

### 6. 导出 ONNX

```bash
python tools/export_onnx.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --out_dir outputs/onnx
```

导出结果：

```text
outputs/onnx/student_image_encoder.onnx
outputs/onnx/student_text_encoder.onnx
```

## 本地 smoke test

如果只是验证工程链路，可以运行 smoke test。该流程会生成合成 RGB 图像和随机 Teacher 特征，并训练一个极小 Student 一轮。

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

smoke test 只验证代码能否跑通，不能作为精度实验结果。

## 小显存配置

如果训练时显存不足，可以修改 `configs/distill_resnet18.yaml`：

```yaml
train:
  batch_size: 16
  grad_accum_steps: 4
```

这样可以降低单步显存压力，同时保持近似有效 batch size。

## 后续实验建议

正式实验至少需要记录：

- Teacher zero-shot accuracy
- Student zero-shot accuracy
- Student / Teacher accuracy retention
- 参数量
- checkpoint 大小
- latency
- FPS
- ONNX 导出是否成功

建议进一步做消融实验：

```text
1. Student only: L_clip
2. Feature distillation: L_img + L_txt + L_clip
3. Similarity distillation: L_sim + L_clip
4. Full method: L_img + L_txt + 2 * L_sim + L_clip
```

医学图像场景下，OpenCLIP ViT-B/32 仍然是通用自然图像 Teacher。若用于甲状腺超声等任务，应谨慎解释 zero-shot 结果，并优先设计领域 prompt 或替换医学 CLIP Teacher。

