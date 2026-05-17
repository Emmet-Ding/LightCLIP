# LightCLIP

LightCLIP implements a lightweight CLIP student trained by distilling a frozen OpenCLIP teacher. The student learns image features, text features, and the cross-modal similarity structure, so deployment can keep only the compact student model.

## Repository Layout

```text
configs/                 Training configs
datasets/                Image-text dataset and CLIP-style transforms
losses/                  Feature, similarity, and CLIP contrastive losses
models/                  ResNet-18/MobileNetV3 student CLIP
tools/                   Metadata, teacher cache, train, eval, benchmark, ONNX scripts
data/                    Local data location, ignored by git
outputs/                 Generated checkpoints/logs/exports, ignored by git
```

## Setup

```bash
conda create -n lightclip_distill python=3.10 -y
conda activate lightclip_distill
pip install -r requirements.txt
```

For CUDA-specific PyTorch wheels, install PyTorch first from the matching PyTorch index, then run `pip install -r requirements.txt`.

## Data Format

Use an ImageFolder layout:

```text
data/images/
├── cat/
│   ├── 0001.jpg
│   └── 0002.jpg
└── dog/
    ├── 0001.jpg
    └── 0002.jpg
```

Generate metadata:

```bash
python tools/build_metadata_from_imagefolder.py \
  --image_root data/images \
  --out_csv data/metadata.csv \
  --prompt "a photo of a {}"
```

The generated CSV contains `image_path`, `text`, and `label`.

## Main Workflow

Precompute frozen teacher features:

```bash
python tools/precompute_teacher_features.py \
  --metadata_csv data/metadata.csv \
  --out data/teacher_cache/teacher_features.pt \
  --model_name ViT-B-32 \
  --pretrained openai \
  --batch_size 64
```

Train the student:

```bash
python tools/train_distill.py --config configs/distill_resnet18.yaml
```

Evaluate student zero-shot accuracy:

```bash
python tools/eval_zeroshot.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --metadata_csv data/metadata.csv \
  --prompt "a photo of a {}"
```

Evaluate the teacher baseline:

```bash
python tools/eval_teacher_zeroshot.py \
  --metadata_csv data/metadata.csv \
  --model_name ViT-B-32 \
  --pretrained openai \
  --prompt "a photo of a {}"
```

Benchmark latency and FPS:

```bash
python tools/benchmark_latency.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --batch_size 1 \
  --iters 200
```

Export ONNX encoders:

```bash
python tools/export_onnx.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --out_dir outputs/onnx
```

## Local Smoke Test

This creates synthetic RGB images and random teacher features, then trains a tiny student for one epoch:

```bash
python tools/create_smoke_data.py --config configs/smoke_resnet18.yaml
python tools/train_distill.py --config configs/smoke_resnet18.yaml
python tools/benchmark_latency.py --ckpt outputs/smoke/checkpoints/student_best.pt --iters 2 --warmup 1
python tools/export_onnx.py --ckpt outputs/smoke/checkpoints/student_best.pt --out_dir outputs/smoke/onnx
```

The smoke test validates code wiring only. It is not an accuracy experiment because the teacher cache is synthetic.

## Notes for Server Experiments

- Keep `data/images`, `data/metadata.csv`, `data/teacher_cache/*.pt`, and `outputs/` outside git.
- For small GPUs, set `train.batch_size: 16` and `train.grad_accum_steps: 4`.
- For medical image experiments, start with domain-specific prompts and treat OpenCLIP ViT-B/32 as a general-domain teacher unless a medical CLIP teacher is substituted.

