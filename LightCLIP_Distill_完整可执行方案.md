# LightCLIP-Distill：基于知识蒸馏的轻量化 CLIP 完整执行方案

版本：v1.0  
目标：训练一个轻量级 Student CLIP，使其在部署阶段不再加载原版大 CLIP，仅使用小模型完成图像-文本匹配、zero-shot 分类或检索。

---

## 0. 项目目标

本方案要完成的事情可以简化成一句话：

> 用冻结的大型 CLIP 作为 Teacher，把它的图像特征、文本特征和图文相似度结构蒸馏到一个轻量 Student 模型中；最终部署时只保留 Student。

训练阶段：

```text
图像 x + 文本 t
     │
     ├── Teacher CLIP，冻结，不反向传播
     │       ├── teacher_image_feature
     │       ├── teacher_text_feature
     │       └── teacher_similarity_matrix
     │
     └── Student CLIP，参与训练
             ├── student_image_feature
             ├── student_text_feature
             └── student_similarity_matrix
```

部署阶段：

```text
图像 x + 文本 t
     │
     └── Student CLIP
             └── 图文相似度 / zero-shot 分类结果
```

本方案的核心约束：

1. 原版 CLIP 只用于训练阶段产生监督信号。
2. Student 需要同时学习视觉特征、文本特征和跨模态相似度结构。
3. 最终部署时不加载 Teacher。
4. 评估必须包含模型大小、内存占用、推理速度和 zero-shot 精度保留率。

---

## 1. 推荐实验路线

建议先做最小可运行版本，再逐步扩展。

### 1.1 最小可运行版本

Teacher：

```text
OpenCLIP ViT-B/32
```

Student：

```text
图像编码器：ResNet-18
文本编码器：2-layer Tiny Transformer
输出维度：512
```

训练数据：

```text
任意 image-text pairs
或者 ImageFolder 分类数据 + prompt 生成文本
```

损失函数：

```text
L = L_img + L_txt + 2 * L_sim + L_clip
```

评估：

```text
zero-shot accuracy
Student / Teacher accuracy retention
参数量
模型文件大小
CPU/GPU latency
FPS
```

### 1.2 后续扩展版本

在 ResNet-18 Student 跑通后，再替换：

```text
Student image encoder:
1. ResNet-18
2. MobileNetV3
3. TinyViT / DeiT-Tiny
```

做消融实验：

```text
1. Student only, no distillation
2. image/text feature distillation only
3. similarity matrix distillation only
4. full distillation
```

---

## 2. 环境搭建

### 2.1 创建 conda 环境

```bash
conda create -n lightclip_distill python=3.10 -y
conda activate lightclip_distill
```

### 2.2 安装 PyTorch

CUDA 12.x 环境可参考：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CPU 环境可使用：

```bash
pip install torch torchvision torchaudio
```

### 2.3 安装项目依赖

```bash
pip install open_clip_torch timm transformers pillow pandas numpy tqdm scikit-learn matplotlib pyyaml psutil onnx onnxruntime
```

验证安装：

```bash
python - <<'PY'
import torch
import open_clip
import torchvision
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("open_clip ok")
PY
```

---

## 3. 项目目录结构

建议建立如下目录：

```bash
mkdir -p LightCLIP-Distill
cd LightCLIP-Distill

mkdir -p configs
mkdir -p data/images
mkdir -p data/teacher_cache
mkdir -p models
mkdir -p losses
mkdir -p datasets
mkdir -p tools
mkdir -p outputs/checkpoints
mkdir -p outputs/logs
mkdir -p outputs/figures
mkdir -p outputs/tables
```

最终结构：

```text
LightCLIP-Distill/
│
├── configs/
│   └── distill_resnet18.yaml
│
├── data/
│   ├── metadata.csv
│   ├── images/
│   └── teacher_cache/
│       └── teacher_features.pt
│
├── datasets/
│   └── image_text_dataset.py
│
├── models/
│   └── student_clip.py
│
├── losses/
│   └── distill_losses.py
│
├── tools/
│   ├── build_metadata_from_imagefolder.py
│   ├── precompute_teacher_features.py
│   ├── train_distill.py
│   ├── eval_zeroshot.py
│   ├── benchmark_latency.py
│   └── export_onnx.py
│
└── outputs/
```

---

## 4. 数据准备

### 4.1 metadata.csv 格式

训练数据统一整理成一个 CSV：

```csv
image_path,text,label
data/images/cat/0001.jpg,a photo of a cat,cat
data/images/dog/0002.jpg,a photo of a dog,dog
```

必须包含：

```text
image_path：图像路径
text：与图像对应的文本描述
label：类别名，可选，但 zero-shot 评估时建议保留
```

### 4.2 从 ImageFolder 自动生成 metadata.csv

适用于这样的数据：

```text
data/images/
├── cat/
│   ├── 0001.jpg
│   └── 0002.jpg
├── dog/
│   ├── 0001.jpg
│   └── 0002.jpg
└── car/
    ├── 0001.jpg
    └── 0002.jpg
```

创建脚本：

```bash
cat > tools/build_metadata_from_imagefolder.py <<'PY'
import argparse
from pathlib import Path
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="data/metadata.csv")
    parser.add_argument("--prompt", type=str, default="a photo of a {}")
    args = parser.parse_args()

    image_root = Path(args.image_root)
    rows = []

    for class_dir in sorted(image_root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for p in sorted(class_dir.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                rows.append({
                    "image_path": str(p),
                    "text": args.prompt.format(label.replace("_", " ")),
                    "label": label
                })

    if len(rows) == 0:
        raise RuntimeError(f"No images found under {image_root}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    main()
PY
```

运行：

```bash
python tools/build_metadata_from_imagefolder.py \
  --image_root data/images \
  --out_csv data/metadata.csv \
  --prompt "a photo of a {}"
```

医学图像可把 prompt 改成：

```bash
python tools/build_metadata_from_imagefolder.py \
  --image_root data/images \
  --out_csv data/metadata.csv \
  --prompt "an ultrasound image of {}"
```

---

## 5. 配置文件

创建配置文件：

```bash
cat > configs/distill_resnet18.yaml <<'YAML'
seed: 42

data:
  metadata_csv: data/metadata.csv
  teacher_cache: data/teacher_cache/teacher_features.pt
  image_size: 224
  context_length: 77

teacher:
  model_name: ViT-B-32
  pretrained: openai

student:
  image_encoder: resnet18
  text_width: 256
  text_layers: 2
  text_heads: 4
  embed_dim: 512
  vocab_size: 49408
  context_length: 77

train:
  epochs: 30
  batch_size: 64
  num_workers: 4
  lr: 1.0e-4
  weight_decay: 0.05
  warmup_epochs: 2
  grad_accum_steps: 1
  amp: true

loss:
  lambda_img: 1.0
  lambda_txt: 1.0
  lambda_sim: 2.0
  lambda_clip: 1.0
  kd_temperature: 2.0

output:
  ckpt_dir: outputs/checkpoints
  log_dir: outputs/logs
YAML
```

---

## 6. 数据集代码

创建数据集文件：

```bash
cat > datasets/image_text_dataset.py <<'PY'
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

def build_image_transform(image_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

class ImageTextDataset(Dataset):
    def __init__(self, metadata_csv, tokenizer, image_size=224, train=True):
        self.df = pd.read_csv(metadata_csv)
        required = {"image_path", "text"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"metadata_csv missing columns: {missing}")

        self.image_paths = self.df["image_path"].astype(str).tolist()
        self.texts = self.df["text"].astype(str).tolist()
        self.labels = self.df["label"].astype(str).tolist() if "label" in self.df.columns else [""] * len(self.df)

        self.transform = build_image_transform(image_size=image_size, train=train)
        self.tokens = tokenizer(self.texts)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "text_tokens": self.tokens[idx],
            "text": self.texts[idx],
            "label": self.labels[idx],
            "index": torch.tensor(idx, dtype=torch.long),
        }
PY
```

---

## 7. Student 模型代码

创建 Student CLIP：

```bash
cat > models/student_clip.py <<'PY'
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet18ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512, pretrained=False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        in_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.proj(feat)
        return feat

class MobileNetV3ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512, pretrained=False):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
        in_dim = net.classifier[-1].in_features
        net.classifier = nn.Identity()
        self.backbone = net
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.proj(feat)
        return feat

class TinyTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size=49408,
        context_length=77,
        width=256,
        layers=2,
        heads=4,
        embed_dim=512,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.ln_final = nn.LayerNorm(width)
        self.proj = nn.Linear(width, embed_dim)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, text_tokens):
        # text_tokens: [B, 77]
        x = self.token_embedding(text_tokens)
        x = x + self.positional_embedding[:x.size(1)].unsqueeze(0)

        # causal mask is not mandatory for this distilled student.
        x = self.transformer(x)
        x = self.ln_final(x)

        # CLIP convention: EOT token is usually the highest token id in each sequence.
        eot_pos = text_tokens.argmax(dim=-1)
        x = x[torch.arange(x.size(0), device=x.device), eot_pos]
        x = self.proj(x)
        return x

class StudentCLIP(nn.Module):
    def __init__(
        self,
        image_encoder="resnet18",
        embed_dim=512,
        vocab_size=49408,
        context_length=77,
        text_width=256,
        text_layers=2,
        text_heads=4,
    ):
        super().__init__()

        if image_encoder == "resnet18":
            self.image_encoder = ResNet18ImageEncoder(embed_dim=embed_dim, pretrained=False)
        elif image_encoder == "mobilenetv3":
            self.image_encoder = MobileNetV3ImageEncoder(embed_dim=embed_dim, pretrained=False)
        else:
            raise ValueError(f"Unsupported image_encoder: {image_encoder}")

        self.text_encoder = TinyTextEncoder(
            vocab_size=vocab_size,
            context_length=context_length,
            width=text_width,
            layers=text_layers,
            heads=text_heads,
            embed_dim=embed_dim,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        z = self.image_encoder(image)
        z = F.normalize(z, dim=-1)
        return z

    def encode_text(self, text_tokens):
        w = self.text_encoder(text_tokens)
        w = F.normalize(w, dim=-1)
        return w

    def forward(self, image, text_tokens):
        image_feat = self.encode_image(image)
        text_feat = self.encode_text(text_tokens)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = logit_scale * image_feat @ text_feat.t()
        return image_feat, text_feat, logits
PY
```

---

## 8. 损失函数代码

创建损失函数：

```bash
cat > losses/distill_losses.py <<'PY'
import torch
import torch.nn.functional as F

def cosine_distill_loss(student_feat, teacher_feat):
    student_feat = F.normalize(student_feat, dim=-1)
    teacher_feat = F.normalize(teacher_feat, dim=-1)
    return 1.0 - F.cosine_similarity(student_feat, teacher_feat, dim=-1).mean()

def clip_contrastive_loss(logits):
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2.0

def similarity_distill_loss(student_logits, teacher_logits, temperature=2.0):
    T = temperature

    loss_i2t = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)

    loss_t2i = F.kl_div(
        F.log_softmax(student_logits.t() / T, dim=1),
        F.softmax(teacher_logits.t() / T, dim=1),
        reduction="batchmean",
    ) * (T * T)

    return (loss_i2t + loss_t2i) / 2.0
PY
```

---

## 9. 预计算 Teacher 特征

创建脚本：

```bash
cat > tools/precompute_teacher_features.py <<'PY'
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import open_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, default="data/metadata.csv")
    parser.add_argument("--out", type=str, default="data/teacher_cache/teacher_features.pt")
    parser.add_argument("--model_name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.metadata_csv)
    image_paths = df["image_path"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    image_features = []
    text_features = []

    with torch.no_grad():
        for start in tqdm(range(0, len(df), args.batch_size), desc="Precompute teacher"):
            end = min(start + args.batch_size, len(df))
            batch_paths = image_paths[start:end]
            batch_texts = texts[start:end]

            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
            images = torch.stack(images).to(device)

            text_tokens = tokenizer(batch_texts).to(device)

            img_feat = model.encode_image(images)
            txt_feat = model.encode_text(text_tokens)

            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            image_features.append(img_feat.cpu())
            text_features.append(txt_feat.cpu())

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "image_features": image_features,
        "text_features": text_features,
        "image_paths": image_paths,
        "texts": texts,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
    }, out)

    print("Saved teacher features:")
    print("  path:", out)
    print("  image_features:", tuple(image_features.shape))
    print("  text_features:", tuple(text_features.shape))


if __name__ == "__main__":
    main()
PY
```

运行：

```bash
python tools/precompute_teacher_features.py \
  --metadata_csv data/metadata.csv \
  --out data/teacher_cache/teacher_features.pt \
  --model_name ViT-B-32 \
  --pretrained openai \
  --batch_size 64
```

---

## 10. 训练 Student

创建训练脚本：

```bash
cat > tools/train_distill.py <<'PY'
import argparse
import os
import random
from pathlib import Path

import yaml
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import open_clip

from datasets.image_text_dataset import ImageTextDataset
from models.student_clip import StudentCLIP
from losses.distill_losses import (
    cosine_distill_loss,
    clip_contrastive_loss,
    similarity_distill_loss,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/distill_resnet18.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = open_clip.get_tokenizer(cfg["teacher"]["model_name"])

    full_dataset = ImageTextDataset(
        metadata_csv=cfg["data"]["metadata_csv"],
        tokenizer=tokenizer,
        image_size=cfg["data"]["image_size"],
        train=True,
    )

    n_total = len(full_dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    teacher_cache = torch.load(cfg["data"]["teacher_cache"], map_location="cpu")
    teacher_img_all = teacher_cache["image_features"].float()
    teacher_txt_all = teacher_cache["text_features"].float()

    student = StudentCLIP(
        image_encoder=cfg["student"]["image_encoder"],
        embed_dim=cfg["student"]["embed_dim"],
        vocab_size=cfg["student"]["vocab_size"],
        context_length=cfg["student"]["context_length"],
        text_width=cfg["student"]["text_width"],
        text_layers=cfg["student"]["text_layers"],
        text_heads=cfg["student"]["text_heads"],
    ).to(device)

    print(f"Student params: {count_params(student) / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]) and device == "cuda")

    epochs = int(cfg["train"]["epochs"])
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))

    ckpt_dir = Path(cfg["output"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        student.train()

        running = {
            "loss": 0.0,
            "img": 0.0,
            "txt": 0.0,
            "sim": 0.0,
            "clip": 0.0,
        }

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs} train")
        for step, batch in enumerate(pbar, start=1):
            images = batch["image"].to(device, non_blocking=True)
            text_tokens = batch["text_tokens"].to(device, non_blocking=True)
            indices = batch["index"].long()

            teacher_img = teacher_img_all[indices].to(device, non_blocking=True)
            teacher_txt = teacher_txt_all[indices].to(device, non_blocking=True)
            teacher_img = F.normalize(teacher_img, dim=-1)
            teacher_txt = F.normalize(teacher_txt, dim=-1)

            with torch.cuda.amp.autocast(enabled=bool(cfg["train"]["amp"]) and device == "cuda"):
                student_img, student_txt, student_logits = student(images, text_tokens)
                teacher_logits = teacher_img @ teacher_txt.t()

                loss_img = cosine_distill_loss(student_img, teacher_img)
                loss_txt = cosine_distill_loss(student_txt, teacher_txt)
                loss_sim = similarity_distill_loss(
                    student_logits,
                    teacher_logits,
                    temperature=float(cfg["loss"]["kd_temperature"]),
                )
                loss_clip = clip_contrastive_loss(student_logits)

                loss = (
                    float(cfg["loss"]["lambda_img"]) * loss_img
                    + float(cfg["loss"]["lambda_txt"]) * loss_txt
                    + float(cfg["loss"]["lambda_sim"]) * loss_sim
                    + float(cfg["loss"]["lambda_clip"]) * loss_clip
                )

                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if step % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            raw_loss = loss.item() * grad_accum_steps
            running["loss"] += raw_loss
            running["img"] += loss_img.item()
            running["txt"] += loss_txt.item()
            running["sim"] += loss_sim.item()
            running["clip"] += loss_clip.item()

            pbar.set_postfix({
                "loss": f"{raw_loss:.4f}",
                "img": f"{loss_img.item():.3f}",
                "txt": f"{loss_txt.item():.3f}",
                "sim": f"{loss_sim.item():.3f}",
                "clip": f"{loss_clip.item():.3f}",
            })

        train_loss = running["loss"] / max(1, len(train_loader))

        student.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:03d}/{epochs} val"):
                images = batch["image"].to(device, non_blocking=True)
                text_tokens = batch["text_tokens"].to(device, non_blocking=True)
                indices = batch["index"].long()

                teacher_img = teacher_img_all[indices].to(device, non_blocking=True)
                teacher_txt = teacher_txt_all[indices].to(device, non_blocking=True)
                teacher_img = F.normalize(teacher_img, dim=-1)
                teacher_txt = F.normalize(teacher_txt, dim=-1)

                student_img, student_txt, student_logits = student(images, text_tokens)
                teacher_logits = teacher_img @ teacher_txt.t()

                loss_img = cosine_distill_loss(student_img, teacher_img)
                loss_txt = cosine_distill_loss(student_txt, teacher_txt)
                loss_sim = similarity_distill_loss(
                    student_logits,
                    teacher_logits,
                    temperature=float(cfg["loss"]["kd_temperature"]),
                )
                loss_clip = clip_contrastive_loss(student_logits)

                loss = (
                    float(cfg["loss"]["lambda_img"]) * loss_img
                    + float(cfg["loss"]["lambda_txt"]) * loss_txt
                    + float(cfg["loss"]["lambda_sim"]) * loss_sim
                    + float(cfg["loss"]["lambda_clip"]) * loss_clip
                )

                val_loss_sum += loss.item()

        val_loss = val_loss_sum / max(1, len(val_loader))

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        last_path = ckpt_dir / "student_last.pt"
        torch.save({
            "model": student.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "val_loss": val_loss,
        }, last_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = ckpt_dir / "student_best.pt"
            torch.save({
                "model": student.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "val_loss": val_loss,
            }, best_path)
            print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
PY
```

运行训练：

```bash
python tools/train_distill.py --config configs/distill_resnet18.yaml
```

---

## 11. Zero-shot 评估

### 11.1 原理

对每个类别构造 prompt：

```text
a photo of a {class_name}
```

然后计算：

```text
image_feature × text_feature
```

预测相似度最高的类别。

精度保留率：

\[
Retention = \frac{Acc_{Student}}{Acc_{Teacher}}
\]

### 11.2 Student zero-shot 评估脚本

```bash
cat > tools/eval_zeroshot.py <<'PY'
import argparse
from pathlib import Path

import yaml
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import open_clip

from datasets.image_text_dataset import build_image_transform
from models.student_clip import StudentCLIP


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_csv, image_size=224):
        self.df = pd.read_csv(metadata_csv)
        if "label" not in self.df.columns:
            raise ValueError("metadata_csv must contain label column for zero-shot evaluation")
        self.image_paths = self.df["image_path"].astype(str).tolist()
        self.labels = self.df["label"].astype(str).tolist()
        self.transform = build_image_transform(image_size=image_size, train=False)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/student_best.pt")
    parser.add_argument("--metadata_csv", type=str, default="data/metadata.csv")
    parser.add_argument("--prompt", type=str, default="a photo of a {}")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]

    tokenizer = open_clip.get_tokenizer(cfg["teacher"]["model_name"])

    student = StudentCLIP(
        image_encoder=cfg["student"]["image_encoder"],
        embed_dim=cfg["student"]["embed_dim"],
        vocab_size=cfg["student"]["vocab_size"],
        context_length=cfg["student"]["context_length"],
        text_width=cfg["student"]["text_width"],
        text_layers=cfg["student"]["text_layers"],
        text_heads=cfg["student"]["text_heads"],
    ).to(device)
    student.load_state_dict(ckpt["model"], strict=True)
    student.eval()

    df = pd.read_csv(args.metadata_csv)
    class_names = sorted(df["label"].astype(str).unique().tolist())
    prompts = [args.prompt.format(c.replace("_", " ")) for c in class_names]

    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = student.encode_text(text_tokens)

    dataset = EvalDataset(args.metadata_csv, image_size=cfg["data"]["image_size"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    class_to_idx = {c: i for i, c in enumerate(class_names)}

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval student zero-shot"):
            images = images.to(device)
            image_features = student.encode_image(images)
            logits = image_features @ text_features.t()
            pred = logits.argmax(dim=1).cpu().tolist()

            target = [class_to_idx[x] for x in labels]
            correct += sum(int(p == y) for p, y in zip(pred, target))
            total += len(target)

    acc = correct / max(1, total)
    print(f"Student zero-shot acc: {acc:.4f}")
    print(f"Num classes: {len(class_names)}")
    print(f"Num samples: {total}")


if __name__ == "__main__":
    main()
PY
```

运行：

```bash
python tools/eval_zeroshot.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --metadata_csv data/metadata.csv \
  --prompt "a photo of a {}"
```

---

## 12. 推理速度与模型大小测试

创建 benchmark 脚本：

```bash
cat > tools/benchmark_latency.py <<'PY'
import argparse
import time
from pathlib import Path

import torch
import yaml
from models.student_clip import StudentCLIP


def count_params(model):
    return sum(p.numel() for p in model.parameters())

def get_file_size_mb(path):
    return Path(path).stat().st_size / 1024 / 1024

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/student_best.pt")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]

    model = StudentCLIP(
        image_encoder=cfg["student"]["image_encoder"],
        embed_dim=cfg["student"]["embed_dim"],
        vocab_size=cfg["student"]["vocab_size"],
        context_length=cfg["student"]["context_length"],
        text_width=cfg["student"]["text_width"],
        text_layers=cfg["student"]["text_layers"],
        text_heads=cfg["student"]["text_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    image = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(device)
    tokens = torch.randint(
        low=0,
        high=cfg["student"]["vocab_size"],
        size=(args.batch_size, cfg["student"]["context_length"]),
        device=device,
    )

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(image, tokens)

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(args.iters):
            _ = model(image, tokens)

        if device == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()

    avg_ms = (t1 - t0) * 1000 / args.iters
    fps = args.batch_size * 1000 / avg_ms

    print(f"Device: {device}")
    print(f"Params: {count_params(model) / 1e6:.2f} M")
    print(f"Checkpoint size: {get_file_size_mb(args.ckpt):.2f} MB")
    print(f"Batch size: {args.batch_size}")
    print(f"Latency: {avg_ms:.3f} ms / batch")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
PY
```

运行：

```bash
python tools/benchmark_latency.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --batch_size 1 \
  --iters 200
```

测试 batch size 32：

```bash
python tools/benchmark_latency.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --batch_size 32 \
  --iters 200
```

---

## 13. 导出 ONNX

部署时可以分别导出图像编码器和文本编码器。

创建脚本：

```bash
cat > tools/export_onnx.py <<'PY'
import argparse
from pathlib import Path
import torch

from models.student_clip import StudentCLIP


class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, image):
        return self.student.encode_image(image)


class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, tokens):
        return self.student.encode_text(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/student_best.pt")
    parser.add_argument("--out_dir", type=str, default="outputs/onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]

    student = StudentCLIP(
        image_encoder=cfg["student"]["image_encoder"],
        embed_dim=cfg["student"]["embed_dim"],
        vocab_size=cfg["student"]["vocab_size"],
        context_length=cfg["student"]["context_length"],
        text_width=cfg["student"]["text_width"],
        text_layers=cfg["student"]["text_layers"],
        text_heads=cfg["student"]["text_heads"],
    )
    student.load_state_dict(ckpt["model"], strict=True)
    student.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_model = ImageEncoderWrapper(student).eval()
    text_model = TextEncoderWrapper(student).eval()

    dummy_image = torch.randn(1, 3, cfg["data"]["image_size"], cfg["data"]["image_size"])
    dummy_tokens = torch.randint(
        low=0,
        high=cfg["student"]["vocab_size"],
        size=(1, cfg["student"]["context_length"]),
        dtype=torch.long,
    )

    torch.onnx.export(
        image_model,
        dummy_image,
        str(out_dir / "student_image_encoder.onnx"),
        input_names=["image"],
        output_names=["image_feature"],
        dynamic_axes={"image": {0: "batch"}, "image_feature": {0: "batch"}},
        opset_version=args.opset,
    )

    torch.onnx.export(
        text_model,
        dummy_tokens,
        str(out_dir / "student_text_encoder.onnx"),
        input_names=["tokens"],
        output_names=["text_feature"],
        dynamic_axes={"tokens": {0: "batch"}, "text_feature": {0: "batch"}},
        opset_version=args.opset,
    )

    print(f"Exported ONNX models to {out_dir}")


if __name__ == "__main__":
    main()
PY
```

运行：

```bash
python tools/export_onnx.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --out_dir outputs/onnx
```

输出：

```text
outputs/onnx/student_image_encoder.onnx
outputs/onnx/student_text_encoder.onnx
```

---

## 14. 完整运行顺序

从零开始执行：

```bash
conda activate lightclip_distill
cd LightCLIP-Distill
```

准备数据：

```bash
python tools/build_metadata_from_imagefolder.py \
  --image_root data/images \
  --out_csv data/metadata.csv \
  --prompt "a photo of a {}"
```

预计算 Teacher：

```bash
python tools/precompute_teacher_features.py \
  --metadata_csv data/metadata.csv \
  --out data/teacher_cache/teacher_features.pt \
  --model_name ViT-B-32 \
  --pretrained openai \
  --batch_size 64
```

训练 Student：

```bash
python tools/train_distill.py \
  --config configs/distill_resnet18.yaml
```

zero-shot 评估：

```bash
python tools/eval_zeroshot.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --metadata_csv data/metadata.csv \
  --prompt "a photo of a {}"
```

速度测试：

```bash
python tools/benchmark_latency.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --batch_size 1 \
  --iters 200
```

导出 ONNX：

```bash
python tools/export_onnx.py \
  --ckpt outputs/checkpoints/student_best.pt \
  --out_dir outputs/onnx
```

---

## 15. 消融实验设计

为了证明方案有效，建议至少做 5 组。

### Group A：Teacher CLIP baseline

```text
原版 CLIP ViT-B/32
不训练
直接 zero-shot 测试
```

作用：

```text
给出精度上限
```

### Group B：Student without distillation

损失：

```text
L = L_clip
```

作用：

```text
验证小模型直接训练是否不足
```

### Group C：Feature distillation only

损失：

```text
L = L_img + L_txt + L_clip
```

作用：

```text
验证单独特征蒸馏的贡献
```

### Group D：Similarity distillation only

损失：

```text
L = L_sim + L_clip
```

作用：

```text
验证图文相似度矩阵蒸馏的贡献
```

### Group E：Full method

损失：

```text
L = L_img + L_txt + 2 * L_sim + L_clip
```

作用：

```text
最终主方案
```

---

## 16. 需要记录的实验结果

建议最终输出 4 张表和 3 张图。

### 16.1 模型规模表

| Model | Image Encoder | Text Encoder | Params | Checkpoint Size | Compression |
|---|---|---|---:|---:|---:|
| Teacher CLIP | ViT-B/32 | CLIP Text Transformer | 待测 | 待测 | 1.0x |
| Student-A | ResNet-18 | Tiny Transformer | 待测 | 待测 | 待测 |
| Student-B | MobileNetV3 | Tiny Transformer | 待测 | 待测 | 待测 |

压缩率计算：

\[
Compression = 1 - \frac{Size_{Student}}{Size_{Teacher}}
\]

### 16.2 推理速度表

| Model | Device | Precision | Batch Size | Latency | FPS |
|---|---|---|---:|---:|---:|
| Teacher CLIP | CPU/GPU | FP32 | 1 | 待测 | 待测 |
| Student | CPU/GPU | FP32 | 1 | 待测 | 待测 |
| Student ONNX | CPU | FP32/INT8 | 1 | 待测 | 待测 |

### 16.3 Zero-shot 精度表

| Model | Dataset | Accuracy | Retention |
|---|---|---:|---:|
| Teacher CLIP | CIFAR-10/ImageFolder | 待测 | 100% |
| Student no distill | 同上 | 待测 | 待测 |
| Student feature distill | 同上 | 待测 | 待测 |
| Student full | 同上 | 待测 | 待测 |

精度保留率：

\[
Retention = \frac{Acc_{Student}}{Acc_{Teacher}}
\]

### 16.4 消融实验表

| Method | L_img | L_txt | L_sim | L_clip | Accuracy | FPS |
|---|---:|---:|---:|---:|---:|---:|
| Student only | 0 | 0 | 0 | 1 | 待测 | 待测 |
| Feature distill | 1 | 1 | 0 | 1 | 待测 | 待测 |
| Similarity distill | 0 | 0 | 1 | 1 | 待测 | 待测 |
| Full | 1 | 1 | 2 | 1 | 待测 | 待测 |

---

## 17. 关键公式

### 17.1 Teacher 图像特征

\[
z_i^T =
\frac{E_I^T(x_i)}
{\|E_I^T(x_i)\|}
\]

### 17.2 Teacher 文本特征

\[
w_i^T =
\frac{E_T^T(t_i)}
{\|E_T^T(t_i)\|}
\]

### 17.3 Student 图像特征

\[
z_i^S =
\frac{E_I^S(x_i)}
{\|E_I^S(x_i)\|}
\]

### 17.4 Student 文本特征

\[
w_i^S =
\frac{E_T^S(t_i)}
{\|E_T^S(t_i)\|}
\]

### 17.5 图像特征蒸馏

\[
\mathcal{L}_{img}
=
1 - \cos(z_i^S, z_i^T)
\]

### 17.6 文本特征蒸馏

\[
\mathcal{L}_{txt}
=
1 - \cos(w_i^S, w_i^T)
\]

### 17.7 图文相似度矩阵

\[
S^T = Z^T (W^T)^\top
\]

\[
S^S = Z^S (W^S)^\top
\]

### 17.8 相似度矩阵蒸馏

\[
\mathcal{L}_{sim}
=
\frac{1}{2}
\left[
KL
\left(
softmax(S^T/T),
softmax(S^S/T)
\right)
+
KL
\left(
softmax((S^T)^\top/T),
softmax((S^S)^\top/T)
\right)
\right]
\]

### 17.9 CLIP 对比学习损失

\[
\mathcal{L}_{i2t}
=
-\frac{1}{N}
\sum_i
\log
\frac{
\exp(z_i^S \cdot w_i^S / \tau)
}{
\sum_j \exp(z_i^S \cdot w_j^S / \tau)
}
\]

\[
\mathcal{L}_{t2i}
=
-\frac{1}{N}
\sum_i
\log
\frac{
\exp(w_i^S \cdot z_i^S / \tau)
}{
\sum_j \exp(w_i^S \cdot z_j^S / \tau)
}
\]

\[
\mathcal{L}_{clip}
=
\frac{1}{2}
(\mathcal{L}_{i2t} + \mathcal{L}_{t2i})
\]

### 17.10 总损失

\[
\mathcal{L}
=
\lambda_1 \mathcal{L}_{img}
+
\lambda_2 \mathcal{L}_{txt}
+
\lambda_3 \mathcal{L}_{sim}
+
\lambda_4 \mathcal{L}_{clip}
\]

推荐初始权重：

\[
\lambda_1=1,\quad
\lambda_2=1,\quad
\lambda_3=2,\quad
\lambda_4=1
\]

---

## 18. 实际训练中的风险点与处理办法

### 18.1 Student 太小导致 zero-shot 能力下降

表现：

```text
训练 loss 下降，但 zero-shot accuracy 很低。
```

原因：

```text
Student 容量不足，无法承载 Teacher 的跨模态空间。
```

处理：

```text
1. 先用 ResNet-18，不要一开始就用极小 MobileNet。
2. 文本编码器不要低于 2 层。
3. embed_dim 先保持 512。
4. 数据量太小时，不要过度压缩模型。
```

### 18.2 batch size 太小导致对比学习不稳定

表现：

```text
L_clip 波动大，图文匹配能力差。
```

处理：

```text
1. 增大 batch size。
2. 使用 gradient accumulation。
3. 优先保证 batch 内负样本数量。
```

### 18.3 只做特征蒸馏，排序能力仍然差

表现：

```text
student embedding 和 teacher embedding 的 cosine 距离变近，
但分类时类别排序仍然错误。
```

处理：

```text
必须加入 L_sim，也就是图文相似度矩阵蒸馏。
```

### 18.4 医学数据场景中的风险

如果用于甲状腺超声或其他医学图像，原版 CLIP 的自然图像语义空间可能不能直接覆盖医学特征。

处理路线：

```text
1. 先在通用数据上验证蒸馏流程。
2. 再用医学图像和医学 prompt 做领域适配。
3. 医学任务中不要直接宣称保留完整开放世界能力。
4. 对固定医学标签任务，可以预计算医学类别文本特征，提高部署稳定性。
```

医学 prompt 示例：

```text
an ultrasound image of a benign thyroid nodule
an ultrasound image of a malignant thyroid nodule
an ultrasound image showing microcalcification
an ultrasound image showing irregular margin
an ultrasound image showing taller-than-wide shape
```

---

## 19. 最终交付物清单

完成该计划后，应至少交付以下内容：

```text
1. 训练代码
   - train_distill.py
   - student_clip.py
   - distill_losses.py

2. Teacher 特征缓存
   - teacher_features.pt

3. Student 模型权重
   - student_best.pt
   - student_last.pt

4. 评估结果
   - zero-shot accuracy
   - accuracy retention
   - ablation study

5. 部署文件
   - student_image_encoder.onnx
   - student_text_encoder.onnx

6. Benchmark 结果
   - 参数量
   - 模型大小
   - CPU/GPU latency
   - FPS
   - 显存/内存占用

7. 可视化图表
   - 模型大小 vs zero-shot accuracy
   - FPS vs zero-shot accuracy
   - 参数量 vs retention
```

---

## 20. 最小执行检查表

按顺序检查：

```text
[ ] conda 环境创建完成
[ ] PyTorch 和 open_clip 安装完成
[ ] data/images 按类别放好图片
[ ] data/metadata.csv 已生成
[ ] Teacher 特征已预计算
[ ] Student 训练能正常启动
[ ] outputs/checkpoints/student_best.pt 已生成
[ ] zero-shot 评估脚本能运行
[ ] benchmark 脚本能输出 FPS
[ ] ONNX 导出成功
```

---

## 21. 建议的第一轮实验参数

第一轮不追求最优，只追求跑通和建立 baseline。

```text
Teacher: ViT-B/32
Student image encoder: ResNet-18
Student text encoder: 2-layer Tiny Transformer
embed_dim: 512
batch_size: 64
epochs: 30
lr: 1e-4
weight_decay: 0.05
kd_temperature: 2.0
loss weights: 1, 1, 2, 1
```

如果显存不足：

```text
batch_size: 16 或 32
grad_accum_steps: 4
```

修改配置：

```yaml
train:
  batch_size: 16
  grad_accum_steps: 4
```

---

## 22. 第一轮结果应如何判断

第一轮实验完成后，主要看四个信号：

### 22.1 loss 是否正常下降

合理现象：

```text
train_loss 下降
val_loss 不剧烈爆炸
L_img 和 L_txt 逐渐下降
L_sim 不长期停滞
```

### 22.2 Student zero-shot 是否明显高于随机

例如 10 类任务：

```text
随机准确率约 10%
Student 应明显高于 10%
```

### 22.3 Student 是否小于 Teacher

用 checkpoint size 和参数量确认：

```text
Student 参数量应明显小于 Teacher
Student checkpoint size 应明显小于 Teacher
```

### 22.4 推理速度是否提升

用同一设备、同一 batch size 对比：

```text
Student latency 应低于 Teacher
Student FPS 应高于 Teacher
```

---

## 23. 后续论文式写法

方法部分可以命名为：

```text
Cross-modal Feature Distillation for Lightweight CLIP
```

核心模块：

```text
1. Teacher-guided Image Feature Distillation
2. Teacher-guided Text Feature Distillation
3. Cross-modal Similarity Matrix Distillation
4. Student CLIP Contrastive Learning
```

最终表述：

```text
We freeze a pretrained CLIP model as the teacher and train a lightweight student model to approximate the teacher's visual embedding, textual embedding, and cross-modal similarity distribution. During deployment, only the compact student model is retained, reducing memory consumption and inference latency while preserving zero-shot recognition ability.
```

中文表述：

```text
本方法冻结预训练 CLIP 作为知识教师，通过图像特征蒸馏、文本特征蒸馏和跨模态相似度矩阵蒸馏，将原模型的视觉-语言对齐空间迁移到轻量级 Student 模型中。部署阶段仅保留 Student，从而降低模型体积、内存占用和推理延迟，同时尽可能保留 zero-shot 分类能力。
```

---

## 24. 当前版本的边界

本方案能直接完成：

```text
1. Teacher-Student CLIP 蒸馏流程
2. ResNet-18 Student 训练
3. Tiny Transformer 文本编码器训练
4. zero-shot 评估
5. latency/FPS benchmark
6. ONNX 导出
```

本方案暂未直接包含：

```text
1. 大规模分布式训练
2. 多机多卡特征缓存
3. TensorRT INT8 校准
4. 医学专业 CLIP Teacher 替换
5. TinyViT/DeiT-Tiny 的完整实现
```

这些可以作为第二阶段扩展，不影响第一阶段跑通主流程。
