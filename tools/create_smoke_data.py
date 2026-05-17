import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke_resnet18.yaml")
    parser.add_argument("--samples_per_class", type=int, default=3)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    metadata_csv = Path(cfg["data"]["metadata_csv"])
    image_root = metadata_csv.parent / "images"
    classes = {
        "red_square": (220, 50, 50),
        "green_square": (40, 180, 80),
        "blue_square": (60, 90, 220),
    }

    rows = []
    for label, color in classes.items():
        class_dir = image_root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(args.samples_per_class):
            image = Image.new("RGB", (96, 96), color=(245, 245, 245))
            draw = ImageDraw.Draw(image)
            margin = 12 + idx * 2
            draw.rectangle([margin, margin, 96 - margin, 96 - margin], fill=color)
            path = class_dir / f"{idx:03d}.png"
            image.save(path)
            rows.append(
                {
                    "image_path": path.as_posix(),
                    "text": f"a photo of a {label.replace('_', ' ')}",
                    "label": label,
                }
            )

    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metadata_csv, index=False, encoding="utf-8")

    embed_dim = int(cfg["student"]["embed_dim"])
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    image_features = torch.randn(len(rows), embed_dim, generator=generator)
    text_features = torch.randn(len(rows), embed_dim, generator=generator)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    cache_path = Path(cfg["data"]["teacher_cache"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_features": image_features,
            "text_features": text_features,
            "image_paths": [row["image_path"] for row in rows],
            "texts": [row["text"] for row in rows],
            "model_name": "synthetic",
            "pretrained": "none",
        },
        cache_path,
    )

    print(f"Saved smoke metadata to {metadata_csv}")
    print(f"Saved smoke teacher cache to {cache_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()

