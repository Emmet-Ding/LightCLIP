import argparse
from pathlib import Path

import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, default="data/metadata.csv")
    parser.add_argument("--out", type=str, default="data/teacher_cache/teacher_features.pt")
    parser.add_argument("--model_name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.metadata_csv)
    required = {"image_path", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata_csv missing columns: {sorted(missing)}")

    image_paths = df["image_path"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()

    pretrained = args.weights_path if args.weights_path else args.pretrained
    model_kwargs = (
        {"force_quick_gelu": True, "weights_only": False}
        if args.weights_path
        else {}
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=pretrained,
        device=device,
        **model_kwargs,
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    image_features = []
    text_features = []

    with torch.no_grad():
        for start in tqdm(range(0, len(df), args.batch_size), desc="Precompute teacher"):
            end = min(start + args.batch_size, len(df))
            batch_paths = image_paths[start:end]
            batch_texts = texts[start:end]

            images = [preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
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
    torch.save(
        {
            "image_features": image_features,
            "text_features": text_features,
            "image_paths": image_paths,
            "texts": texts,
            "model_name": args.model_name,
            "pretrained": pretrained,
        },
        out,
    )

    print("Saved teacher features:")
    print("  path:", out)
    print("  image_features:", tuple(image_features.shape))
    print("  text_features:", tuple(text_features.shape))


if __name__ == "__main__":
    main()
