import argparse

import open_clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


class TeacherEvalDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_csv, preprocess):
        self.df = pd.read_csv(metadata_csv)
        if "label" not in self.df.columns:
            raise ValueError("metadata_csv must contain label column for zero-shot evaluation")
        self.image_paths = self.df["image_path"].astype(str).tolist()
        self.labels = self.df["label"].astype(str).tolist()
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.preprocess(image), self.labels[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, default="data/metadata.csv")
    parser.add_argument("--model_name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--prompt", type=str, default="a photo of a {}")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()

    df = pd.read_csv(args.metadata_csv)
    class_names = sorted(df["label"].astype(str).unique().tolist())
    prompts = [args.prompt.format(name.replace("_", " ")) for name in class_names]

    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    dataset = TeacherEvalDataset(args.metadata_csv, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval teacher zero-shot"):
            images = images.to(device, non_blocking=True)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            pred = (image_features @ text_features.t()).argmax(dim=1).cpu().tolist()
            target = [class_to_idx[label] for label in labels]
            correct += sum(int(p == y) for p, y in zip(pred, target))
            total += len(target)

    acc = correct / max(1, total)
    print(f"Teacher zero-shot acc: {acc:.4f}")
    print(f"Num classes: {len(class_names)}")
    print(f"Num samples: {total}")


if __name__ == "__main__":
    main()

