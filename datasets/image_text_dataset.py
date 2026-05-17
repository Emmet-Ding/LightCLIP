from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def build_image_transform(image_size: int = 224, train: bool = True):
    augmentations = []
    if train:
        augmentations.append(transforms.RandomHorizontalFlip(p=0.5))

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            *augmentations,
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


class ImageTextDataset(Dataset):
    def __init__(self, metadata_csv, tokenizer, image_size: int = 224, train: bool = True):
        self.metadata_csv = Path(metadata_csv)
        self.df = pd.read_csv(self.metadata_csv)

        required = {"image_path", "text"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"metadata_csv missing columns: {sorted(missing)}")

        self.image_paths = self.df["image_path"].astype(str).tolist()
        self.texts = self.df["text"].astype(str).tolist()
        self.labels = (
            self.df["label"].astype(str).tolist()
            if "label" in self.df.columns
            else [""] * len(self.df)
        )
        self.transform = build_image_transform(image_size=image_size, train=train)
        self.tokens = tokenizer(self.texts)

    def __len__(self):
        return len(self.image_paths)

    def _resolve_image_path(self, path: str) -> Path:
        image_path = Path(path)
        if image_path.is_absolute():
            return image_path
        return image_path

    def __getitem__(self, idx):
        path = self._resolve_image_path(self.image_paths[idx])
        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "text_tokens": self.tokens[idx].long(),
            "text": self.texts[idx],
            "label": self.labels[idx],
            "index": torch.tensor(idx, dtype=torch.long),
        }

