import argparse
import sys
from pathlib import Path

from _path import add_project_root_to_path

add_project_root_to_path()

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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

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
        dynamo=False,
    )
    torch.onnx.export(
        text_model,
        dummy_tokens,
        str(out_dir / "student_text_encoder.onnx"),
        input_names=["tokens"],
        output_names=["text_feature"],
        dynamic_axes={"tokens": {0: "batch"}, "text_feature": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"Exported ONNX models to {out_dir}")


if __name__ == "__main__":
    main()
