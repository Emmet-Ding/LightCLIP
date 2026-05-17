import argparse
import time
from pathlib import Path

from _path import add_project_root_to_path

add_project_root_to_path()

import torch

from models.student_clip import StudentCLIP


def count_params(model):
    return sum(param.numel() for param in model.parameters())


def get_file_size_mb(path):
    return Path(path).stat().st_size / 1024 / 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/student_best.pt")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    image_size = args.image_size or cfg["data"]["image_size"]

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

    image = torch.randn(args.batch_size, 3, image_size, image_size, device=device)
    tokens = torch.randint(
        low=0,
        high=cfg["student"]["vocab_size"],
        size=(args.batch_size, cfg["student"]["context_length"]),
        device=device,
    )

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(image, tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.iters):
            _ = model(image, tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    avg_ms = elapsed * 1000 / max(1, args.iters)
    fps = args.batch_size * 1000 / avg_ms
    print(f"Device: {device}")
    print(f"Params: {count_params(model) / 1e6:.2f} M")
    print(f"Checkpoint size: {get_file_size_mb(args.ckpt):.2f} MB")
    print(f"Batch size: {args.batch_size}")
    print(f"Latency: {avg_ms:.3f} ms / batch")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    main()

