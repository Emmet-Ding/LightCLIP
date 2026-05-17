import argparse
import csv
import random
from pathlib import Path

from _path import add_project_root_to_path

add_project_root_to_path()

import numpy as np
import open_clip
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets.image_text_dataset import ImageTextDataset
from losses.distill_losses import (
    clip_contrastive_loss,
    cosine_distill_loss,
    similarity_distill_loss,
)
from models.student_clip import StudentCLIP


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(param.numel() for param in model.parameters())


def make_loader(dataset, cfg, shuffle: bool, drop_last: bool):
    pin_memory = bool(cfg["train"].get("pin_memory", torch.cuda.is_available()))
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def run_epoch(model, loader, teacher_cache, optimizer, scaler, cfg, device, train: bool):
    model.train(train)
    totals = {"loss": 0.0, "img": 0.0, "txt": 0.0, "sim": 0.0, "clip": 0.0}
    steps = 0
    grad_accum = max(1, int(cfg["train"].get("grad_accum_steps", 1)))

    if train:
        optimizer.zero_grad(set_to_none=True)

    desc = "train" if train else "val"
    autocast_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    disable_tqdm = bool(cfg["train"].get("disable_tqdm", False))
    for step, batch in enumerate(tqdm(loader, desc=desc, leave=False, disable=disable_tqdm), start=1):
        images = batch["image"].to(device, non_blocking=True)
        text_tokens = batch["text_tokens"].to(device, non_blocking=True)
        index = batch["index"].long()
        teacher_img = teacher_cache["image_features"][index].to(device, non_blocking=True)
        teacher_txt = teacher_cache["text_features"][index].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                student_img, student_txt, student_logits = model(images, text_tokens)
                teacher_logits = teacher_img @ teacher_txt.t()

                loss_img = cosine_distill_loss(student_img, teacher_img)
                loss_txt = cosine_distill_loss(student_txt, teacher_txt)
                loss_sim = similarity_distill_loss(
                    student_logits,
                    teacher_logits,
                    temperature=cfg["loss"]["kd_temperature"],
                )
                loss_clip = clip_contrastive_loss(student_logits)
                loss = (
                    cfg["loss"]["lambda_img"] * loss_img
                    + cfg["loss"]["lambda_txt"] * loss_txt
                    + cfg["loss"]["lambda_sim"] * loss_sim
                    + cfg["loss"]["lambda_clip"] * loss_clip
                )

            if train:
                scaler.scale(loss / grad_accum).backward()
                if step % grad_accum == 0 or step == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = images.size(0)
        totals["loss"] += float(loss.detach()) * batch_size
        totals["img"] += float(loss_img.detach()) * batch_size
        totals["txt"] += float(loss_txt.detach()) * batch_size
        totals["sim"] += float(loss_sim.detach()) * batch_size
        totals["clip"] += float(loss_clip.detach()) * batch_size
        steps += batch_size

    return {key: value / max(1, steps) for key, value in totals.items()}


def save_checkpoint(path, model, optimizer, epoch, cfg, metric):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg,
            "metric": metric,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/distill_resnet18.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    set_seed(cfg["seed"])
    if "cudnn_enabled" in cfg.get("train", {}):
        torch.backends.cudnn.enabled = bool(cfg["train"]["cudnn_enabled"])

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device(args.device)
    tokenizer = open_clip.get_tokenizer(cfg["teacher"]["model_name"])

    dataset = ImageTextDataset(
        metadata_csv=cfg["data"]["metadata_csv"],
        tokenizer=tokenizer,
        image_size=cfg["data"]["image_size"],
        train=True,
    )
    if len(dataset) < 2:
        raise ValueError("Need at least 2 image-text pairs for contrastive training.")

    val_ratio = float(cfg["train"].get("val_ratio", 0.1))
    n_val = max(1, int(round(len(dataset) * val_ratio)))
    n_train = len(dataset) - n_val
    if n_train < 1:
        n_train, n_val = len(dataset), 0

    if n_val:
        train_set, val_set = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg["seed"]),
        )
    else:
        train_set, val_set = dataset, None

    drop_last = bool(cfg["train"].get("drop_last", True)) and len(train_set) >= cfg["train"]["batch_size"]
    train_loader = make_loader(train_set, cfg, shuffle=True, drop_last=drop_last)
    val_loader = make_loader(val_set, cfg, shuffle=False, drop_last=False) if val_set else None

    teacher_cache = torch.load(cfg["data"]["teacher_cache"], map_location="cpu")
    if len(teacher_cache["image_features"]) != len(dataset):
        raise ValueError("Teacher cache size does not match metadata rows.")

    student = StudentCLIP(
        image_encoder=cfg["student"]["image_encoder"],
        embed_dim=cfg["student"]["embed_dim"],
        vocab_size=cfg["student"]["vocab_size"],
        context_length=cfg["student"]["context_length"],
        text_width=cfg["student"]["text_width"],
        text_layers=cfg["student"]["text_layers"],
        text_heads=cfg["student"]["text_heads"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")

    ckpt_dir = Path(cfg["output"]["ckpt_dir"])
    log_dir = Path(cfg["output"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.csv"

    best_val = float("inf")
    start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint(args.resume, student, optimizer, device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_ckpt = ckpt_dir / "student_best.pt"
        if best_ckpt.exists():
            best_checkpoint = torch.load(best_ckpt, map_location="cpu")
            best_val = float(best_checkpoint.get("metric", checkpoint.get("metric", best_val)))
        else:
            best_val = float(checkpoint.get("metric", best_val))

    print(f"Device: {device}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"Student params: {count_params(student) / 1e6:.2f} M")
    if args.resume:
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    append_log = bool(args.resume) and log_path.exists()
    with open(log_path, "a" if append_log else "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "split",
                "loss",
                "img",
                "txt",
                "sim",
                "clip",
            ],
        )
        if not append_log:
            writer.writeheader()

        total_epochs = int(args.epochs or cfg["train"]["epochs"])
        for epoch in range(start_epoch, total_epochs + 1):
            train_metrics = run_epoch(student, train_loader, teacher_cache, optimizer, scaler, cfg, device, train=True)
            writer.writerow({"epoch": epoch, "split": "train", **train_metrics})

            if val_loader:
                with torch.no_grad():
                    val_metrics = run_epoch(student, val_loader, teacher_cache, optimizer, scaler, cfg, device, train=False)
                writer.writerow({"epoch": epoch, "split": "val", **val_metrics})
                monitor = val_metrics["loss"]
            else:
                val_metrics = None
                monitor = train_metrics["loss"]

            print(
                f"epoch {epoch:03d} train_loss={train_metrics['loss']:.4f}"
                + (f" val_loss={val_metrics['loss']:.4f}" if val_metrics else "")
            )

            save_checkpoint(ckpt_dir / "student_last.pt", student, optimizer, epoch, cfg, monitor)
            if monitor < best_val:
                best_val = monitor
                save_checkpoint(ckpt_dir / "student_best.pt", student, optimizer, epoch, cfg, monitor)
            file.flush()

    print(f"Saved checkpoints to {ckpt_dir}")
    print(f"Saved log to {log_path}")


if __name__ == "__main__":
    main()
