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
    if not image_root.exists():
        raise FileNotFoundError(f"image_root does not exist: {image_root}")

    rows = []
    for class_dir in sorted(image_root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in IMG_EXTS:
                rows.append(
                    {
                        "image_path": str(path.as_posix()),
                        "text": args.prompt.format(label.replace("_", " ")),
                        "label": label,
                    }
                )

    if not rows:
        raise RuntimeError(f"No images found under {image_root}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()

