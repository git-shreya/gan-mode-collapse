"""
Dataset preparation for the Myntra Fashion Product Images (small) dataset.

Source: Hugging Face mirror `ashraq/fashion-product-images-small`
(44,072 product photos from Myntra, originally published on Kaggle by
Param Aggarwal). Images are about 60x80 px on a white background.

What this script does:
  1. Keeps the `top_k` most frequent article types (e.g. Tshirts, Kurtas,
     Watches, Casual Shoes ...). These categories are the "modes" we use
     to measure mode collapse later.
  2. Pads every image to a square with white, resizes to 64x64.
  3. Saves everything to one compressed .npz file so training in Colab
     does not have to decode JPEGs every epoch.

Usage:
    python -m src.data --out data/fashion64.npz --top_k 20
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

HF_DATASET = "ashraq/fashion-product-images-small"


def pad_and_resize(img: Image.Image, size: int = 64) -> np.ndarray:
    """White-pad to square (product photos have white backgrounds), then resize."""
    img = img.convert("RGB")
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    canvas = canvas.resize((size, size), Image.LANCZOS)
    return np.asarray(canvas, dtype=np.uint8)


def prepare(out_path: str, top_k: int = 20, size: int = 64, test_frac: float = 0.1, seed: int = 0):
    from datasets import load_dataset  # imported here so the rest of the repo works without it

    print(f"Loading {HF_DATASET} ...")
    ds = load_dataset(HF_DATASET, split="train")

    # Count categories without decoding images
    types = ds["articleType"]
    names, counts = np.unique(np.array(types, dtype=object).astype(str), return_counts=True)
    order = np.argsort(-counts)
    keep = [str(names[i]) for i in order[:top_k]]
    keep_counts = {str(names[i]): int(counts[i]) for i in order[:top_k]}
    print("Keeping categories:")
    for k in keep:
        print(f"  {k:<20s} {keep_counts[k]}")

    cls_to_idx = {c: i for i, c in enumerate(keep)}
    keep_set = set(keep)
    ds = ds.filter(lambda t: t in keep_set, input_columns=["articleType"])

    images, labels, ids = [], [], []
    for i, row in enumerate(ds):
        try:
            images.append(pad_and_resize(row["image"], size))
        except Exception as e:  # a handful of files can be corrupt
            print(f"  skipping id={row['id']}: {e}")
            continue
        labels.append(cls_to_idx[row["articleType"]])
        ids.append(row["id"])
        if (i + 1) % 5000 == 0:
            print(f"  processed {i + 1}/{len(ds)}")

    images = np.stack(images)
    labels = np.array(labels, dtype=np.int64)
    ids = np.array(ids, dtype=np.int64)

    # Stratified train / test split (test split is only used for the category classifier)
    rng = np.random.default_rng(seed)
    is_test = np.zeros(len(labels), dtype=bool)
    for c in range(len(keep)):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        is_test[idx[: max(1, int(len(idx) * test_frac))]] = True

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, images=images, labels=labels, ids=ids,
                        is_test=is_test, class_names=np.array(keep))
    meta = {"source": HF_DATASET, "n_images": int(len(labels)), "size": size,
            "classes": keep, "counts": {k: int((labels == i).sum()) for i, k in enumerate(keep)}}
    with open(os.path.splitext(out_path)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {len(labels)} images to {out_path}")


def load_npz(path: str):
    """Returns dict with images (N,H,W,3 uint8), labels, is_test, class_names (list)."""
    d = np.load(path, allow_pickle=False)
    return {"images": d["images"], "labels": d["labels"], "is_test": d["is_test"],
            "class_names": [str(c) for c in d["class_names"]]}


def to_tensor_uint8(images: np.ndarray):
    """(N,H,W,3) uint8 numpy -> (N,3,H,W) uint8 torch tensor."""
    import torch
    return torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()


def make_dummy(out_path: str, n: int = 512, n_classes: int = 5, size: int = 64, seed: int = 0):
    """Tiny synthetic dataset for smoke-testing the pipeline without internet."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, n)
    images = np.full((n, size, size, 3), 255, dtype=np.uint8)
    for i, c in enumerate(labels):
        colour = np.array([(c * 50) % 256, (c * 90) % 256, (c * 130) % 256], dtype=np.uint8)
        s = 10 + 4 * c
        images[i, s:size - s, s:size - s] = colour
    is_test = rng.random(n) < 0.1
    np.savez_compressed(out_path, images=images, labels=labels, ids=np.arange(n), is_test=is_test,
                        class_names=np.array([f"class_{c}" for c in range(n_classes)]))
    print(f"Dummy dataset saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/fashion64.npz")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--dummy", action="store_true", help="create a tiny synthetic dataset for testing")
    a = p.parse_args()
    if a.dummy:
        make_dummy(a.out, size=a.size)
    else:
        prepare(a.out, a.top_k, a.size)
