"""
Product-category classifier (ResNet-18, ImageNet-pretrained, fine-tuned).

It is NOT part of the GAN. It is used for two things:
  1. Evaluation: label every generated image with a category, so we can count
     how many of the 20 product categories each GAN actually produces
     (a direct, readable measure of mode collapse).
  2. The demo app: the GAN is unconditional, so to show "only kurtas" the app
     generates a batch and keeps the images the classifier is confident about.

    python -m src.classifier --data data/fashion64.npz --out checkpoints/classifier.pt
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .data import load_npz, to_tensor_uint8

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
INPUT_SIZE = 96


def build_model(n_classes, pretrained=True):
    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    m = torchvision.models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def preprocess(x, from_range="uint8"):
    """x: (N,3,64,64) uint8 or float in [-1,1] -> normalised, resized float tensor."""
    if from_range == "uint8":
        x = x.float() / 255.0
    else:  # generator output in [-1, 1]
        x = (x.clamp(-1, 1) + 1) / 2
    x = F.interpolate(x, size=INPUT_SIZE, mode="bilinear", align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class CategoryClassifier:
    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.class_names = ck["class_names"]
        self.test_acc = ck.get("test_acc")
        self.model = build_model(len(self.class_names), pretrained=False).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict(self, gen_images, batch=256):
        """gen_images: float tensor in [-1,1]. Returns (pred_idx, confidence)."""
        preds, confs = [], []
        for i in range(0, gen_images.shape[0], batch):
            x = preprocess(gen_images[i:i + batch].to(self.device), from_range="tanh")
            p = self.model(x).softmax(1)
            c, k = p.max(1)
            preds.append(k.cpu()); confs.append(c.cpu())
        return torch.cat(preds), torch.cat(confs)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    d = load_npz(args.data)
    X = to_tensor_uint8(d["images"]); y = torch.from_numpy(d["labels"])
    te = torch.from_numpy(d["is_test"])
    Xtr, ytr, Xte, yte = X[~te], y[~te], X[te], y[te]
    names = d["class_names"]
    model = build_model(len(names), pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ytr))
        total, correct = 0, 0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb = Xtr[idx].to(device)
            if True:  # light augmentation: horizontal flip
                flip = torch.rand(xb.shape[0], device=device) < 0.5
                xb[flip] = xb[flip].flip(3)
            xb = preprocess(xb); yb = ytr[idx].to(device)
            out = model(xb)
            loss = F.cross_entropy(out, yb, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (out.argmax(1) == yb).sum().item(); total += len(yb)
        sched.step()
        acc_te = evaluate(model, Xte, yte, device)
        print(f"epoch {ep + 1}/{args.epochs}  train_acc={correct / total:.3f}  test_acc={acc_te:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "class_names": names, "test_acc": acc_te}, args.out)
    print(f"Saved classifier to {args.out} (test accuracy {acc_te:.3f})")


@torch.no_grad()
def evaluate(model, X, y, device, batch=512):
    model.eval()
    correct = 0
    for i in range(0, len(y), batch):
        out = model(preprocess(X[i:i + batch].to(device)))
        correct += (out.argmax(1).cpu() == y[i:i + batch]).sum().item()
    return correct / max(1, len(y))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/fashion64.npz")
    p.add_argument("--out", default="checkpoints/classifier.pt")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--no_pretrained", action="store_true")
    train(p.parse_args())
