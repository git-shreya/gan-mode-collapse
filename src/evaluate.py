"""
Benchmark the baseline and improved generators on the same footing.

    python -m src.evaluate --data data/fashion64.npz \
        --baseline runs/baseline/G_final.pt --improved runs/improved/G_final.pt \
        --classifier checkpoints/classifier.pt --n 10000

Metrics (all computed on the same number of generated images per model):
  FID                - quality + diversity together (Heusel et al., 2017). Lower is better.
  Precision          - share of generated images that look like real data (quality).
  Recall             - share of the real data distribution the generator covers (diversity).
                       Improved precision/recall, k=3 (Kynkaanniemi et al., 2019).
  Categories covered - of the 20 product categories, how many the generator produces at
                       a reasonable rate (>= half their real frequency). Mode collapse shows
                       up here as missing categories.
  KL(real || gen)    - distance between real and generated category distributions.
  Inference latency  - ms per image for the generator (batch 1 and batch 64).

Outputs go to results/: metrics.json, metrics.csv and figures.
"""
import argparse
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import linalg
from torchvision.utils import make_grid, save_image

from .classifier import CategoryClassifier
from .data import load_npz, to_tensor_uint8
from .models import load_generator
from .train import fixed_noise


# ----------------------------------------------------------------- features
class Inception:
    def __init__(self, device):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        self.net = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, x_uint8, batch=200):
        out = []
        for i in range(0, x_uint8.shape[0], batch):
            out.append(self.net(x_uint8[i:i + batch].to(self.device))[0].cpu())
        return torch.cat(out).double().numpy()


def to_uint8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


@torch.no_grad()
def generate(G, n, device, batch=500, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    imgs = []
    for i in range(0, n, batch):
        z = torch.randn(min(batch, n - i), G.nz, generator=g).to(device)
        imgs.append(G(z).cpu())
    return torch.cat(imgs)


# ----------------------------------------------------------------- metrics
def fid_from_feats(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean = linalg.sqrtm(s1.dot(s2))
    if isinstance(covmean, tuple):  # older SciPy versions return (sqrtm, error)
        covmean = covmean[0]
    if not np.isfinite(covmean).all():
        eps = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + eps).dot(s2 + eps))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def _kth_radii(feats, k, device, chunk=2000):
    f = torch.from_numpy(feats).float().to(device)
    radii = []
    for i in range(0, f.shape[0], chunk):
        d = torch.cdist(f[i:i + chunk], f)
        radii.append(d.kthvalue(k + 1, dim=1).values)  # +1 skips the point itself
    return f, torch.cat(radii)


def _inside(query, ref, ref_radii, chunk=2000):
    hits = []
    for i in range(0, query.shape[0], chunk):
        d = torch.cdist(query[i:i + chunk], ref)
        hits.append((d <= ref_radii[None, :]).any(1))
    return torch.cat(hits).float().mean().item()


def precision_recall(real_f, fake_f, k=3, device="cpu"):
    r, r_rad = _kth_radii(real_f, k, device)
    f, f_rad = _kth_radii(fake_f, k, device)
    precision = _inside(f, r, r_rad)  # fakes that fall on the real manifold
    recall = _inside(r, f, f_rad)     # reals that fall on the fake manifold
    return precision, recall


def category_stats(pred, real_labels, n_classes):
    real_p = np.bincount(real_labels, minlength=n_classes) / len(real_labels)
    gen_p = np.bincount(pred, minlength=n_classes) / len(pred)
    covered = int((gen_p >= 0.5 * real_p).sum())
    gen_smooth = (np.bincount(pred, minlength=n_classes) + 1) / (len(pred) + n_classes)  # Laplace smoothing
    kl = float(np.sum(real_p * np.log(real_p / gen_smooth)))
    tvd = float(0.5 * np.abs(real_p - gen_p).sum())
    return real_p, gen_p, covered, kl, tvd


@torch.no_grad()
def latency(G, device, batch, reps=50):
    z = torch.randn(batch, G.nz, device=device)
    for _ in range(5):
        G(z)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        G(z)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps / batch * 1000.0  # ms per image


# ----------------------------------------------------------------- main
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    d = load_npz(args.data)
    names = d["class_names"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d["labels"]), size=min(args.n, len(d["labels"])), replace=False)
    real_u8 = to_tensor_uint8(d["images"][idx])
    inc = Inception(device)
    print("Extracting Inception features for real images ...")
    real_f = inc(real_u8)
    clf = CategoryClassifier(args.classifier, device)

    models = {"baseline": args.baseline, "improved": args.improved}
    results, gen_dists, fakes_f, fakes = {}, {}, {}, {}
    for name, path in models.items():
        print(f"\n== {name} ==")
        G = load_generator(path, device)
        x = generate(G, len(idx), device)
        fakes[name] = x
        f = inc(to_uint8(x))
        fakes_f[name] = f
        fid = fid_from_feats(real_f, f)
        prec, rec = precision_recall(real_f, f, k=3, device=device)
        pred, conf = clf.predict(x)
        real_p, gen_p, covered, kl, tvd = category_stats(pred.numpy(), d["labels"], len(names))
        gen_dists[name] = gen_p
        res = {
            "FID": round(fid, 2), "precision": round(prec, 4), "recall": round(rec, 4),
            "categories_covered": f"{covered}/{len(names)}", "categories_covered_n": covered,
            "KL_real_gen": round(kl, 4), "TVD_real_gen": round(tvd, 4),
            "mean_classifier_confidence": round(conf.mean().item(), 4),
            f"latency_ms_per_img_bs1_{device}": round(latency(G, device, 1), 3),
            f"latency_ms_per_img_bs64_{device}": round(latency(G, device, 64), 3),
            "latency_ms_per_img_bs1_cpu": round(latency(G.cpu(), "cpu", 1, reps=20), 3),
            "generator_params_M": round(sum(p.numel() for p in G.parameters()) / 1e6, 2),
        }
        results[name] = res
        for k, v in res.items():
            print(f"  {k:<32s} {v}")

    with open(os.path.join(args.out, "metrics.json"), "w") as fjs:
        json.dump({"n_images": int(len(idx)), "classifier_test_acc": clf.test_acc, **results}, fjs, indent=2)
    pd.DataFrame(results).T.to_csv(os.path.join(args.out, "metrics.csv"))

    # ---- figures ----
    real_p = np.bincount(d["labels"], minlength=len(names)) / len(d["labels"])
    order = np.argsort(-real_p)
    xs = np.arange(len(names)); w = 0.28
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(xs - w, real_p[order], w, label="Real data", color="#9aa5b1")
    ax.bar(xs, gen_dists["baseline"][order], w, label="Baseline DCGAN", color="#e07a5f")
    ax.bar(xs + w, gen_dists["improved"][order], w, label="Improved (SN + hinge)", color="#3d7ea6")
    ax.set_xticks(xs); ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right")
    ax.set_ylabel("Share of images"); ax.set_title("Product categories: real vs generated")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "category_distribution.png"), dpi=160)
    plt.close(fig)

    # side-by-side grid from the same fixed noise
    z = fixed_noise(32, device=device)
    grids = []
    for name, path in models.items():
        G = load_generator(path, device)
        with torch.no_grad():
            grids.append(make_grid(G(z).cpu(), nrow=8, normalize=True, value_range=(-1, 1), padding=2))
    sep = torch.ones(3, grids[0].shape[1], 12)
    save_image(torch.cat([grids[0], sep, grids[1]], 2), os.path.join(args.out, "side_by_side.png"))

    # nearest real neighbour check (is the improved GAN just copying training images?)
    f_imp = torch.from_numpy(fakes_f["improved"][:8]).float()
    dist = torch.cdist(f_imp, torch.from_numpy(real_f).float())
    nn_idx = dist.argmin(1).numpy()
    gen8 = (fakes["improved"][:8].clamp(-1, 1) + 1) / 2
    real8 = real_u8[nn_idx].float() / 255.0
    save_image(torch.cat([gen8, real8]), os.path.join(args.out, "nearest_neighbours.png"), nrow=8, padding=2)

    plot_training_curves(args.runs, args.out)
    print(f"\nSaved metrics and figures to {args.out}/")


def plot_training_curves(runs_dir, out):
    logs = {}
    for name in ("baseline", "improved"):
        p = os.path.join(runs_dir, name, "log.csv")
        if os.path.exists(p):
            logs[name] = pd.read_csv(p)
    if not logs:
        return
    colours = {"baseline": "#e07a5f", "improved": "#3d7ea6"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, df in logs.items():
        ep = df["epoch"] + 1
        axes[0].plot(ep, df["loss_d"], label=f"{name} D", color=colours[name])
        axes[0].plot(ep, df["loss_g"], "--", label=f"{name} G", color=colours[name])
        axes[1].plot(ep, df["loss_g_std"], label=name, color=colours[name])
        axes[2].plot(ep, df["pixel_diversity"], label=name, color=colours[name])
    axes[0].set_title("Losses per epoch (different loss scales)")
    axes[1].set_title("Generator loss std within epoch (instability)")
    axes[2].set_title("Sample diversity on fixed noise (higher = less collapse)")
    for a in axes:
        a.set_xlabel("epoch"); a.legend(frameon=False); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "training_curves.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/fashion64.npz")
    p.add_argument("--baseline", default="runs/baseline/G_final.pt")
    p.add_argument("--improved", default="runs/improved/G_final.pt")
    p.add_argument("--classifier", default="checkpoints/classifier.pt")
    p.add_argument("--runs", default="runs")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--out", default="results")
    main(p.parse_args())
