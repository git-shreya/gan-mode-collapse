"""
Train the baseline or the improved GAN.

    python -m src.train --model baseline --data data/fashion64.npz --out runs/baseline
    python -m src.train --model improved --data data/fashion64.npz --out runs/improved

Both runs use the same data, epochs, batch size, optimiser, learning rate and
seed. Only the discriminator normalisation and the loss change.

Colab free disconnects sometimes, so a checkpoint is saved every few epochs.
Re-running the same command with --resume continues from the last checkpoint.
"""
import argparse
import csv
import json
import os
import time

import torch
from torchvision.utils import save_image

from .data import load_npz, to_tensor_uint8
from .losses import LOSSES
from .models import NZ, build

FIXED_NOISE_SEED = 1234  # same 64 noise vectors for both models -> fair side-by-side grids


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fixed_noise(n=64, nz=NZ, device="cpu"):
    g = torch.Generator().manual_seed(FIXED_NOISE_SEED)
    return torch.randn(n, nz, generator=g).to(device)


@torch.no_grad()
def pixel_diversity(G, z):
    """Mean pairwise L2 distance between generated images. Drops sharply if G collapses."""
    G.eval()
    x = G(z).flatten(1)
    G.train()
    d = torch.cdist(x, x)
    n = x.shape[0]
    return (d.sum() / (n * (n - 1))).item()


def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)

    data = load_npz(args.data)
    real = to_tensor_uint8(data["images"][~data["is_test"]] if args.train_split_only else data["images"])
    real = real.to(device)  # ~35k x 3 x 64 x 64 uint8 = ~430 MB, fits easily on a T4
    n = real.shape[0]
    steps_per_epoch = n // args.batch
    print(f"device={device}  images={n}  steps/epoch={steps_per_epoch}  model={args.model}")

    G, D = build(args.model, NZ)
    G, D = G.to(device), D.to(device)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=tuple(args.betas))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=tuple(args.betas))
    loss_d_fn, loss_g_fn = LOSSES[args.model]
    z_fixed = fixed_noise(64, NZ, device)

    start_epoch = 0
    ckpt_path = os.path.join(args.out, "ckpt_last.pt")
    log_path = os.path.join(args.out, "log.csv")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        optG.load_state_dict(ck["optG"]); optD.load_state_dict(ck["optD"])
        start_epoch = ck["epoch"] + 1
        torch.set_rng_state(ck["rng_cpu"])
        if device == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ck["rng_cuda"])
        print(f"Resumed from epoch {start_epoch}")
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss_d", "loss_g", "d_real", "d_fake",
                                    "loss_d_std", "loss_g_std", "pixel_diversity", "sec"])
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        perm = torch.randperm(n, device=device)
        ld_hist, lg_hist, dr_hist, df_hist = [], [], [], []
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch:(i + 1) * args.batch]
            x = real[idx].float().div(127.5).sub(1.0)
            if args.flip:
                flip = torch.rand(x.shape[0], device=device) < 0.5
                x[flip] = x[flip].flip(3)

            # ---- discriminator step ----
            z = torch.randn(x.shape[0], NZ, device=device)
            fake = G(z)
            real_logits = D(x)
            fake_logits = D(fake.detach())
            loss_d = loss_d_fn(real_logits, fake_logits)
            optD.zero_grad(set_to_none=True)
            loss_d.backward()
            optD.step()

            # ---- generator step ----
            fake_logits = D(fake)
            loss_g = loss_g_fn(fake_logits)
            optG.zero_grad(set_to_none=True)
            loss_g.backward()
            optG.step()

            ld_hist.append(loss_d.item()); lg_hist.append(loss_g.item())
            dr_hist.append(real_logits.mean().item()); df_hist.append(fake_logits.mean().item())

        ld, lg = torch.tensor(ld_hist), torch.tensor(lg_hist)
        div = pixel_diversity(G, z_fixed)
        sec = time.time() - t0
        row = [epoch, ld.mean().item(), lg.mean().item(), sum(dr_hist) / len(dr_hist),
               sum(df_hist) / len(df_hist), ld.std().item(), lg.std().item(), div, sec]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{v:.5f}" if isinstance(v, float) else v for v in row])
        print(f"[{args.model}] epoch {epoch + 1}/{args.epochs}  loss_D={row[1]:.3f}  loss_G={row[2]:.3f}  "
              f"diversity={div:.2f}  ({sec:.0f}s)")

        with torch.no_grad():
            G.eval()
            save_image(G(z_fixed), os.path.join(args.out, "samples", f"epoch_{epoch + 1:03d}.png"),
                       nrow=8, normalize=True, value_range=(-1, 1))
            G.train()

        if (epoch + 1) % args.ckpt_every == 0 or epoch + 1 == args.epochs:
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "optG": optG.state_dict(),
                        "optD": optD.state_dict(), "epoch": epoch, "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state() if device == "cuda" else None},
                       ckpt_path)

    final = os.path.join(args.out, "G_final.pt")
    torch.save({"G": G.state_dict(), "nz": NZ, "model": args.model, "epochs": args.epochs}, final)
    print(f"Saved generator to {final}")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "improved"], required=True)
    p.add_argument("--data", default="data/fashion64.npz")
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--betas", type=float, nargs=2, default=[0.5, 0.999])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt_every", type=int, default=2)
    p.add_argument("--flip", action="store_true", default=True)
    p.add_argument("--train_split_only", action="store_true",
                   help="exclude the classifier test split from GAN training")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    a.out = a.out or os.path.join("runs", a.model)
    return a


if __name__ == "__main__":
    train(parse())
