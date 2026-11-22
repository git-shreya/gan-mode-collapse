"""
Generator and discriminator for 64x64 images.

Both GANs in this project share the SAME generator. The only differences
between the baseline and the improved model are in the discriminator and
the loss, so any change in results comes from our modification alone:

  baseline : DCGAN discriminator with BatchNorm + binary cross-entropy loss
             (Radford et al., 2016; non-saturating loss from Goodfellow et al., 2014)
  improved : same discriminator layers, BatchNorm replaced by spectral
             normalization on every layer + hinge loss (Miyato et al., 2018)

Because spectral normalization only touches the discriminator, the
generator used at inference time is identical, so the fix costs nothing
in generation speed.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

NZ = 100  # latent size


class Generator(nn.Module):
    def __init__(self, nz: int = NZ, ngf: int = 64, nc: int = 3):
        super().__init__()
        self.nz = nz
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),       # 4x4
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),      # 32x32
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),           # 64x64
            nn.Tanh(),
        )

    def forward(self, z):
        if z.dim() == 2:
            z = z[:, :, None, None]
        return self.net(z)


class Discriminator(nn.Module):
    """use_sn=False -> DCGAN discriminator (BatchNorm). use_sn=True -> spectral norm, no BatchNorm."""

    def __init__(self, ndf: int = 64, nc: int = 3, use_sn: bool = False):
        super().__init__()
        sn = spectral_norm if use_sn else (lambda m: m)
        norm = (lambda c: nn.Identity()) if use_sn else (lambda c: nn.BatchNorm2d(c))
        self.net = nn.Sequential(
            sn(nn.Conv2d(nc, ndf, 4, 2, 1, bias=use_sn)),            # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=use_sn)),       # 16x16
            norm(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=use_sn)),   # 8x8
            norm(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=use_sn)),   # 4x4
            norm(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=use_sn)),         # 1x1 -> logit
        )

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    """DCGAN initialisation: N(0, 0.02) for convs, N(1, 0.02) for BatchNorm scale."""
    name = m.__class__.__name__
    if "Conv" in name:
        w = m.weight_orig if hasattr(m, "weight_orig") else m.weight
        nn.init.normal_(w, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def build(model_type: str, nz: int = NZ):
    assert model_type in ("baseline", "improved")
    G = Generator(nz)
    D = Discriminator(use_sn=(model_type == "improved"))
    G.apply(weights_init)
    D.apply(weights_init)
    return G, D


def load_generator(path: str, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    nz = ckpt.get("nz", NZ)
    G = Generator(nz).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    return G
