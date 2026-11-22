"""
Adversarial losses. D outputs raw scores (logits), not probabilities.

Baseline (Goodfellow et al., 2014, non-saturating form):
    L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
    L_G = -E[log D(G(z))]

Improved, hinge loss (Lim & Ye, 2017; used with spectral norm by Miyato et al., 2018):
    L_D = E[max(0, 1 - D(x))] + E[max(0, 1 + D(G(z)))]
    L_G = -E[D(G(z))]
"""
import torch
import torch.nn.functional as F


def bce_d(real_logits, fake_logits):
    return (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) +
            F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))


def bce_g(fake_logits):
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def hinge_d(real_logits, fake_logits):
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def hinge_g(fake_logits):
    return -fake_logits.mean()


LOSSES = {"baseline": (bce_d, bce_g), "improved": (hinge_d, hinge_g)}
