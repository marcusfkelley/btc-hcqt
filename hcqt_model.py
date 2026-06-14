#!/usr/bin/env python3
"""
HCQT front-ends + the HCQT_BTC wrapper.

The wrapper prepends a small learned module that collapses the (harmonics, 144)
HCQT stack into the 144-dim per-frame feature the UNMODIFIED pretrained BTC
expects, then runs BTC. Three front-end variants for the sweep:

  linear : per-bin learned weighted sum of the harmonic views (1x1 conv).
           Initialized to PASS THROUGH the h=1 channel, so at init the wrapper
           is byte-for-byte the original BTC (its mean/std normalization is
           valid) — training only learns how much of the OTHER harmonics to mix
           in. Lowest-risk.
  conv   : 2-layer conv over frequency (adds local overtone context).
  deep   : 3-layer conv (most capacity, highest risk of disturbing BTC).

The front-end output is normalized with BTC's own mean/std before BTC, so the
pretrained weights see their familiar input distribution.
"""
import torch
import torch.nn as nn

from hcqt import HARMONICS


def _passthrough_init(conv, n_harm, harm_index):
    """Init a Conv1d(n_harm -> 1, k=1) to select the h=1 channel exactly."""
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, harm_index, 0] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()


class LinearFrontEnd(nn.Module):
    def __init__(self, n_harm):
        super().__init__()
        self.conv = nn.Conv1d(n_harm, 1, kernel_size=1, bias=False)
        _passthrough_init(self.conv, n_harm, HARMONICS.index(1) if 1 in HARMONICS else 0)

    def forward(self, x):           # x (N, H, 144)
        return self.conv(x).squeeze(1)   # (N, 144)


class ConvFrontEnd(nn.Module):
    def __init__(self, n_harm, width=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_harm, width, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(width, 1, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class DeepFrontEnd(nn.Module):
    def __init__(self, n_harm, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_harm, width, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(width, width // 2, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(width // 2, 1, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class FDAA(nn.Module):
    """Frequency-Domain Adaptive Attention (BTC-FDAA-FGF). A learned per-(harmonic,
    frequency) gate that adaptively re-weights frequency components — emphasizing
    the salient bins. Passthrough-init (sigmoid bias +4 => gate ~0.98) so it starts
    as the identity and only learns the re-weighting."""
    def __init__(self, n_harm):
        super().__init__()
        self.g = nn.Sequential(nn.Conv1d(n_harm, 8, 1), nn.ReLU(), nn.Conv1d(8, n_harm, 1))
        nn.init.zeros_(self.g[-1].weight)
        nn.init.constant_(self.g[-1].bias, 4.0)

    def forward(self, x):                # x (N, H, F)
        return x * torch.sigmoid(self.g(x))


class FGF(nn.Module):
    """Fine-Grained aggregation Fourier module (BTC-FDAA-FGF). A real-FFT mixing
    over the (log-)frequency axis captures the periodic harmonic structure of
    chords. Residual with a learned scalar, 0-init => starts as passthrough."""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):                # x (N, F)
        return x + self.alpha * torch.fft.fft(x, dim=-1).real


class FDAAFGFFrontEnd(nn.Module):
    """Deep front-end wrapped with FDAA (before) + FGF (after) — our reimplementation
    of the BTC-FDAA-FGF recipe on top of the HCQT base. Both added modules are
    passthrough-init, so resuming btc_hcqt_deep's deep weights starts ~unchanged."""
    def __init__(self, n_harm, width=32):
        super().__init__()
        self.fdaa = FDAA(n_harm)
        self.deep = nn.Sequential(
            nn.Conv1d(n_harm, width, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(width, width // 2, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(width // 2, 1, kernel_size=5, padding=2),
        )
        self.fgf = FGF()

    def forward(self, x):                # x (N, H, F)
        x = self.fdaa(x)
        x = self.deep(x).squeeze(1)      # (N, F)
        return self.fgf(x)


FRONTENDS = {"linear": LinearFrontEnd, "conv": ConvFrontEnd, "deep": DeepFrontEnd,
             "fdaafgf": FDAAFGFFrontEnd}


class HCQT_BTC(nn.Module):
    def __init__(self, btc_model, mean, std, n_harm, frontend="linear"):
        super().__init__()
        self.frontend = FRONTENDS[frontend](n_harm)
        self.btc = btc_model
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

    def forward(self, hcqt, labels):     # hcqt (B, T, H, 144)
        B, T, H, F = hcqt.shape
        x = self.frontend(hcqt.reshape(B * T, H, F)).reshape(B, T, F)
        x = (x - self.mean) / self.std
        return self.btc(x, labels)

    def set_body_trainable(self, flag):
        for p in self.btc.parameters():
            p.requires_grad = flag
