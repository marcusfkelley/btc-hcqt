#!/usr/bin/env python3
"""
Harmonic CQT (HCQT) feature — the representation fix for real-audio 7th
discrimination. Instead of one CQT, stack several CQTs each tuned to a
harmonic (0.5x..5x the fundamental) so a note's overtones line up across the
harmonic axis. Makes overtone structure explicit → the maj7-vs-maj distinction
becomes a learnable pattern instead of energy buried under other notes'
overtones.

Shape: (n_harmonics, n_bins, frames), log-magnitude, float32.
The h=1 layer is exactly BTC's standard CQT, so the pretrained model still
sees its familiar input in channel index HARMONICS.index(1).

Matches BTC large-voca feature config (sr 22050, hop 2048, 144 bins @ 24/oct,
fmin = C1) so the harmonic-1 channel is identical to what BTC trained on.
"""
import numpy as np
import librosa

SR = 22050
HOP = 2048
N_BINS = 144
BINS_PER_OCTAVE = 24
HARMONICS = [0.5, 1, 2, 3, 4, 5]   # default 6-harmonic stack
FMIN = librosa.note_to_hz("C1")    # BTC default CQT fmin


def hcqt(y, sr=SR, harmonics=HARMONICS, n_bins=N_BINS,
         bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP):
    """Return (n_harmonics, n_bins, frames) log-magnitude HCQT."""
    layers = []
    for h in harmonics:
        c = librosa.cqt(y, sr=sr, fmin=FMIN * h, n_bins=n_bins,
                        bins_per_octave=bins_per_octave, hop_length=hop_length)
        layers.append(np.log(np.abs(c) + 1e-6))
    # different fmin can yield a frame off-by-one; trim to the shortest
    n = min(layer.shape[1] for layer in layers)
    return np.stack([layer[:, :n] for layer in layers], axis=0).astype("float32")


def hcqt_from_file(path, sr=SR, harmonics=HARMONICS):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return hcqt(y, sr=sr, harmonics=harmonics)


if __name__ == "__main__":
    # smoke test: 2s of noise -> expected shape
    import sys
    if len(sys.argv) > 1:
        f = hcqt_from_file(sys.argv[1])
    else:
        rng = np.random.default_rng(0)
        f = hcqt(rng.standard_normal(SR * 2).astype("float32"))
    print(f"HCQT shape {f.shape}  (harmonics, bins, frames)  "
          f"min={f.min():.2f} max={f.max():.2f} mean={f.mean():.2f}")
    # the h=1 channel must equal a plain BTC CQT
    base_idx = HARMONICS.index(1)
    print(f"harmonic-1 channel index = {base_idx} (BTC-compatible)")
