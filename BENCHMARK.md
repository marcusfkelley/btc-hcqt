# Benchmark — the honest numbers

## Harness
`bench_score.py` runs **mir_eval** standard chord metrics — root, thirds, triads, sevenths,
majmin, MIREX — **duration-weighted**, with **per-track bootstrap 95% confidence intervals**.
All benchmarks are held out from BTC's training set.

## Datasets
- **GuitarSet** (NYU MARL, CC-BY-4.0) — 180 solo-guitar comping tracks. Audio + labels both
  free → fully reproducible.
- **Schubert Winterreise** (Zenodo 5139893, CC-BY-3.0) — 48 free classical performances
  (HU33 + SC06), **33% seventh chords** — a real test of dense harmony.

## Results

### GuitarSet (neutral ground), duration-weighted %
| model | root | sevenths | majmin | mirex |
|---|---|---|---|---|
| **baseline BTC** | **80.9** | **64.6** | **77.0** | 76.1 |
| BTC+HCQT (Beatles-FT) — *our best* | 80.5 | 63.0 | — | — |
| BTC+HCQT (pre fine-tune) | 80.4 | 58.8 | 75.2 | 76.1 |
| clean-from-scratch (thin / 7rich) | ~71–73 | ~49–53 | ~65–67 | ~67–69 |

### Schubert Winterreise (classical, 33% sevenths)
| model | root | sevenths | mirex |
|---|---|---|---|
| **ours (BTC+HCQT, Beatles-FT)** | **73.8** | **55.6** | **65.3** |
| baseline BTC | 73.1 | 55.3 | 64.1 |

**Net:** a statistical dead heat across two held-out genres. BTC +0.4 root on guitar; ours
+0.7 root / +1.2 mirex on classical. Every gap sits inside the confidence intervals.

## The methodological lesson (the most useful thing here)
Early on, an in-house "modern-7th" metric showed HCQT **doubling** seventh detection (27→48%).
It was a **recall artifact** — a 7th-saturated training set taught the model to *over-call*
7ths (high recall, low precision). On frame-wise mir_eval the ranking **flipped**: baseline
BTC, which calls fewer 7ths but gets them right, is best at them. **Never trust a bespoke
recall metric for a "we improved it" claim — use frame-wise mir_eval with CIs.**

## Standing conclusions
1. **Baseline BTC is hard to beat** — real-music training generalizes to held-out real audio
   better than synthetic-trained or representation-tweaked variants.
2. **Bespoke recall metrics mislead** — the "doubling" inverted under frame-wise mir_eval.
3. **Corpus composition** — 7th-saturation causes over-calling; realistic balance + multi-key
   transposition beat raw 7th density.
4. **HCQT-on-synthetic causes catastrophic forgetting** of a pretrained body's real-music
   knowledge; freeze the body or fine-tune on real audio.
5. **Beating a strong baseline needs a *system*** (ensemble + LM decoder + separation), not a
   single architecture change.
6. **The durable asset is the benchmark + the honest negatives**, independent of beating BTC.

## Context
The published 2025 SOTA over BTC — **BTC-FDAA-FGF** (Computers & Electrical Engineering 2025) —
is itself built on an **HCQT front-end** (the same representation here), adding two modules
(FDAA + FGF) for +1.2–2.2% MIREX. Neither it nor ChordFormer has public code — part of why an
open, reproducible benchmark + honest comparison is useful on its own.
