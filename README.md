# BTC-HCQT

**An honest, reproducible study of open-source automatic chord recognition (a.k.a. chord detection) — an HCQT variant of BTC, plus the benchmark harness behind it. Audio → a time-stamped chord progression, in Python.**

We set out to improve the best open chord-recognition model, **BTC** (Park et al., ISMIR 2019).
We tried a Harmonic-CQT (HCQT) front-end and a stack of other levers. **Honest result: our best
variant ties baseline BTC on held-out public benchmarks — it does not clearly beat it.** That's
a useful finding, and this repo ships everything to reproduce it.

> We are **not** claiming a better chord model. As far as we can measure, BTC is the best open
> one and the field has plateaued (~77–82% root across BTC, CREMA, Chordino, and our variants).
> The value here is a **reproducible benchmark**, **honest negative results**, and an
> **extensible HCQT base**. Full numbers → [BENCHMARK.md](BENCHMARK.md). Extend it →
> [EXTENSION-GUIDE.md](EXTENSION-GUIDE.md).

## What's here

| File | What it is |
|---|---|
| `hcqt.py` | Harmonic-CQT feature extraction (`HARMONICS = [0.5, 1, 2, 3, 4, 5]×`) |
| `hcqt_model.py` | HCQT front-ends (linear / conv / deep / FDAA+FGF) mounted on the BTC body |
| `hcqt_precompute.py` | Precompute HCQT features for a dataset |
| `hcqt_finetune.py` | Train / fine-tune the HCQT model (freeze-then-unfreeze schedule) |
| `hcqt_eval.py` | Run a trained HCQT checkpoint on audio → chord predictions |
| `btc_hcqt_beatlesft.pt` | **The parity model** — Beatles-fine-tuned BTC+HCQT (ties baseline BTC) |
| `bench_score.py` | mir_eval scorer (root / thirds / triads / sevenths / majmin / mirex + bootstrap CIs) |
| `bench_guitarset.py`, `bench_schubert.py`, `bench_isophonics.py` | Dataset loaders → refs + manifests |
| `run_eval.py` | Baseline lanes — BTC / CREMA / Chordino |
| `bench_ensemble.py`, `chordino_to_harte.py` | Ensemble experiment + a Chordino→Harte label fixer |
| `EXTENSION-GUIDE.md` | A concrete recipe to extend this base to melody / bass / transcription |

## Quick start

```bash
# 1. Get the BTC model architecture. Our checkpoint carries the fine-tuned weights;
#    you only need BTC-ISMIR19's btc_model.py to instantiate the body. Run this from
#    inside this folder so it lands at ./BTC-ISMIR19 (the scripts look there).
git clone https://github.com/jayg996/BTC-ISMIR19

# 2. Install deps
pip install torch librosa mir_eval numpy pyyaml soundfile

# 3. Run the model on audio. manifest = {"tracks":[{"id","title","audioUrl"}]}; audioUrl may be file://
python hcqt_eval.py btc_hcqt_beatlesft.pt your_manifest.json out/

# 4. Score against a reference. refs = {id: [[start, end, "C:maj"], ...]}
python bench_score.py --ref refs.json "ours=out/results-btc.json"
```

## Benchmark (held-out, public data)

| Model | GuitarSet root / 7ths | Schubert root / 7ths / mirex |
|---|---|---|
| baseline BTC | **80.9** / **64.6** | 73.1 / 55.3 / 64.1 |
| **ours (BTC+HCQT, Beatles-FT)** | 80.5 / 63.0 | **73.8 / 55.6 / 65.3** |

A dead heat — BTC noses ahead on guitar, we nose ahead on classical, every gap within the 95%
CIs. Full tables, methodology, and the lessons: [BENCHMARK.md](BENCHMARK.md).

## Built on / thanks
- **BTC** — Park et al., *A Bi-directional Transformer for Musical Chord Recognition*, ISMIR
  2019 — [jayg996/BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19).
- **HCQT** — Bittner et al., *Deep Salience Representations for F0 Estimation in Polyphonic
  Music*, ISMIR 2017.
- Benchmarks — GuitarSet (NYU MARL, CC-BY-4.0); Schubert Winterreise (Zenodo 5139893, CC-BY-3.0).

## License
Code + weights: **MIT** (see [LICENSE](LICENSE)). Built on BTC (MIT) — attribution above.

## From
Built by [Selekt](https://selektaudio.com) — cleared-sample tools for producers and composers.
