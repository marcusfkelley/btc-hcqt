# BTC-HCQT

**An honest, reproducible study of open-source automatic chord recognition (a.k.a. chord detection) — an HCQT variant of BTC, plus the benchmark harness behind it. Audio → a time-stamped chord progression, in Python.**

We set out to improve the best open chord-recognition model, **BTC** (Park et al., ISMIR 2019).
We tried a Harmonic-CQT (HCQT) front-end and a stack of other levers. **Honest result: our best
variant ties baseline BTC on held-out public benchmarks — it does not clearly beat it.** That's
a useful finding. This repo keeps the methodology, code, and benchmark results public;
the original research checkpoint is no longer offered as a current download.

> We are **not** claiming a better chord model. As far as we can measure, BTC is the best open
> one and the field has plateaued (~77–82% root across BTC, CREMA, Chordino, and our variants).
> The value here is a **documented benchmark**, **honest negative results**, and an
> **extensible HCQT base**. Full numbers → [BENCHMARK.md](BENCHMARK.md). Extend it →
> [EXTENSION-GUIDE.md](EXTENSION-GUIDE.md).

## Checkpoint availability and commercial-use boundary

The `btc_hcqt_beatlesft.pt` research checkpoint has been withdrawn from the current
repository. It inherited pretrained BTC weights and was fine-tuned on Beatles recordings.
The [upstream BTC authors](https://github.com/jayg996/BTC-ISMIR19) describe training on
Isophonics, Robbie Williams, and UsPop2002 audio obtained from music services; they do not
distribute that audio because of copyright. We have **not established that this
BTC-derived checkpoint is cleared for commercial deployment**. Do not treat this repository,
its MIT-licensed code, or its benchmark as an assurance that the checkpoint is suitable for
use in a commercial product.

For questions about the checkpoint or research access, [contact Selekt](https://selektaudio.com/contact)
and mention `BTC-HCQT`. We can explain its provenance and consider requests, but we cannot
grant rights to the underlying recordings. For a commercial project requiring documented
training rights, use a model trained on appropriately licensed data and verify the rights
and provenance of **all** starting weights, not just later fine-tuning data.

Earlier revisions of this repository described the checkpoint as MIT-licensed. This
update does not retroactively change prior grants or remove copies and Git history.

## What's here

| File | What it is |
|---|---|
| `hcqt.py` | Harmonic-CQT feature extraction (`HARMONICS = [0.5, 1, 2, 3, 4, 5]×`) |
| `hcqt_model.py` | HCQT front-ends (linear / conv / deep / FDAA+FGF) mounted on the BTC body |
| `hcqt_precompute.py` | Precompute HCQT features for a dataset |
| `hcqt_finetune.py` | Train / fine-tune the HCQT model (freeze-then-unfreeze schedule) |
| `hcqt_eval.py` | Run a trained HCQT checkpoint on audio → chord predictions |
| `btc_hcqt_beatlesft.pt` | Research checkpoint used for the reported parity result; withdrawn from current downloads (see above) |
| `bench_score.py` | mir_eval scorer (root / thirds / triads / sevenths / majmin / mirex + bootstrap CIs) |
| `bench_guitarset.py`, `bench_schubert.py`, `bench_isophonics.py` | Dataset loaders → refs + manifests |
| `run_eval.py` | Baseline lanes — BTC / CREMA / Chordino |
| `bench_ensemble.py`, `chordino_to_harte.py` | Ensemble experiment + a Chordino→Harte label fixer |
| `EXTENSION-GUIDE.md` | A concrete recipe to extend this base to melody / bass / transcription |

## Using the code with your own checkpoint

```bash
# 1. Get the BTC model architecture. The scripts expect it at ./BTC-ISMIR19.
git clone https://github.com/jayg996/BTC-ISMIR19

# 2. Install deps
pip install torch librosa mir_eval numpy pyyaml soundfile

# 3. Supply a checkpoint whose provenance and rights you have verified.
#    manifest = {"tracks":[{"id","title","audioUrl"}]}; audioUrl may be file://
python hcqt_eval.py /path/to/your_checkpoint.pt your_manifest.json out/

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
The source code and documentation in the current repository are offered under the
[MIT license](LICENSE). The previously published research checkpoint is not included in
the current files; see the availability and provenance notice above. Neither the code
license nor access to any checkpoint grants rights to training recordings or to audio
submitted for analysis. Chord predictions are estimates, not copyright or sample-clearance
decisions.

## From
Built by [Selekt](https://selektaudio.com) — cleared-sample tools for producers and composers.
