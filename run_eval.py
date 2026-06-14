#!/usr/bin/env python3
"""
Chord-model ear-eval runner (box-side).

For each track in tracks.json, runs every AVAILABLE model lane on:
  - master  (the full mix)
  - bed     (bass + other stems summed: harmony without drums/vocals)

Lanes (each optional — a lane that fails to import is skipped, not fatal):
  - crema   : CREMA chord model (conv-recurrent, extended vocab incl 7ths)
  - btc     : BTC transformer (ISMIR'19, large vocab incl 7ths)
  - chordino: NNLS-chroma Chordino (classic, extended vocab) via chord-extractor

Output: results-{lane}.json  → { trackId: { master: [...segs], bed: [...segs] } }
Each seg: { "start": float, "end": float, "label": "C:maj7", "conf": float|null }

Usage:  python3 run_eval.py tracks.json outdir/
"""
import json
import os
import sys
import tempfile
import traceback
import urllib.request

TRACKS_PATH = sys.argv[1] if len(sys.argv) > 1 else "tracks.json"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "results"
os.makedirs(OUTDIR, exist_ok=True)

UA = {"User-Agent": "selekt-chord-eval/1.0 (+https://selektaudio.com)"}


def fetch(url: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def load_mono(path: str, sr: int = 44100):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr


def make_bed(track) -> str | None:
    """Sum bass+other stems into one 'harmonic bed' wav (no drums/vocals)."""
    bed = track.get("bedUrls")
    if not bed or not bed.get("other"):
        return None
    import numpy as np
    import soundfile as sf
    parts = []
    for key in ("other", "bass"):
        url = bed.get(key)
        if not url:
            continue
        suffix = ".mp3" if ".mp3" in url.lower() else ".wav"
        p = fetch(url, suffix)
        y, sr = load_mono(p)
        parts.append(y)
        os.unlink(p)
    if not parts:
        return None
    n = max(len(p) for p in parts)
    mix = np.zeros(n, dtype="float32")
    for p in parts:
        mix[: len(p)] += p
    peak = float(abs(mix).max() or 1.0)
    if peak > 1.0:
        mix /= peak
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(out, mix, 44100)
    return out


# ── Lane: CREMA ──────────────────────────────────────────────────────────
def lane_crema():
    from crema.analyze import analyze  # noqa: F401

    def run(path):
        jam = analyze(filename=path)
        ann = jam.annotations.search(namespace="chord")[0]
        return [
            {"start": float(o.time), "end": float(o.time + o.duration),
             "label": str(o.value), "conf": float(o.confidence) if o.confidence is not None else None}
            for o in ann.data
        ]
    return run


# ── Lane: BTC (jayg996/BTC-ISMIR19) ─────────────────────────────────────
def lane_btc():
    """Wraps the BTC repo's inference. Expects repo cloned at ./BTC-ISMIR19
    with btc_model_large_voca.pt. We import its modules directly."""
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BTC-ISMIR19")
    if not os.path.isdir(repo):
        raise ImportError("BTC repo not present")
    sys.path.insert(0, repo)
    import torch
    import numpy as np
    import librosa
    from btc_model import BTC_model  # type: ignore
    from utils.hparams import HParams  # type: ignore
    from utils import logger  # noqa: F401
    from utils.mir_eval_modules import idx2voca_chord  # type: ignore

    config = HParams.load(os.path.join(repo, "run_config.yaml"))
    config.feature["large_voca"] = True
    config.model["num_chords"] = 170
    # BTC_CKPT env var swaps in a fine-tuned checkpoint (same format:
    # {'model','mean','std'}) — the before/after eval is just this swap.
    model_file = os.environ.get("BTC_CKPT") or os.path.join(repo, "test", "btc_model_large_voca.pt")
    if not os.path.isfile(model_file):
        model_file = os.path.join(repo, "btc_model_large_voca.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BTC_model(config=config.model).to(device)
    checkpoint = torch.load(model_file, map_location=device)
    mean, std = checkpoint["mean"], checkpoint["std"]
    model.load_state_dict(checkpoint["model"])
    model.eval()
    idx_to_chord = idx2voca_chord()

    # Match the repo's test.py: fps = song_hz / hop_length; window = model.timestep
    feature_per_second = config.mp3["song_hz"] / config.feature["hop_length"]
    seq_len = config.model["timestep"]

    def run(path):
        y, sr = librosa.load(path, sr=config.mp3["song_hz"], mono=True)
        feature = librosa.cqt(y, sr=sr,
                              n_bins=config.feature["n_bins"],
                              bins_per_octave=config.feature["bins_per_octave"],
                              hop_length=config.feature["hop_length"])
        feature = np.log(np.abs(feature) + 1e-6).T  # (frames, bins)
        feature = (feature - mean) / std
        n_timestep = seq_len
        num_pad = n_timestep - (feature.shape[0] % n_timestep)
        feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant")
        num_instance = feature.shape[0] // n_timestep

        segs = []
        start_time = 0.0
        prev = None
        with torch.no_grad():
            feat_t = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
            for t in range(num_instance):
                chunk = feat_t[:, n_timestep * t: n_timestep * (t + 1), :]
                self_attn_output, _ = model.self_attn_layers(chunk)
                prediction, _ = model.output_layer(self_attn_output)
                prediction = prediction.squeeze()
                for i in range(n_timestep):
                    idx = int(prediction[i].item())
                    tcur = (n_timestep * t + i) / feature_per_second
                    if prev is None:
                        prev, start_time = idx, tcur
                    elif idx != prev:
                        segs.append({"start": round(start_time, 3), "end": round(tcur, 3),
                                     "label": idx_to_chord[prev], "conf": None})
                        prev, start_time = idx, tcur
        if prev is not None:
            segs.append({"start": round(start_time, 3),
                         "end": round(len(y) / sr, 3),
                         "label": idx_to_chord[prev], "conf": None})
        return [s for s in segs if s["end"] - s["start"] >= 0.20]
    return run


# ── Lane: Chordino ───────────────────────────────────────────────────────
def lane_chordino():
    from chord_extractor.extractors import Chordino  # type: ignore
    ch = Chordino()

    def run(path):
        res = ch.extract(path)
        segs = []
        for i, c in enumerate(res):
            end = res[i + 1].timestamp if i + 1 < len(res) else c.timestamp + 2.0
            segs.append({"start": float(c.timestamp), "end": float(end),
                         "label": str(c.chord), "conf": None})
        return segs
    return run


LANES = {}
# LANES env filter ("chordino" or "btc,crema") — re-run one lane without
# redoing the others; results-{lane}.json files for other lanes are untouched.
_only = [s.strip() for s in os.environ.get("LANES", "").split(",") if s.strip()]
for name, factory in (("crema", lane_crema), ("btc", lane_btc), ("chordino", lane_chordino)):
    if _only and name not in _only:
        print(f"[lane] {name}: skipped (LANES filter)", flush=True)
        continue
    try:
        LANES[name] = factory()
        print(f"[lane] {name}: READY", flush=True)
    except Exception as e:
        print(f"[lane] {name}: unavailable ({type(e).__name__}: {e})", flush=True)

if not LANES:
    print("No lanes available — install at least one model.", flush=True)
    sys.exit(1)

tracks = json.load(open(TRACKS_PATH))["tracks"]
results = {name: {} for name in LANES}

for n, t in enumerate(tracks, 1):
    print(f"\n[{n}/{len(tracks)}] {t['title'][:60]}", flush=True)
    try:
        master = fetch(t["audioUrl"], ".mp3" if ".mp3" in t["audioUrl"].lower() else ".wav")
    except Exception as e:
        print(f"  master download failed: {e}", flush=True)
        continue
    bed = None
    try:
        bed = make_bed(t)
    except Exception as e:
        print(f"  bed build failed: {e}", flush=True)

    for name, run in LANES.items():
        entry = {}
        for lane_name, p in (("master", master), ("bed", bed)):
            if not p:
                continue
            try:
                entry[lane_name] = run(p)
                print(f"  {name}/{lane_name}: {len(entry[lane_name])} segs", flush=True)
            except Exception as e:
                print(f"  {name}/{lane_name} FAILED: {e}", flush=True)
                traceback.print_exc()
        results[name][t["id"]] = entry
        # checkpoint after every track so a crash loses nothing
        json.dump(results[name], open(os.path.join(OUTDIR, f"results-{name}.json"), "w"))

    os.unlink(master)
    if bed:
        os.unlink(bed)

print("\nDONE", flush=True)
for name in LANES:
    print(f"  results-{name}.json: {len(results[name])} tracks", flush=True)
