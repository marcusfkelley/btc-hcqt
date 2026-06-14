#!/usr/bin/env python3
"""
Precompute HCQT features for the whole training corpus (box-side, CPU-parallel).
Reads train-labels.jsonl, writes hcqt-feat/{id}.npz with:
  feat (frames, n_harmonics, 144) float32  +  lab (frames,) int64  (170-class).

Resume-safe; --shard i/N processes only its slice so the work splits across
boxes. Run:  python3 hcqt_precompute.py [--shard 0/6]
"""
import argparse
import json
import os
import re
from multiprocessing import Pool

import numpy as np
import librosa

from hcqt import hcqt, SR, HOP, HARMONICS  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "hcqt-feat")
os.makedirs(CACHE, exist_ok=True)
FPS = SR / HOP                                        # ~10.77 frames/sec
N_IDX = 169                                           # "N" (no chord)

ap = argparse.ArgumentParser()
ap.add_argument("--shard", default="0/1")            # i/N
args = ap.parse_args()
SI, SN = (int(x) for x in args.shard.split("/"))

# ── 170-class label mapping (identical to finetune.py) ───────────────────
ROOTS = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
         "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10,
         "Bb": 10, "B": 11}
QUAL_IDX = {"min": 0, "maj": 1, "dim": 2, "aug": 3, "min6": 4, "maj6": 5,
            "min7": 6, "minmaj7": 7, "maj7": 8, "7": 9, "dim7": 10,
            "hdim7": 11, "sus2": 12, "sus4": 13}
OURS_TO_BTC = {"": "maj", "m": "min", "7": "7", "maj7": "maj7", "m7": "min7",
               "sus4": "sus4", "sus2": "sus2", "dim": "dim", "dim7": "dim7",
               "m7b5": "hdim7", "aug": "aug", "6": "maj6", "m6": "min6",
               "mmaj7": "minmaj7"}
SYM_RE = re.compile(r"^([A-G][#b]?)(maj7|mmaj7|m7b5|dim7|sus4|sus2|aug|m7|m6|dim|m|7|6)?$")


def sym_to_idx(sym):
    m = SYM_RE.match(sym)
    if not m:
        return N_IDX
    root = ROOTS[m.group(1)]
    qual = OURS_TO_BTC.get(m.group(2) or "")
    if qual is None:
        return N_IDX
    return root * 14 + QUAL_IDX[qual]


labels = [json.loads(l) for l in open(os.path.join(BASE, "train-labels.jsonl")) if l.strip()]
mine = [r for i, r in enumerate(labels) if i % SN == SI]
print(f"shard {SI}/{SN}: {len(mine)} of {len(labels)} clips, {len(HARMONICS)} harmonics", flush=True)


def work(item):
    cid, split = item["id"], item["split"]
    out = os.path.join(CACHE, f"{cid}.npz")
    if os.path.exists(out):
        return None
    wav = os.path.join(BASE, "train-audio", split, f"{cid}.wav")
    if not os.path.exists(wav):
        return f"missing {cid}"
    try:
        y, _ = librosa.load(wav, sr=SR, mono=True)
        h = hcqt(y)                          # (H, 144, frames)
        feat = np.transpose(h, (2, 0, 1))    # (frames, H, 144)
        fr = np.full(feat.shape[0], N_IDX, dtype="int64")
        for ch in item.get("chords", []):
            a = int(round(ch["start"] * FPS))
            b = int(round(ch["end"] * FPS))
            fr[a:b] = sym_to_idx(ch["label"])
        np.savez_compressed(out, feat=feat.astype("float32"), lab=fr)
        return None
    except Exception as e:  # noqa: BLE001
        return f"FAIL {cid}: {e}"


with Pool(min(24, os.cpu_count() or 4)) as pool:
    errs = [e for e in pool.map(work, mine) if e]
print(f"shard {SI}/{SN} DONE — {len(mine) - len(errs)} ok, {len(errs)} errors", flush=True)
for e in errs[:15]:
    print(" ", e, flush=True)
