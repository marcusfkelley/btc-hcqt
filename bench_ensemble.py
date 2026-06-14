#!/usr/bin/env python3
"""Chord ensemble combiner for the bench harness.

--chordino given  -> 3-model MAJORITY VOTE (BTC + CREMA + Chordino): per frame the
                     chord 2+ models agree on wins; on a 3-way split, fall back to
                     CREMA (strongest single on root). This is the classic ensemble
                     that beats any individual.
--chordino absent -> 2-model 7th-UPGRADE (CREMA primary; upgrade CREMA's triad to
                     BTC's 7th at the same root).

Outputs combined predictions ({id:{master:[segs]}}) that bench_score.py scores.
  python3 bench_ensemble.py --crema C.json --btc B.json [--chordino H.json] --out O.json
"""
import argparse
import json
from collections import Counter

ROOTS = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6,
         "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11}
ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
TRIAD = {"maj", "min", ""}
SEVENTH = {"7", "maj7", "min7", "9", "maj9", "min9", "dom7", "m7b5", "hdim7", "dim7", "minmaj7"}
FPS = 10


def parse(label):
    if not label or label in ("N", "X"):
        return (None, None)
    s = str(label).split("/")[0]
    r, q = (s.split(":", 1) if ":" in s else (s, "maj"))
    root = ROOTS.get(r)
    return (root, q.split("(")[0]) if root is not None else (None, None)


def qclass(q):
    q = (q or "").lower()
    if q in ("maj7", "maj9", "maj11", "maj13"):
        return "maj7"
    if q in ("m7", "min7", "m9", "min9", "min11", "hdim7", "m7b5"):
        return "min7"
    if q in ("7", "9", "11", "13", "dom7"):
        return "7"
    if q.startswith("dim"):
        return "dim"
    if q.startswith("aug"):
        return "aug"
    if q.startswith("sus"):
        return "sus4"
    if q in ("m", "min", "minor", "m6", "min6"):
        return "min"
    return "maj"


def canon(label):
    r, q = parse(label)
    return None if r is None else (r, qclass(q))


def clabel(c):
    return "N" if c is None else f"{ROOT_NAMES[c[0]]}:{c[1]}"


def frames(segs):
    segs = [s for s in (segs or []) if s["end"] > s["start"]]
    if not segs:
        return []
    end = max(s["end"] for s in segs)
    fr = ["N"] * (int(end * FPS) + 1)
    for s in sorted(segs, key=lambda x: x["start"]):
        for i in range(int(s["start"] * FPS), min(int(s["end"] * FPS), len(fr))):
            fr[i] = s["label"]
    return fr


def to_segs(fr):
    out, i = [], 0
    while i < len(fr):
        j = i
        while j < len(fr) and fr[j] == fr[i]:
            j += 1
        out.append({"start": round(i / FPS, 3), "end": round(j / FPS, 3), "label": fr[i], "conf": None})
        i = j
    return out


def lane(d, tid, ln):
    e = d.get(tid) or {}
    return e.get(ln) or e.get("master")


ap = argparse.ArgumentParser()
ap.add_argument("--crema", required=True)
ap.add_argument("--btc", required=True)
ap.add_argument("--chordino", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--lane", default="master")
args = ap.parse_args()
crema = json.load(open(args.crema))
btc = json.load(open(args.btc))
chord = json.load(open(args.chordino)) if args.chordino else None

combined = {}
for tid, ce in crema.items():
    cf = frames(ce.get(args.lane) or ce.get("master"))
    bf = frames(lane(btc, tid, args.lane))
    if not cf:
        continue
    out = []
    if chord is not None:                              # 3-model majority vote
        hf = frames(lane(chord, tid, args.lane))
        for i in range(len(cf)):
            votes = [canon(cf[i]),
                     canon(bf[i]) if i < len(bf) else None,
                     canon(hf[i]) if i < len(hf) else None]
            cnt = Counter(v for v in votes if v is not None)
            if cnt and cnt.most_common(1)[0][1] >= 2:
                out.append(clabel(cnt.most_common(1)[0][0]))
            else:
                out.append(cf[i])                      # CREMA fallback
    else:                                              # 2-model 7th-upgrade
        for i, clab in enumerate(cf):
            cr, cq = parse(clab)
            lab = clab
            if cr is not None and cq in TRIAD and i < len(bf):
                br, bq = parse(bf[i])
                if br == cr and bq in SEVENTH:
                    lab = bf[i]
            out.append(lab)
    combined[tid] = {"master": to_segs(out)}

json.dump(combined, open(args.out, "w"))
mode = "3-model majority vote" if chord is not None else "2-model 7th-upgrade"
print(f"ensemble ({mode}): {len(combined)} tracks -> {args.out}")
