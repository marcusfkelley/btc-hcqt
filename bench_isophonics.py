#!/usr/bin/env python3
"""Isophonics (.lab) -> (refs.json, manifest.json) for the mir_eval harness.

Isophonics Reference Annotations (Beatles / Queen / Carole King / Zweieck): the
canonical corpus BTC TRAINED on — our Condition A ("close on their turf"). The
chord annotations (.lab, Harte syntax) are free from isophonics.net; the AUDIO
is owned/purchased and supplied locally (evaluating on a track you own is fine;
we never redistribute it). We match each .lab to its audio by a normalized stem
(strip non-alphanumerics), which absorbs "01_-_Come_Together.lab" vs
"01 Come Together.mp3" style mismatches; --map lets you override stragglers.

  python3 bench_isophonics.py --ann chordlabs/ --audio beatles_audio/ \
        --out refs-beatles.json --manifest bench-beatles.json
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--ann", required=True, help="dir of .lab chord annotations (searched recursively)")
ap.add_argument("--audio", required=True, help="dir of owned audio (searched recursively)")
ap.add_argument("--out", default="refs-beatles.json")
ap.add_argument("--manifest", default="bench-beatles.json")
ap.add_argument("--map", default="", help="optional JSON {labStem: audioPath} for manual fixes")
args = ap.parse_args()


def norm(s):
    return "".join(c.lower() for c in s if c.isalnum())


audio = {}
for p in glob.glob(os.path.join(args.audio, "**", "*"), recursive=True):
    if p.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".aiff")):
        audio.setdefault(norm(os.path.splitext(os.path.basename(p))[0]), p)
manual = json.load(open(args.map)) if args.map else {}

refs, tracks, miss = {}, [], []
for lab in sorted(glob.glob(os.path.join(args.ann, "**", "*.lab"), recursive=True)):
    stem = os.path.splitext(os.path.basename(lab))[0]
    segs = []
    for line in open(lab):
        parts = line.split()
        if len(parts) >= 3:
            try:
                a, b = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if b > a and parts[2] not in ("N", "X"):
                segs.append([a, b, parts[2]])
    wav = manual.get(stem) or audio.get(norm(stem))
    if not segs or not wav:
        miss.append(stem)
        continue
    refs[stem] = segs
    tracks.append({"id": stem, "title": stem, "audioUrl": "file://" + os.path.abspath(wav)})

json.dump(refs, open(args.out, "w"))
json.dump({"tracks": tracks}, open(args.manifest, "w"))
print(f"isophonics: {len(tracks)} matched, {len(miss)} unmatched")
if miss:
    print("  unmatched stems (need audio or a --map entry):", ", ".join(miss[:8]), "..." if len(miss) > 8 else "")
