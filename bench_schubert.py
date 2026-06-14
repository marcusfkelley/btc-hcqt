#!/usr/bin/env python3
"""Schubert Winterreise (SWD) -> (refs, manifest) for the mir_eval harness.

Free classical benchmark (Zenodo 5139893, CC-BY): the HU33 + SC06 performances
ship free audio. Chord annotations are semicolon-delimited CSV with a Harte
'shorthand' column. Held out from BTC's pop/rock training -> a neutral, different-
genre cross-check. Usage: python3 bench_schubert.py /root/bench/swd
"""
import csv
import glob
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
ANN = os.path.join(ROOT, "02_Annotations", "ann_audio_chord")
AUD = os.path.join(ROOT, "01_RawData", "audio_wav")

refs, tracks = {}, []
for csvf in sorted(glob.glob(os.path.join(ANN, "*.csv"))):
    stem = os.path.basename(csvf)[:-4]                 # Schubert_D911-01_HU33
    if not (stem.endswith("HU33") or stem.endswith("SC06")):   # free-audio performances only
        continue
    wav = os.path.join(AUD, stem + ".wav")
    if not os.path.exists(wav):
        continue
    segs = []
    with open(csvf) as f:
        rdr = csv.reader(f, delimiter=";")
        next(rdr, None)                                # header
        for row in rdr:
            if len(row) < 3:
                continue
            try:
                a, b = float(row[0]), float(row[1])
            except ValueError:
                continue
            lab = row[2].strip().strip('"')            # 'shorthand' = Harte
            if b > a and lab not in ("N", "X"):
                segs.append([a, b, lab])
    if segs:
        refs[stem] = segs
        tracks.append({"id": stem, "title": stem, "audioUrl": "file://" + os.path.abspath(wav)})

json.dump(refs, open(os.path.join(ROOT, "refs-swd.json"), "w"))
json.dump({"tracks": tracks}, open(os.path.join(ROOT, "bench-swd.json"), "w"))
n7 = sum(1 for v in refs.values() for s in v if any(q in s[2] for q in ("7", "maj7", "min7")))
print(f"swd: {len(tracks)} tracks, {sum(len(v) for v in refs.values())} chords ({n7} sevenths)")
