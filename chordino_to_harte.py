#!/usr/bin/env python3
"""Convert chord-extractor / Chordino labels ("Fm", "Ebmaj7", "Bb6") to Harte
("F:min", "Eb:maj7", "Bb:maj6") so bench_score / the ensemble parse them correctly.
Without this, every minor/7th silently becomes no-chord (Chordino scored 42 instead
of its real ~70). Usage: python3 chordino_to_harte.py in.json out.json"""
import json
import re
import sys

QMAP = {"": "maj", "m": "min", "maj": "maj", "min": "min", "maj7": "maj7", "m7": "min7",
        "min7": "min7", "7": "7", "dim": "dim", "aug": "aug", "sus4": "sus4", "sus2": "sus2",
        "6": "maj6", "m6": "min6", "min6": "min6", "maj6": "maj6", "dim7": "dim7",
        "hdim7": "hdim7", "m7b5": "hdim7", "9": "7", "maj9": "maj7", "m9": "min7",
        "min9": "min7", "11": "7", "13": "7", "add9": "maj", "mmaj7": "minmaj7", "minmaj7": "minmaj7"}


def to_harte(lab):
    if not lab or lab in ("N", "X"):
        return "N"
    s = lab.split("/")[0].strip()
    m = re.match(r"^([A-G][#b]?)(.*)$", s)
    if not m:
        return "N"
    return m.group(1) + ":" + QMAP.get(m.group(2).strip(), "maj")


d = json.load(open(sys.argv[1]))
for e in d.values():
    for ln in e:
        for s in e[ln]:
            s["label"] = to_harte(s["label"])
json.dump(d, open(sys.argv[2], "w"))
print(f"converted {len(d)} tracks -> {sys.argv[2]}")
