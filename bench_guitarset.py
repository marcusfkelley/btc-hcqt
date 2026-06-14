#!/usr/bin/env python3
"""GuitarSet -> (refs.json, manifest.json) for the mir_eval benchmark harness.

GuitarSet (Xi et al., ISMIR 2018, CC-BY-4.0): 360 short solo-guitar excerpts,
each recorded as a _comp (chordal comping) and a _solo (monophonic) take over a
lead-sheet progression. We score the _comp takes only — chords are not audible
in a single-note solo, so scoring _solo would unfairly punish every model. Held
out from BTC's full-band pop/rock training, and the solo-guitar timbre is a
clean cross-domain generalization probe. Chord labels are already Harte syntax.

Both audio and annotations ship freely, so this benchmark is FULLY reproducible
end-to-end — we can publish audio + labels + predictions + this code and anyone
re-runs the exact numbers.

  python3 bench_guitarset.py /root/bench/guitarset
"""
import glob
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
ann_files = glob.glob(os.path.join(ROOT, "**", "*.jams"), recursive=True)
aud = {os.path.basename(p): p for p in glob.glob(os.path.join(ROOT, "**", "*.wav"), recursive=True)}

refs, tracks, skipped = {}, [], 0
for jf in sorted(ann_files):
    tid = os.path.basename(jf)[:-5]            # strip ".jams"
    if "_comp" not in tid:                     # chordal takes only
        continue
    j = json.load(open(jf))
    chord = next((a for a in j["annotations"] if a["namespace"] == "chord"), None)
    if not chord:
        continue
    segs = [[float(o["time"]), float(o["time"] + o["duration"]), o["value"]]
            for o in chord["data"] if o["duration"] > 0 and o["value"] not in ("N", "X")]
    wav = aud.get(tid + "_mic.wav") or next((p for n, p in aud.items() if n.startswith(tid)), None)
    if not segs or not wav:
        skipped += 1
        continue
    refs[tid] = segs
    tracks.append({"id": tid, "title": tid, "audioUrl": "file://" + os.path.abspath(wav)})

json.dump(refs, open(os.path.join(ROOT, "refs-guitarset.json"), "w"))
json.dump({"tracks": tracks}, open(os.path.join(ROOT, "bench-guitarset.json"), "w"))
n7 = sum(1 for v in refs.values() for s in v if any(q in s[2] for q in ("7", "maj7", "min7")))
print(f"guitarset: {len(tracks)} comp tracks, "
      f"{sum(len(v) for v in refs.values())} chord segments ({n7} sevenths), {skipped} skipped")
