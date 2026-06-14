#!/usr/bin/env python3
"""
mir_eval benchmark scorer — the citable head-to-head harness for the paper.

Scores one or more model prediction files against a reference-annotation file
with mir_eval's STANDARD chord metrics (root / thirds / triads / sevenths /
majmin / mirex), duration-weighted per track, then aggregated across tracks
with a per-track bootstrap 95% CI. CIs are the whole point: "comes close" and
"beats" become quantified claims, not eyeballed single numbers (the lesson from
the 2-track JAAH noise).

Contract — both sides are dataset-agnostic so every benchmark reuses this:
  predictions : run_eval.py / hcqt_eval.py output
                {trackId: {"master": [{start,end,label}, ...]}}
  references  : {trackId: [[start, end, "C:maj7"], ...]}
                produced by the bench_<dataset>.py loaders (Harte/mir_eval syntax)

  python3 bench_score.py --ref refs-guitarset.json --out scores-guitarset.json \
        baseBTC=results/gs-base.json  thin=results/gs-thin.json  thin_hcqt=results/gs-thinhcqt.json
"""
import argparse
import json
import random

import numpy as np
import mir_eval

METRICS = ["root", "thirds", "triads", "sevenths", "majmin", "mirex"]
NC = mir_eval.chord.NO_CHORD


def sanitize(label):
    """Coerce a model/annotation label into something mir_eval can encode.
    Unknown / X / silence -> no-chord; un-parseable quality -> root triad."""
    if not label or label in ("N", "X", "None"):
        return NC
    try:
        mir_eval.chord.encode(label)
        return label
    except Exception:
        root = str(label).split(":")[0].split("/")[0]
        try:
            mir_eval.chord.encode(root + ":maj")
            return root + ":maj"
        except Exception:
            return NC


def to_arrays(segs):
    """segs: list of {start,end,label} OR [start,end,label] -> (intervals, labels)."""
    rows = []
    for s in segs:
        if isinstance(s, dict):
            a, b, lab = s["start"], s["end"], s.get("label", "N")
        else:
            a, b, lab = s[0], s[1], s[2]
        if b > a:
            rows.append((float(a), float(b), sanitize(lab)))
    rows.sort()
    if not rows:
        return None, None
    iv = np.array([[a, b] for a, b, _ in rows], dtype=float)
    return iv, [lab for _, _, lab in rows]


def score_track(ref_segs, est_segs):
    ref_iv, ref_lab = to_arrays(ref_segs)
    est_iv, est_lab = to_arrays(est_segs)
    if ref_iv is None:
        return None, 0.0
    if est_iv is None:  # model emitted nothing -> all no-chord over the ref span
        est_iv, est_lab = np.array([[ref_iv[0, 0], ref_iv[-1, 1]]]), [NC]
    # clip estimate to the reference span and fill gaps with no-chord (mir_eval contract)
    est_iv, est_lab = mir_eval.util.adjust_intervals(
        est_iv, est_lab, ref_iv.min(), ref_iv.max(), NC, NC)
    intervals, r_lab, e_lab = mir_eval.util.merge_labeled_intervals(
        ref_iv, ref_lab, est_iv, est_lab)
    durations = mir_eval.util.intervals_to_durations(intervals)
    out = {}
    for m in METRICS:
        comp = getattr(mir_eval.chord, m)(r_lab, e_lab)
        out[m] = mir_eval.chord.weighted_accuracy(comp, durations)
    return out, float(ref_iv[-1, 1] - ref_iv[0, 0])


def weighted_mean(per_track, weights, metric):
    w = np.array(weights, float)
    v = np.array([t[metric] for t in per_track], float)
    return float((v * w).sum() / w.sum()) if w.sum() else float("nan")


def bootstrap_ci(per_track, weights, metric, n=2000, seed=20260613):
    rng = random.Random(seed)
    k = len(per_track)
    if k < 2:
        m = weighted_mean(per_track, weights, metric)
        return m, m
    means = []
    idxs = list(range(k))
    for _ in range(n):
        samp = [rng.choice(idxs) for _ in range(k)]
        pt = [per_track[i] for i in samp]
        wt = [weights[i] for i in samp]
        means.append(weighted_mean(pt, wt, metric))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="references JSON {id: [[s,e,label],...]}")
    ap.add_argument("--lane", default="master", help="prediction lane to score (master|bed)")
    ap.add_argument("--out", default="", help="optional JSON dump of full results")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("models", nargs="+", help="name=predictions.json ...")
    args = ap.parse_args()

    refs = json.load(open(args.ref))
    print(f"reference: {len(refs)} tracks  ({args.ref})\n")

    table = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        preds = json.load(open(path))
        per_track, weights, ids = [], [], []
        for tid, ref_segs in refs.items():
            if tid not in preds:          # not run for this model -> skip (don't score as no-chord)
                continue
            entry = preds.get(tid) or {}
            est = entry.get(args.lane) or entry.get("master") or []
            sc, dur = score_track(ref_segs, est)
            if sc is None or dur <= 0:
                continue
            per_track.append(sc)
            weights.append(dur)
            ids.append(tid)
        row = {}
        for m in METRICS:
            mean = weighted_mean(per_track, weights, m)
            lo, hi = bootstrap_ci(per_track, weights, m, n=args.boot)
            row[m] = {"mean": mean, "lo": lo, "hi": hi}
        row["_n"] = len(per_track)
        table[name] = row

    hdr = "model".ljust(16) + "n".rjust(4) + "".join(f"  {m}".rjust(20) for m in METRICS)
    print(hdr)
    print("-" * len(hdr))
    for name, row in table.items():
        line = name.ljust(16) + str(row["_n"]).rjust(4)
        for m in METRICS:
            c = row[m]
            line += f"  {100*c['mean']:5.1f} [{100*c['lo']:4.1f}-{100*c['hi']:4.1f}]".rjust(20)
        print(line)
    print("\n(values = duration-weighted accuracy %, [95% bootstrap CI over tracks])")

    if args.out:
        json.dump(table, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
