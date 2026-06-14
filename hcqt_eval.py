#!/usr/bin/env python3
"""
HCQT-BTC inference — same output format as run_eval.py (results-btc.json:
{trackId: {master: [segs]}}) so every existing scorer works unchanged.

  python3 hcqt_eval.py <checkpoint.pt> <manifest.json> <outdir>
"""
import json
import os
import sys
import tempfile
import urllib.request

import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, "BTC-ISMIR19")
sys.path.insert(0, REPO)

import librosa  # noqa: E402
from btc_model import BTC_model  # noqa: E402
from utils.hparams import HParams  # noqa: E402
from utils.mir_eval_modules import idx2voca_chord  # noqa: E402
from hcqt import hcqt, SR, HOP  # noqa: E402
from hcqt_model import HCQT_BTC  # noqa: E402

CKPT, MANIFEST, OUTDIR = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUTDIR, exist_ok=True)

config = HParams.load(os.path.join(REPO, "run_config.yaml"))
config.feature["large_voca"] = True
config.model["num_chords"] = 170
TIMESTEP = config.model["timestep"]
FPS = SR / HOP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

c = torch.load(CKPT, map_location=device)
btc = BTC_model(config=config.model).to(device)
btc.load_state_dict(c["btc"])
model = HCQT_BTC(btc, c["mean"], c["std"], c["n_harm"], c["frontend_kind"]).to(device)
model.frontend.load_state_dict(c["frontend"])
model.eval()
idx_to_chord = idx2voca_chord()
print(f"loaded {os.path.basename(CKPT)} ({c['frontend_kind']}, {c['n_harm']} harm)", flush=True)

UA = {"User-Agent": "selekt-hcqt-eval/1.0"}


def fetch(url):
    suffix = ".mp3" if ".mp3" in url.lower() else ".wav"
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    if url.startswith("file://"):
        return url[7:], False
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as f:
        f.write(r.read())
    return p, True


def infer(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    feat = np.transpose(hcqt(y), (2, 0, 1))           # (frames, H, 144)
    n_pad = TIMESTEP - (feat.shape[0] % TIMESTEP)
    feat = np.pad(feat, ((0, n_pad), (0, 0), (0, 0)), mode="constant")
    n_inst = feat.shape[0] // TIMESTEP
    t = torch.tensor(feat, dtype=torch.float32).to(device)
    segs, prev, start = [], None, 0.0
    with torch.no_grad():
        for k in range(n_inst):
            chunk = t[TIMESTEP * k: TIMESTEP * (k + 1)].unsqueeze(0)   # (1,T,H,144)
            B, T, H, F = chunk.shape
            x = model.frontend(chunk.reshape(B * T, H, F)).reshape(B, T, F)
            x = (x - model.mean) / model.std
            attn, _ = model.btc.self_attn_layers(x)
            pred, _ = model.btc.output_layer(attn)
            pred = pred.squeeze()
            for i in range(TIMESTEP):
                idx = int(pred[i].item())
                tcur = (TIMESTEP * k + i) / FPS
                if prev is None:
                    prev, start = idx, tcur
                elif idx != prev:
                    segs.append({"start": round(start, 3), "end": round(tcur, 3),
                                 "label": idx_to_chord[prev], "conf": None})
                    prev, start = idx, tcur
    if prev is not None:
        segs.append({"start": round(start, 3), "end": round(len(y) / SR, 3),
                     "label": idx_to_chord[prev], "conf": None})
    return [s for s in segs if s["end"] - s["start"] >= 0.20]


tracks = json.load(open(MANIFEST))["tracks"]
results = {}
for n, tk in enumerate(tracks, 1):
    try:
        path, tmp = fetch(tk["audioUrl"])
        results[tk["id"]] = {"master": infer(path)}
        if tmp:
            os.unlink(path)
        print(f"  [{n}/{len(tracks)}] {tk['title'][:50]}: {len(results[tk['id']]['master'])} segs", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  [{n}/{len(tracks)}] FAIL {tk['id']}: {e}", flush=True)
    json.dump(results, open(os.path.join(OUTDIR, "results-btc.json"), "w"))

print(f"HCQT EVAL DONE {len(results)} tracks -> {OUTDIR}", flush=True)
