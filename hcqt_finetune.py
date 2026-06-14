#!/usr/bin/env python3
"""
HCQT-BTC fine-tune (box-side). One point in the sweep per invocation:

  python3 hcqt_finetune.py --frontend linear --freeze-epochs 4 --lr 3e-4 \
        --epochs 24 --out btc_hcqt_linear.pt

Loads precomputed HCQT features (hcqt-feat/), initializes from the published
BTC checkpoint (front-end fresh, body pretrained), trains with a freeze-then-
unfreeze schedule, saves {frontend, btc, mean, std, n_harm, frontend_kind}.
Validation = seeded 5% of split=train; test-* never touched.
"""
import argparse
import json
import os
import random
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, "BTC-ISMIR19")
sys.path.insert(0, REPO)

import torch  # noqa: E402
from btc_model import BTC_model  # noqa: E402
from utils.hparams import HParams  # noqa: E402
from hcqt import HARMONICS  # noqa: E402
from hcqt_model import HCQT_BTC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frontend", default="linear", choices=["linear", "conv", "deep", "fdaafgf"])
ap.add_argument("--freeze-epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--body-lr", type=float, default=1e-5)
ap.add_argument("--epochs", type=int, default=24)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--val-frac", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=20260613)
ap.add_argument("--base", default="", help="clean BTC checkpoint to mount the HCQT front-end on (default: published Beatles BTC). Use btc_scratch.pt / btc_scratch_7rich.pt for the license-clean variants.")
ap.add_argument("--resume", default="", help="resume from an existing HCQT ckpt (frontend+btc+mean+std) and keep fine-tuning it — e.g. fine-tune btc_hcqt_deep on real Beatles audio")
ap.add_argument("--out", default="btc_hcqt.pt")
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

config = HParams.load(os.path.join(REPO, "run_config.yaml"))
config.feature["large_voca"] = True
config.model["num_chords"] = 170
TIMESTEP = config.model["timestep"]
N_HARM = len(HARMONICS)
FEAT = config.feature["n_bins"]          # 144
CACHE = os.path.join(BASE, "hcqt-feat")

labels = [json.loads(l) for l in open(os.path.join(BASE, "train-labels.jsonl")) if l.strip()]
train_items = [l for l in labels if l["split"] == "train" and os.path.exists(os.path.join(CACHE, f"{l['id']}.npz"))]
random.shuffle(train_items)
n_val = max(1, int(len(train_items) * args.val_frac))
val_items, train_items = train_items[:n_val], train_items[n_val:]
print(f"[{args.frontend}] clips train={len(train_items)} val={len(val_items)}", flush=True)

store = {}


def windows(items):
    idx = []
    for it in items:
        p = os.path.join(CACHE, f"{it['id']}.npz")
        if it["id"] not in store:
            z = np.load(p)
            store[it["id"]] = (z["feat"], z["lab"])    # (frames,H,144),(frames,)
        frames = store[it["id"]][0].shape[0]
        for w in range(max(1, int(np.ceil(frames / TIMESTEP)))):
            idx.append((it["id"], w * TIMESTEP))
    return idx


train_idx, val_idx = windows(train_items), windows(val_items)
print(f"[{args.frontend}] windows train={len(train_idx)} val={len(val_idx)}", flush=True)


def batch_of(sl):
    B = len(sl)
    fb = np.zeros((B, TIMESTEP, N_HARM, FEAT), dtype="float32")
    lb = np.full((B, TIMESTEP), 169, dtype="int64")
    for i, (cid, s) in enumerate(sl):
        feat, lab = store[cid]
        seg = feat[s:s + TIMESTEP]
        fb[i, :seg.shape[0]] = seg
        ls = lab[s:s + TIMESTEP]
        lb[i, :ls.shape[0]] = ls
    return torch.from_numpy(fb), torch.from_numpy(lb)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.resume:
    rk = torch.load(args.resume, map_location=device)
    btc = BTC_model(config=config.model).to(device)
    btc.load_state_dict(rk["btc"])
    fe_kind = args.frontend or rk.get("frontend_kind")     # allow UPGRADING the front-end
    model = HCQT_BTC(btc, rk["mean"], rk["std"], N_HARM, fe_kind).to(device)
    fe_state = rk["frontend"]
    if fe_kind == "fdaafgf" and rk.get("frontend_kind") == "deep":   # deep weights -> .deep submodule
        fe_state = {("deep." + k[4:] if k.startswith("net.") else k): v for k, v in fe_state.items()}
    missing, unexpected = model.frontend.load_state_dict(fe_state, strict=False)
    print(f"[{args.frontend}] RESUME {args.resume} (fe {rk.get('frontend_kind')}->{fe_kind}; "
          f"{len(missing)} fresh modules)", flush=True)
else:
    if args.base:
        ckpt_path = args.base if os.path.isfile(args.base) else os.path.join(BASE, args.base)
    else:
        ckpt_path = os.path.join(REPO, "test", "btc_model_large_voca.pt")
        if not os.path.isfile(ckpt_path):
            ckpt_path = os.path.join(REPO, "btc_model_large_voca.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"[{args.frontend}] base ckpt = {ckpt_path}", flush=True)
    btc = BTC_model(config=config.model).to(device)
    btc.load_state_dict(ckpt["model"])
    model = HCQT_BTC(btc, ckpt["mean"], ckpt["std"], N_HARM, args.frontend).to(device)
print(f"[{args.frontend}] device={device} freeze_epochs={args.freeze_epochs} lr={args.lr}/{args.body_lr}", flush=True)


def make_opt(body_trainable):
    groups = [{"params": model.frontend.parameters(), "lr": args.lr}]
    if body_trainable:
        groups.append({"params": model.btc.parameters(), "lr": args.body_lr})
    return torch.optim.Adam(groups, betas=(0.9, 0.98), eps=1e-9)


def run_epoch(index, opt, train):
    model.train() if train else model.eval()
    order = list(index)
    if train:
        random.shuffle(order)
    tot = correct = 0
    losses = []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i in range(0, len(order), args.batch):
            fb, lb = batch_of(order[i:i + args.batch])
            fb, lb = fb.to(device), lb.to(device)
            if train:
                opt.zero_grad()
            pred, loss, _, _ = model(fb, lb)
            if train:
                loss.backward(); opt.step()
            losses.append(float(loss.item()))
            tot += lb.numel(); correct += int((pred == lb.flatten()).sum().item())
    return float(np.mean(losses)), correct / tot


best, patience, frozen = 0.0, 0, True
model.set_body_trainable(False)
opt = make_opt(False)
OUT = os.path.join(BASE, args.out)
for epoch in range(1, args.epochs + 1):
    if frozen and epoch > args.freeze_epochs:      # unfreeze body
        model.set_body_trainable(True)
        opt = make_opt(True)
        frozen = False
        print(f"[{args.frontend}] epoch {epoch}: unfroze BTC body", flush=True)
    tl, ta = run_epoch(train_idx, opt, True)
    vl, va = run_epoch(val_idx, opt, False)
    print(f"[{args.frontend}] ep{epoch:02d} train {tl:.3f}/{ta:.3f}  val {vl:.3f}/{va:.3f}", flush=True)
    if va > best:
        best, patience = va, 0
        torch.save({"frontend": model.frontend.state_dict(), "btc": model.btc.state_dict(),
                    "mean": float(model.mean), "std": float(model.std),
                    "n_harm": N_HARM, "frontend_kind": args.frontend, "val_acc": va}, OUT)
        print(f"[{args.frontend}]   saved -> {args.out} ({va:.4f})", flush=True)
    else:
        patience += 1
        if patience >= 5 and not frozen:
            print(f"[{args.frontend}] early stop", flush=True)
            break

print(f"[{args.frontend}] HCQT DONE best val {best:.4f} -> {args.out}", flush=True)
