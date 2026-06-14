# Extending BTC-HCQT: a concrete build guide (chords → melody, bass, transcription)

> **DRAFT — for owner review.** Ships in the public GitHub repo.

## What this is

The published **BTC** large-vocabulary chord model (Park et al., ISMIR 2019) with a
**Harmonic-CQT (HCQT)** front-end instead of the standard CQT. On chord recognition it
**ties** baseline BTC on held-out public benchmarks (`BENCHMARK.md`). Not a more accurate
chord model — a chord-solid one built on a representation that's *also* the right substrate
for pitch-level tasks. Below is the **actual recipe** to take it there: not "you could," but
"here is exactly what we'd do, what we'd use, and why."

## Read this first (honest)

Everything below is a **reasoned, runnable plan — not measured results.** We haven't trained
these heads. And purpose-built tools exist for notes-from-audio: **basic-pitch** (Spotify),
**MT3** (Google), **Omnizart** (a unified MIR toolkit) — for pure transcription they'll beat
a from-here build. This guide is for someone who wants **chords *and* pitch-level output from
one chord-aware base on a shared representation**, and would rather start from a tied-with-BTC
model with a good harmonic front-end already in place than wire it all up themselves.

## Why HCQT is the right substrate

HCQT stacks several CQTs at harmonic multiples of each bin (`HARMONICS = [0.5, 1, 2, 3, 4, 5]×`
in `hcqt.py`) so a note's overtones align along a harmonic axis — exactly what multi-pitch and
melody estimation need. It's the representation introduced for "deep salience" f0 estimation
(Bittner et al., 2017). The front-end already computes the features pitch tasks want; the chord
head just doesn't read them out that way. A new head does.

---

## The build, step by step — worked example: melody → MIDI

This is precisely what I'd do if you asked me to build it.

### 1. Define the output
A per-frame **pitch-salience map**: rows = frames, columns = a quantized pitch grid (the HCQT's
own frequency bins, or 1–3 bins/semitone over ~MIDI 36–84), **sigmoid**-activated, plus a 1-D
**voicing** signal (is the melody sounding this frame). *Why:* this is the proven deep-salience
target — the same paper HCQT comes from — and it decodes cleanly to notes. It's also what
basic-pitch learns, so we know it works.

### 2. Pick the data (and why)
- **GuitarSet** — note-level annotations, license-clean (CC-BY), **already in our pipeline**.
  Best first cut: clean license + we already have the loaders.
- **MedleyDB-Melody / MDB-stem-synth** — melody f0 on real multitrack audio (check per-track
  license; some are non-commercial).
- **MAESTRO** — piano, perfectly MIDI-aligned, large — for the polyphonic variant later.

  *Why these:* each gives time-aligned f0/note truth at the resolution a salience head needs;
  GuitarSet keeps the first run license-clean and reuses code we already wrote.

### 3. Build the targets
Convert each track's f0/note annotations into the salience matrix **at the HCQT's frame rate**
(reuse our hop / `timestep`), placing a small **Gaussian bump** around each true pitch bin.
*Why the blur:* soft targets train far more stably than hard one-hots and match the
deep-salience recipe.

### 4. Add the head (architecture)
Leave the HCQT front-end and the BTC chord body **untouched**. Tap the HCQT feature tensor (or
the front-end's output) into a **small 2-D conv stack (3–4 layers) → 1×1 conv → sigmoid**,
producing `(frames × pitch_bins)`. *Why conv + small:* pitch salience is local in
time–frequency; basic-pitch proves a tiny conv net suffices. It hangs **in parallel** to the
chord head — one backbone, two readouts.

### 5. Train it (regime)
- **Stage 1 — frozen:** freeze front-end + chord body, train **only** the new head. Loss =
  BCE on the salience map + BCE on voicing. *Why:* the front-end is already harmonically
  meaningful and the chord body is precious — don't disturb them while the head learns.
- **Stage 2 — optional joint fine-tune:** unfreeze the front-end at a **low LR**, but keep the
  chord head in the loss and **replay chord data**. *Why the replay:* unfrozen training without
  it is exactly what caused our chord **catastrophic forgetting** earlier (Beatles 80.6→73.2).
  Don't repeat that mistake.

### 6. How we'd actually run it
- Precompute HCQT features for the note dataset with the existing `hcqt.py` extractor (same
  params as the chord model).
- Adapt `hcqt_finetune.py` — same windowing/batching over `timestep` frames — swap the chord
  loss for the salience BCE and point it at the note targets. New flag, new head, same scaffold.
- Hardware: one RTX 3090/4090; Stage 1 is a few hours (small head, frozen backbone).
- Save the checkpoint in the same format as the chord model so the repo can load either head.

### 7. Decode salience → MIDI
Salience map → **peak-pick** per frame (argmax for mono melody; threshold-all for poly) →
**Viterbi / HMM smoothing** for pitch continuity + a voicing threshold → segment the runs into
**note on/off events** → write MIDI with `pretty_midi`. *Why Viterbi:* it kills frame-to-frame
flicker and enforces realistic note continuity.

### 8. Evaluate honestly
mir_eval again — but the **melody metrics**: Raw Pitch Accuracy, Raw Chroma Accuracy, Overall
Accuracy, on a held-out split (e.g. MedleyDB-Melody). **Baseline = basic-pitch**, reported
straight: it's strong and may win. The honest contribution is "chords + melody from one base,"
not "best melody extractor."

---

## Variations (same recipe, small deltas)

- **Bass line:** restrict the pitch grid to the bass register and lean on HCQT's **0.5×
  sub-harmonic**; data = MedleyDB bass stems. Everything else is identical.
- **Polyphonic transcription:** don't argmax — threshold all peaks, **add an onset head**
  (basic-pitch style), decode to a piano-roll; data = MAESTRO / MusicNet / Slakh.
- **Bass note / slash chords:** combine the bass head's pitch with the chord head → completer
  chord symbols (e.g. `C/E`) than BTC's labels give on their own.

## One model, one pass (the payoff)

Once each head works frozen, train **multi-task** — chords + melody (+ bass) — summing weighted
losses off the shared HCQT backbone, with chord-data replay to hold accuracy. A single forward
pass then yields chords *and* notes. That's the honest "does more than chord-only BTC."

## Honest scorecard
- ✅ Reuses code already in this repo (`hcqt.py`, `hcqt_finetune.py`, `bench_score.py`).
- ✅ Built on the representation pitch tasks are designed around.
- ⚠️ Unbuilt — a reasoned, runnable plan, not measured results.
- ⚠️ For pure transcription, basic-pitch / MT3 / Omnizart will likely beat a from-here build.
  Start here only if you want chords + pitch from one chord-aware base.
