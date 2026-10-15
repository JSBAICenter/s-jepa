# S-JEPA student scripts — the four core components, hands-on

These four annotated scripts walk you through **how S-JEPA actually works**, in the
order the concepts build on each other. Each one is self-contained, heavily commented,
prints a stage-by-stage story with SHAPES at every step, and saves a figure to
`figures/`. Everything runs on the lab RTX 3090 in well under a minute (the model is
only 59.9M parameters; fp32, batch-of-one clips).

```
                          16 kHz mono waveform  (B, 1, T_samples)
                                     │
        ┌────────────────────────────┴─────────────────────────────┐
        │  ENCODER  (01_encoder.py)                                │
        │  conv frontend (7 convs, stride 320 = 20 ms/frame, 50 fps)│
        │  → proj 512→768 + LayerNorm → positional conv             │
        │  → 6 transformer layers  →  per-frame features (768-d)    │
        └───────────────┬────────────────────────┬──────────────────┘
                        │                        │
             masked input (0s at masked     UNMASKED input, EMA copy
             frames, 65% policy)            ("target encoder", no grads)
                        │                        │
        ┌───────────────┴──────────┐   ┌─────────┴────────────────┐
        │  PREDICTOR  (04)         │   │  GMM vocabulary  (02)    │
        │  mask token at masked    │   │  500 Gaussians over      │
        │  positions + positions   │   │  layer-2 features        │
        │  → small transformer →   │   │  → posterior P(k|frame)  │
        │  predicted features for  │   │    = the TARGET          │
        │  the MASKED frames       │   └─────────┬────────────────┘
        └───────────────┬──────────┘             │
                        │                        │
        ┌───────────────┴──────────┐             │
        │  CLUSTER HEAD  (03)      │             │
        │  3-layer MLP → 500 logits│             │
        │  softmax = PREDICTED     │             │
        │  distribution            │             │
        └───────────────┬──────────┘             │
                        │                        │
                        └────► KL( target ‖ predicted )  ◄─────┘
                               at masked positions
                    *** THIS IS S-JEPA'S ENTIRE TRAINING LOSS ***
                (no audio is ever reconstructed — only "which
                 vocabulary items do the hidden frames belong to?")
```

## The four components in one paragraph each

- **Encoder** (`01`): turns raw audio into one 768-d vector per 20-ms frame. A fixed
  conv frontend sets the time grid (50 frames/s), then 6 transformer layers let every
  frame attend to every other frame. This is the representation everything else lives in.
- **GMM fitting / vocabulary** (`02`): a Gaussian Mixture Model is a *soft clusterer*:
  each frame gets a probability of belonging to each of K Gaussian components. You fit a
  tiny one (K=20) by hand on spectrograms to see the EM algorithm work, then meet the
  real thing: a 500-way vocabulary over the encoder's layer-2 features, trained online
  on ~6 billion frames. Its posterior is the training TARGET.
- **Cluster head** (`03`): a small MLP that maps any 768-d feature to 500 scores
  (softmax = a distribution over the vocabulary). During training it is applied to the
  predictor's outputs; the loss pulls its softmax toward the GMM posterior.
- **Predictor** (`04`): a small transformer that receives the visible frames plus a
  learned mask token at the masked positions and outputs a *predicted representation*
  for each masked frame. Feeding those through the cluster head and comparing with the
  GMM posterior via KL is the whole objective.

## Run order and what you should see

Run them from this directory (`s-jepa/students/`); all defaults point at the kit.
Expected numbers below are from the verified run on `13_secs.wav` (values will match
to ±small noise; figures land in `figures/`).

### 1. `python3 01_encoder.py`
A stage table: waveform `(1, 1, 217920)` → conv frontend `(1, 512, 680)` (680 frames
≈ 50 fps, hop 320 = 20 ms) → projection `(1, 680, 768)` → six transformer layers
`(1, 680, 768)` with per-layer norms ≈ 26 (they stay FLAT — every block ends in a
LayerNorm; the *content* changes with depth even when the norm doesn't).
Figure: `figures/01_encoder_layers.png`.

### 2. `python3 02_gmm_fitting.py`
Your hand-rolled EM (K=20 on log-mel): log-likelihood −391,040 → −308,020 over 30
iterations (monotone, then flattening); your GMM uses 16 of its 20 components on the
clip (posterior entropy 0.056 nats). The trained 500-way GMM on encoder layer-2
features: **170 distinct components on the same clip**, posterior entropy 0.046,
mean max-prob 0.982 — a working acoustic vocabulary. The script prints the contrast:
same crispness, completely different space/vocabulary/co-adaptation.
Figures: `02_em_convergence.png`, `02_responsibilities.png` (a time×component heatmap
of your GMM's posterior — you can literally see speech structure).

### 3. `python3 03_cluster_head.py`
The GMM assigns frames crisply (entropy 0.046); the head's softmax is much softer
(entropy 6.06 vs 6.21 uniform). Agreement with the vocabulary: elementwise 1.2%,
Hungarian optimal-relabeling **10.3%** vs a shuffled placebo of **7.2%** — weak but
REAL signal (see `../../GETTING_STARTED.md` §6 for why weak ≠ broken, and how the
image-port checkpoints differ: there the alignment is *zero* because the GMM gets
re-seeded). Figure: `03_cluster_vs_gmm.png` — two aligned timelines; the top strip is
the vocabulary segmenting the clip in time.

### 4. `python3 04_predictor.py`
Masking ~50% of frames (nominal 65% policy minus span overlaps — same behavior as
`train.py`'s fast mask). The predictor fills every masked frame: predicted softmax
entropy 4.74 nats; **KL(target ‖ predicted) ≈ 3.5 nats per masked frame — the number
gradient descent actually minimizes**. Bonus diagnostics: cosine between predicted
features and EMA target features ≈ 0.08 (small — features are never explicitly
matched; only the *distributions* are). Figure: `04_prediction_quality.png`.

## Requirements
`torch`, `numpy`, `matplotlib` (+ `soundfile` only if you feed `.flac` files).
GPU optional but fast; every script takes `< 1 min` on a 3090/A6000. **No cv2** (it
crashes on lab machines). Clips must be **16 kHz mono** and **≤ 15 s** (the
predictor's positional embedding covers at most 750 frames = 15 s at 50 fps).

**On the lab machines everything is already installed — two steps and you're in:**

```bash
# ONCE per account (no sudo) — wires nvcc + the kit venvs into your ~/.bashrc:
/home/restricted-share/setup/setup_user_env.sh && source ~/.bashrc

# EVERY session — activate the audio venv, then run the scripts from this dir:
sjepa-audio        # shorthand for: source /home/restricted-share/venvs/sjepa-audio/bin/activate
python3 01_encoder.py
```

(Working from home instead? See "From home" below — there you pip-install into
your own venv; the lab's setup scripts don't apply.)

## Follow-up exercises
1. In `02`, change K (10 / 50 / 100) and watch convergence speed and the number of
   "dead" components. Which K explains 4 clips best without overfitting?
2. In `02`, point the fitting set at more `../../clips/librispeech-sample/*.flac`
   (the loader handles flac) — do speaker-specific components appear?
3. In `04`, raise `MASK_RATIO` to 0.90 (edit the constant): does the predicted softmax
   flatten (higher entropy)? Does the KL grow? Connect to GETTING_STARTED §6:
   uncertainty ⇒ predict the marginal.
4. In `03`, re-run with `--ckpt ../checkpoints/portable_step61000.pt`: does the
   Hungarian accuracy move with 3k more training steps?
5. Compare layers: modify `01` to record cosine-similarity between layer-2 features
   and final-layer features of the same frame. Which pair of adjacent layers changes
   the representation most?
6. (stretch) Recompute script 03's agreement *under the training mask* (visible vs
   masked positions separately) — the head is trained only on predictor outputs, so
   clean-feature and masked-position behaviour differ.

## From home (GitHub clone — no lab access needed)

The trained checkpoint ships via **Git LFS** on the org repo **JSBAICenter/JEPA, branch
`sjepa-students`** (`checkpoints/portable_step64000.pt`, 242 MB — the later, more-trained
snapshot; the 61k one stays on the lab share). The public `JSBAICenter/s-jepa` fork carries
code + these scripts only — GitHub blocks LFS uploads to public forks. JEPA is a **private**
repo: you need org access (SSH key, or HTTPS with a personal access token). The lab's
sample clips are NOT in the repo — record your own (16 kHz mono, **<= 15 s**).

**Getting the 242 MB checkpoint — any ONE of these routes is enough:**
1. **In the clone already** — the Git LFS file downloads with the clone (org members; block below).
2. **Google Drive** (access-restricted — ask the lab to be added):
   <https://drive.google.com/drive/folders/1wYqKOMZY82I4cf5ehr0wnhPc3kToWxNM?usp=drive_link>
   — download `portable_step64000.pt` and place it at `checkpoints/portable_step64000.pt`.
3. **Lab share** — `/home/restricted-share/audio/checkpoints/` (on lab machines).

**No checkpoint yet? You can still do plenty on your laptop:**
- `python3 students/02_gmm_fitting.py --audio my_voice.wav [--audio b.wav ...]` — with the
  checkpoint absent it runs **PART A only**: the complete clustering lesson (hand-rolled
  log-mel + EM from scratch, fitted on YOUR voice). Needs no weights at all.
- `python3 students/01_encoder.py --audio my_voice.wav` (leave --ckpt pointing at a missing
  file) — falls back to a **randomly-initialized encoder**: identical architecture and
  stage-by-stage shapes (the structure walkthrough); values clearly labeled as meaningless.
- Read the paper — **`paper/sjepa_audio.pdf`** is in this repo (plus the follow-up
  `paper/sjepa_followup_component_mapping.pdf` on the component-mapping problem our
  coherence checks probe).
- Guided code reading, no GPU needed: in `model.py` find (a) the conv that sets the 50 fps
  frame rate, (b) where the predictor's mask token replaces masked frames, (c) the cluster
  head; in `train.py` find the KL loss line and the EMA update loop.

```bash
git clone -b sjepa-students git@github.com:JSBAICenter/JEPA.git s-jepa
cd s-jepa                                 # the LFS checkpoint downloads with the clone
# if you cloned with LFS skipped:  git lfs install && git lfs pull
pip install torch scipy matplotlib sounddevice   # CPU-only is fine; scripts auto-fallback

# record ~10 s of your voice:
python3 students/record_audio.py --seconds 10 --out my_voice.wav
#  ...or with ffmpeg:  macOS: ffmpeg -f avfoundation -i ":0" -t 10 -ar 16000 -ac 1 my_voice.wav
#                      Linux: ffmpeg -f pulse -i default -t 10 -ar 16000 -ac 1 my_voice.wav
#                      phone: ffmpeg -i voice.m4a -ar 16000 -ac 1 my_voice.wav

python3 students/01_encoder.py     --ckpt checkpoints/portable_step64000.pt --audio my_voice.wav
python3 students/02_gmm_fitting.py --ckpt checkpoints/portable_step64000.pt --audio my_voice.wav [--audio more.wav ...]
python3 students/03_cluster_head.py --ckpt checkpoints/portable_step64000.pt --audio my_voice.wav
python3 students/04_predictor.py   --ckpt checkpoints/portable_step64000.pt --audio my_voice.wav
```

(02 accepts repeated `--audio` flags — give it 2-4 clips for a richer EM fit; at home the
fit set defaults to just the clips you pass. Figures land in `students/figures/`.)
If your clone has no LFS data (`checkpoints/*.pt` are ~130-byte pointer files), run
`git lfs install && git lfs pull`. If LFS bandwidth is exhausted (free-tier quota),
ask the lab to enable a data pack or use the lab machines.

## Pointers
- `../../GETTING_STARTED.md` — the kit's main guide (concepts, inventory, §6 coherence
  story, more exercises).
- `../../paper/sjepa_audio.pdf` — the S-JEPA paper; read the method section after
  running 01–04.
- `../model.py`, `../train.py`, `../fit_gmm.py` — the real training code (read-only):
  every construct in these scripts is a commented replica of a piece of it.
- `../../examples/load_audio_sjepa.py` — the verified minimal loader these scripts
  borrow from.
- `paper/sjepa_audio.pdf` + `paper/sjepa_followup_component_mapping.pdf` — the method
  paper and the component-mapping follow-up (both in this repo).

## Data provenance

- `clips/librispeech-sample/` — **LibriSpeech** (Panayotov et al., ICASSP 2015;
  OpenSLR SLR12, https://openslr.org/12, CC BY 4.0), sampled from the lab's local copy.
- `clips/13_secs.wav`, `2mins.wav` — S-JEPA project test clips (author-provided);
  `2mins.wav` is misnamed (~13 s, same as `13_secs.wav`).
- Checkpoints `portable_step61000/64000.pt` — released S-JEPA (audio) training
  snapshots from the S-JEPA author; paper: arXiv 2606.19398.
