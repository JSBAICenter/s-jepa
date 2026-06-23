# S-JEPA

**Self-supervised speech representation learning with a Joint-Embedding Predictive Architecture and GMM soft-target KL distillation.**

S-JEPA learns speech representations by masked prediction in a convolutional-transformer speech encoder, but instead of predicting hard cluster IDs it predicts **soft posterior distributions** from a Gaussian Mixture Model and minimizes a KL divergence. Targets are refined iteratively: an initial GMM is fit on MFCC features, then subsequent iterations fit (or continuously update) a GMM on the model's own EMA-encoded features.

---

## Table of Contents

- [Method](#method)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training Pipeline](#training-pipeline)
- [Key Hyperparameters](#key-hyperparameters)
- [Citation](#citation)

---

## Method

Training proceeds in iterations, each producing a stronger target signal for the next.

**Iteration 1 — Frozen MFCC GMM.**
A diagonal-covariance GMM is fit offline on 39-dim MFCC features (13 coefficients + Δ + ΔΔ). During training this GMM is frozen and produces soft posteriors over `K` components for every frame. The encoder is trained to match these posteriors at masked positions via KL divergence.

**Iteration 2+ — Online GMM from an EMA encoder.**
A slow exponential-moving-average copy of the encoder produces features for a chosen layer. A GMM whose parameters are updated online (as an EMA of per-batch sufficient statistics) supplies the soft targets. No gradient flows through the GMM — its means, variances, and weights drift via EMA, so the targets co-adapt with the representation without collapsing. The GMM is seeded with a fast mini-batch k-means pass over EMA features.

In both iterations the loss is the same: the masked-position KL between the predictor's cluster logits and the GMM soft posteriors. Visible-position KL is logged but not optimized.

---

## Architecture

```
                       waveform [B, 1, T]
                            │
              ConvFeatureExtractor (7 conv blocks, total stride 320)
                            │   →  ~50 Hz frame rate at 16 kHz
                   post_extract_proj → LayerNorm
                            │
                   (mask: zero out masked frames)
                            │
              ConvPositionalEncoding  →  encoder LayerNorm
                            │
              N × TransformerLayer  (pre-LN MHSA + FFN)
                            │
            ┌───────────────┴───────────────┐
            │                               │
   cluster_head(context)         JEPAPredictor fills masked
   → visible logits              frames → cluster_head → masked logits
                                            │
                                    KL( masked logits ‖ GMM posteriors )
```

Default encoder (~94M parameters): `code_dim=768`, `conv_dim=512`, `num_heads=12`, `ff_dim=3072`, `num_layers=12`.

The encoder sees zeros at masked positions; the lightweight `JEPAPredictor` reconstructs the masked frames in embedding space, and a shared MLP `cluster_head` maps both visible and predicted frames to `K` logits.

---

## Repository Structure

```
.
├── model.py                  # Shared model components (encoder, predictor, transformer)
├── fit_gmm.py                # Offline GMM fitting (MFCC for iter 1, encoder feats for iter 2)
├── train.py                  # Main training script (frozen-GMM and online-GMM modes)
├── ds_config_jepa_pretrain.json   # DeepSpeed configuration
├── requirements.txt
└── README.md
```

---

## Installation

Requires Python 3.9+ and a CUDA-capable GPU (multi-GPU recommended).

```bash
git clone https://github.com/gioannides/s-jepa.git
cd s-jepa
pip install -r requirements.txt
```

`requirements.txt`:

```
torch>=2.1
torchaudio>=2.1
deepspeed>=0.12
numpy
tqdm
```

---

## Data Preparation

The dataset is read from **JSONL manifest files**. Each line is a JSON object with a `wav_path` field:

```json
{"wav_path": "relative/or/absolute/path/to/audio.wav"}
```

Point `--jsonl` at one or more directories (comma-separated); the loader globs every `*.jsonl` inside them. Relative `wav_path` entries are resolved against a configurable base path. Audio is loaded, downmixed to mono, resampled to 16 kHz, and randomly cropped to `--max_seconds`.

On the first run, the trainer builds a single globally-shuffled `path_index.txt` in the output directory; all ranks and workers read disjoint slices of it, so the manifest is only globbed and shuffled once.

---

## Training Pipeline

### Step 1 — Fit the MFCC GMM (iteration 1 target)

```bash
CUDA_VISIBLE_DEVICES=0 python fit_gmm.py \
    --jsonl "/path/to/manifests/" \
    --out_dir ./gmm_mfcc \
    --K 100 \
    --feature_type mfcc \
    --target_frames 200000000 \
    --reservoir_size 200000000 \
    --n_iter 20
```

Produces `./gmm_mfcc/gmm.pt` containing `means`, `variances`, `weights`.

### Step 2 — Pretrain with the frozen MFCC GMM (iteration 1)

```bash
deepspeed --master_port=14446 train.py \
    --jsonl "/path/to/manifests/" \
    --out_dir ./model_iter1 \
    --ds_config ds_config_jepa_pretrain.json \
    --gmm_path ./gmm_mfcc/gmm.pt \
    --num_layers 12 \
    --batch_size 64 --lr 1e-4 \
    --p_noise 0.0 --p_mix 0.0 \
    --max_steps 100000 --save_every 1000 \
    --loop_forever --resume
```

### Step 3 — Continue with an online GMM from the EMA encoder (iteration 2+)

```bash
deepspeed --master_port=54446 train.py \
    --jsonl "/path/to/manifests/" \
    --out_dir ./model_iter2 \
    --ds_config ds_config_jepa_pretrain.json \
    --pt_ckpt ./model_iter1/ckpts/portable_step100000.pt \
    --online_gmm --reseed_gmm \
    --K 500 \
    --num_layers 12 \
    --ema_decay 1.0 --param_ema 0.999 --ema_layer 7 \
    --batch_size 64 --lr 1e-5 \
    --p_noise 0.0 --p_mix 0.0 \
    --max_steps 100000 --save_every 1000 \
    --loop_forever --resume
```

Each iteration can target a deeper encoder layer (`--ema_layer`) and/or more clusters (`--K`) as representations mature.

### Optional flags

| Flag | Purpose |
|------|---------|
| `--online_gmm` | Enable iteration-2 mode (EMA encoder + online GMM). |
| `--reseed_gmm` | On resume, re-run k-means seeding from the current EMA encoder. |
| `--auto_layer` | Auto-select the clustering layer by effective rank. |
| `--fresh_encoder` | Random-init the online encoder, seed EMA from a checkpoint. |
| `--p_noise`, `--p_mix` | Probabilities for additive-noise / utterance-mixing augmentation (`-1` randomizes). |
| `--seed_frames`, `--seed_kmeans_iters` | Control GMM k-means seeding. |

A portable, framework-agnostic checkpoint (`portable_step*.pt`) is written alongside each DeepSpeed checkpoint. It bundles the online encoder, EMA encoder, GMM parameters, and step count for clean resumption and downstream feature extraction.

---

## Key Hyperparameters

A stable recipe that has worked well in practice:

- **Iteration 1 (phase 0):** standard warmup-decay LR schedule.
- **Iteration 2+ :** constant `lr=1e-5`, `ema_decay=1.0`, `param_ema=0.999`.
- **On downstream plateau:** increase the number of GMM components (`--K`) and/or the amount of training data.

## Citation

```
@misc{ioannides2026sjepasoftclustering,
      title={S-JEPA : Soft Clustering Anchors for Self-Supervised Speech Representation Learning}, 
      author={Georgios Ioannides and Adrian Kieback and Judah Goldfeder and Linsey Pang and Aman Chadha and Aaron Elkins and Yann LeCun and Ravid Shwartz-Ziv},
      year={2026},
      eprint={2606.19398},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2606.19398}, 
}
```