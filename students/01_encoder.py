#!/usr/bin/env python3
"""
01_encoder.py — stage-by-stage walkthrough of the S-JEPA AUDIO ENCODER.

WHAT YOU WILL LEARN
-------------------
How a raw waveform becomes the per-frame representation that everything else in
S-JEPA is built on. The encoder has four stages:

    waveform (16 kHz mono)
        │  1. ConvFeatureExtractor   — 7 strided 1-D convolutions.
        │     Total stride = 5*2*2*2*2*2*2 = 320 samples = 20 ms per frame,
        │     i.e. the model "looks at" the audio in 20-ms frames, 50 per second.
        │  2. post_extract_proj      — linear map 512 -> 768 + LayerNorm.
        │  3. ConvPositionalEncoding — a depthwise conv that mixes in *relative*
        │     position (so frame t knows a bit about its neighbours).
        │  4. 6 x TransformerLayer   — the actual representation: every frame
        │     attends to every other frame (12 heads, 768-dim, 3072-dim FFN).

We replicate model.py's `OnlineEncoder.encode()` step by step (in eval mode, no
masking: every frame is "visible") and print a table of shapes and statistics
after each stage, then plot how large the per-frame features are at every layer.

USAGE
-----
    python3 01_encoder.py [--ckpt ../checkpoints/portable_step64000.pt]
                          [--audio ../../clips/13_secs.wav] [--gpu 0]

Output: a stage table on stdout, and students/figures/01_encoder_layers.png
(per-layer feature norms over time + a mean-norm bar chart).

NOTE the shapes are printed as (batch, frames, dim) or (batch, dim, frames) —
watch the transpose between the conv stage (channels-first) and the transformer
stage (channels-last).
"""
import argparse
import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# The model class lives in the official repo one level up from students/.
# (code/MATCH_NOTES.md verified: OnlineEncoder + this exact recipe strict-loads
#  both checkpoints in ../checkpoints/ — 59.9M parameters, 6-layer variant.)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))                 # .../audio/s-jepa
sys.path.insert(0, os.path.join(HERE, '..', '..', 'code'))   # pinned copy fallback

from model import OnlineEncoder  # noqa: E402

RECIPE = dict(code_dim=768, conv_dim=512, num_heads=12, ff_dim=3072,
              num_layers=6, K=500, num_experts=1)

AUDIO_ROOT = os.path.join(HERE, '..', '..')                  # .../audio


def load_wav_16k_mono(path):
    """Read a 16 kHz mono .wav with the stdlib only. -> float32 (1, 1, T)."""
    import wave
    with wave.open(path) as w:
        assert w.getframerate() == 16000, f"expected 16 kHz, got {w.getframerate()}"
        assert w.getnchannels() == 1, "expected mono"
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    # int16 -> float32 in [-1, 1): the convention the checkpoints were trained with
    x = raw.astype(np.float32) / 32768.0
    return torch.from_numpy(x)[None, None]                   # (1, 1, T_samples)


def stage_table(rows):
    """Pretty-print the (stage, shape, mean-norm, std) table."""
    w = max(len(r[0]) for r in rows)
    print(f"\n{'stage':<{w}}  {'shape':<22} {'mean ‖x‖':>10} {'std':>8}")
    print('-' * (w + 46))
    for name, shape, norm, std in rows:
        print(f"{name:<{w}}  {str(shape):<22} {norm:>10.2f} {std:>8.3f}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(AUDIO_ROOT, 'checkpoints', 'portable_step64000.pt'))
    ap.add_argument('--audio', default=os.path.join(AUDIO_ROOT, 'clips', '13_secs.wav'))
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    dev = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # ---- 1. build the model and load the EMA (target) encoder weights -------
    # 'ema_encoder' = the slow exponential-moving-average copy used to produce
    # training targets; it is the most stable snapshot of the representation.
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    enc = OnlineEncoder(**RECIPE)
    enc.load_state_dict(ck['ema_encoder'], strict=True)
    enc = enc.to(dev).float().eval()                          # eval: no dropout

    # ---- 2. load the clip ----------------------------------------------------
    wav = load_wav_16k_mono(args.audio).to(dev)               # (1, 1, T_samples)
    n_samples = wav.shape[-1]
    dur_s = n_samples / 16000.0
    print(f"clip: {os.path.basename(args.audio)} — {dur_s:.1f} s "
          f"({n_samples} samples @ 16 kHz)")

    rows = [('input waveform', tuple(wav.shape), float(wav.norm(dim=-1).mean()),
             float(wav.std()))]

    # ---- 3. stage 1: conv feature extractor ---------------------------------
    # 7 convolutions, kernel/stride (10,5) then (3,2)x3 then (2,2)x2.
    # Stride multiplies: 5*2*2*2*2*2*2 = 320 samples per output frame
    #   => 320 / 16000 Hz = 0.020 s = 20 ms per frame, ~50 frames/s ("50 fps").
    # Output is (B, channels=512, frames): channels-FIRST, like an image CNN.
    x = enc.feature_extractor(wav)                            # (1, 512, T_frames)
    T_frames = x.shape[-1]
    fps = T_frames / dur_s
    rows.append(('1. ConvFeatureExtractor (7 convs, stride 320 = 20 ms)',
                 tuple(x.shape), float(x.transpose(1, 2).norm(dim=-1).mean()),
                 float(x.std())))

    # ---- 4. stage 2: project 512 -> 768 + LayerNorm --------------------------
    # Transformers work channels-LAST: (B, frames, dim). Note the transpose.
    x = x.transpose(1, 2)                                     # (1, T, 512)
    x = enc.post_extract_proj(x)                              # (1, T, 768)
    x = enc.layer_norm(x)                                     # (1, T, 768)
    rows.append(('2. post_extract_proj 512->768 + LayerNorm',
                 tuple(x.shape), float(x.norm(dim=-1).mean()), float(x.std())))

    # ---- 5. stage 3: convolutional positional encoding -----------------------
    # A depthwise conv (kernel 128) adds a GELU-activated blend of neighbours to
    # each frame — cheap *relative* position information before attention.
    x = enc.pos_conv(x)                                       # (1, T, 768)
    rows.append(('3. ConvPositionalEncoding (relative position)',
                 tuple(x.shape), float(x.norm(dim=-1).mean()), float(x.std())))
    x = enc.encoder_layer_norm(x)

    # ---- 6. stage 4: the 6 transformer layers --------------------------------
    # Each layer: multi-head self-attention (all frames see all frames) then a
    # 3072-dim feed-forward block, both with residuals + LayerNorm.
    # We capture the output of EVERY layer to plot how the representation
    # evolves with depth (deeper layers = more abstract, usually larger norms).
    layer_feats = []
    for i, layer in enumerate(enc.layers):
        x = layer(x)                                          # (1, T, 768)
        layer_feats.append(x)
        rows.append((f'4. TransformerLayer {i}', tuple(x.shape),
                     float(x.norm(dim=-1).mean()), float(x.std())))

    print(f"\nframes: {T_frames}  (~{fps:.0f} frames/s, hop = {enc.hop} samples = "
          f"{1000 * enc.hop / 16000:.0f} ms per frame)")
    stage_table(rows)
    print("\nTakeaways:")
    print(" * The conv frontend is fixed (not trained further: feature_grad_mult)")
    print(" * 20 ms per frame, ~50 fps — every later object in S-JEPA lives on")
    print("   this time grid (masks, predictor targets, GMM assignments).")
    print(" * Per-layer norms stay ~FLAT (~26): every transformer block ends with a")
    print("   LayerNorm that standardizes the scale — the CONTENT keeps changing")
    print("   with depth even when the norm does not (see 02/03 for what that means).")
    print(" * 680 frames x 768 dims is the 'representation' JEPA predicts in —")
    print("   never the 217,920 audio samples themselves.")

    # ---- 7. figure: per-layer norms over time + mean norms -------------------
    import matplotlib
    matplotlib.use('Agg')                                     # headless-safe
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    t = np.arange(T_frames) * 20.0 / 1000.0                   # frame -> seconds
    for i, f in enumerate(layer_feats):
        norms = f[0].norm(dim=-1).cpu().numpy()
        axes[0].plot(t, norms, lw=1.0, label=f'layer {i}')
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('per-frame feature norm')
    axes[0].set_title('Representation norm over time, per encoder layer')
    axes[0].legend(fontsize=8, ncol=2)

    means = [float(f.norm(dim=-1).mean()) for f in layer_feats]
    axes[1].bar([f'L{i}' for i in range(len(means))], means, color='tab:blue')
    axes[1].set_ylabel('mean ‖feature‖')
    axes[1].set_title('Mean feature norm by depth')

    fig.suptitle(f'S-JEPA encoder stages — {os.path.basename(args.audio)}', y=1.02)
    fig.tight_layout()
    fig_dir = os.path.join(HERE, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, '01_encoder_layers.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f"\nsaved {out}")


if __name__ == '__main__':
    main()
