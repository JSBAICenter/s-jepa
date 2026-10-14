#!/usr/bin/env python3
"""
03_cluster_head.py — the CLUSTER HEAD: reading out the vocabulary.

WHAT YOU WILL LEARN
-------------------
The cluster head is a small 3-layer MLP (768 -> 768 -> 768 -> 500) that maps a
frame's features to a score for every one of the 500 vocabulary components.
SOFTMAX of those scores is the model's *predicted* posterior over the
vocabulary. During training this head is applied to the PREDICTOR's outputs at
masked positions, and its softmax is pulled toward the GMM posterior computed
on the EMA encoder's unmasked features with a KL-divergence loss (train.py:
F.kl_div(log_pred_mask, target_mask)). So BY CONSTRUCTION, a fully-trained,
fully-successful head should agree with the GMM's assignments.

This script MEASURES that agreement on real data and shows the honest result:
in these mid-training checkpoints the head is only WEAKLY aligned with the
vocabulary (see ../../GETTING_STARTED.md section 6 for why that can be
expected, not just "broken"). You will see:

  1. an aligned timeline: GMM argmax (top strip) vs cluster-head argmax
     (bottom strip) over the clip — the vocabulary segments the audio in time;
  2. elementwise agreement (strict same-index match);
  3. Hungarian OPTIMAL-RELABELING accuracy — "if you were allowed to rename
     every head output to whichever vocabulary component fits best, how well
     would they match?" — with a SHUFFLED PLACEBO baseline so you can tell a
     real signal from noise. (If the head used a different component ORDERING
     than the GMM, elementwise agreement would be ~0 but Hungarian would be
     high — that's why we compute both.)

USAGE
-----
    python3 03_cluster_head.py [--ckpt ../checkpoints/portable_step64000.pt]
                               [--audio ../../clips/13_secs.wav] [--gpu 0]
Output: stdout numbers + figures/03_cluster_vs_gmm.png
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'code'))
from model import OnlineEncoder  # noqa: E402

RECIPE = dict(code_dim=768, conv_dim=512, num_heads=12, ff_dim=3072,
              num_layers=6, K=500, num_experts=1)
AUDIO_ROOT = os.path.join(HERE, '..', '..')
GMM_LAYER = 2          # the layer the GMM was fit on (crispest posterior —
                       # verified in ../../GETTING_STARTED.md section 3/6)


def load_wav_16k_mono(path):
    import wave
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return torch.from_numpy(x.astype(np.float32) / 32768.0)[None, None]


def gmm_posterior(feats, means, variances, weights):
    """Diagonal-GMM posterior (T, K) — same math as the hand-rolled version of
    02_gmm_fitting.py, just on torch tensors."""
    D = feats.shape[-1]
    d2 = ((feats[:, None, :] - means[None]) ** 2 / variances[None]).sum(-1)
    logdet = torch.log(variances).sum(-1)
    joint = -0.5 * (d2 + logdet + D * np.log(2 * np.pi)) + torch.log(weights.clamp(min=1e-12))
    return joint.softmax(-1)


def hungarian_acc(a, b, K):
    """Best-possible one-to-one relabeling accuracy between label vectors
    a (head) and b (GMM): solve a maximum-weight bipartite matching on the
    confusion histogram. This is PERMUTATION-INVARIANT: it would stay high
    even if the head used a completely different component numbering."""
    C = np.zeros((K, K), dtype=np.int64)
    np.add.at(C, (a, b), 1)
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-C)
    return C[r, c].sum() / len(a)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(AUDIO_ROOT, 'checkpoints', 'portable_step64000.pt'))
    ap.add_argument('--audio', default=os.path.join(AUDIO_ROOT, 'clips', '13_secs.wav'))
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    dev = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    enc = OnlineEncoder(**RECIPE)
    enc.load_state_dict(ck['ema_encoder'], strict=True)
    enc = enc.to(dev).float().eval()
    K = RECIPE['K']

    wav = load_wav_16k_mono(args.audio).to(dev)
    # T_frames from the network itself (the conv stack's exact arithmetic):
    # 7 convs with stride product 320 => about (and not always exactly) len/320.
    T_est = wav.shape[-1] // enc.hop
    print(f"clip: {os.path.basename(args.audio)} — {wav.shape[-1]/16000:.1f} s, ~{T_est} frames")

    # ---- the GMM posterior: the "ground truth" vocabulary assignment --------
    feats_gmmlayer = enc.encode_layer(wav, GMM_LAYER)[0]     # (T, 768)
    T = feats_gmmlayer.shape[0]                              # exact frame count
    post = gmm_posterior(feats_gmmlayer, ck['gmm_means'].float().to(dev),
                         ck['gmm_variances'].float().to(dev),
                         ck['gmm_weights'].float().to(dev))
    gmm_arg = post.argmax(-1).cpu().numpy()
    gmm_ent = -(post * post.clamp(min=1e-12).log()).sum(-1).mean().item()
    print(f"GMM posterior @layer {GMM_LAYER}: entropy {gmm_ent:.4f} nats, "
          f"{len(np.unique(gmm_arg))} distinct components on this clip")

    # ---- the cluster head's predicted assignment -----------------------------
    # NOTE: the head was trained on PREDICTOR outputs at masked positions, so
    # applying it to clean encoder features is deliberately "off-distribution"
    # — one of the reasons its output stays soft (GETTING_STARTED section 6).
    feats_final = enc.encode(wav, torch.ones(1, T, device=dev))[0].permute(0, 2, 1)[0]
    logits = enc.cluster_head(feats_final)                   # (T, 500)
    sm = logits.softmax(-1)
    head_arg = logits.argmax(-1).cpu().numpy()
    head_ent = -(sm * sm.clamp(min=1e-12).log()).sum(-1).mean().item()
    print(f"cluster_head on final-layer features: softmax entropy {head_ent:.4f} nats "
          f"(uniform over {K} would be {np.log(K):.2f}), "
          f"{len(np.unique(head_arg))} distinct outputs used")

    # ---- the three agreement measures ----------------------------------------
    agree = float((head_arg == gmm_arg).mean())
    hung = hungarian_acc(head_arg, gmm_arg, K)
    rng = np.random.default_rng(0)
    placebo = hungarian_acc(head_arg, gmm_arg[rng.permutation(len(gmm_arg))], K)
    print(f"\nelementwise agreement (same component index): {agree:.4f}"
          f"   [chance ~ {1/len(np.unique(gmm_arg)):.4f}]")
    print(f"Hungarian optimal-relabeling accuracy:        {hung:.4f}")
    print(f"shuffled PLACEBO (same statistic, no signal): {placebo:.4f}")
    print(f"signal above placebo: {hung - placebo:+.4f}")

    print("""
HOW TO READ THIS (see ../../GETTING_STARTED.md section 6 for the full story):
 * The GMM assigns frames crisply (entropy ~0.04) — a working 500-way acoustic
   vocabulary segmenting the clip in time.
 * The head's distribution is much softer (entropy ~6.0 vs 6.21 uniform) and
   agrees with the GMM only weakly — BUT measurably above chance/placebo.
 * WHY weak is not necessarily 'broken':
     1. KL under irreducible uncertainty: if visible context cannot determine
        a masked frame's cluster, the KL-optimal prediction is the soft
        CONDITIONAL MARGINAL, not a confident guess.
     2. The head is only ever trained on the PREDICTOR's outputs (the
        visible-position KL is logged, never optimized) — so head(clean
        features) is off-distribution by design.
     3. These checkpoints are mid-training (61-64k of the 100k-step recipe).
 * CONTRAST (worth reading): in the image-port checkpoints
   (/home/restricted-share/JSBCAI_I_JEPA/ANALYSIS_step1540000.md) periodic GMM
   re-seeding destroys component identity and the head-GMM alignment is ZERO.
   Here the vocabulary is continuously EMA-updated (gmm_counts ~6e9) and the
   weak-but-real signal survives.""")

    # ---- figure: aligned timelines -------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True)
    t = np.arange(T) * 0.020
    axes[0].scatter(t, gmm_arg, s=3, c=gmm_arg, cmap='turbo')
    axes[0].set_ylabel('GMM component\n(argmax of posterior)')
    axes[0].set_title(f'The vocabulary in time: GMM assignment vs cluster-head prediction — '
                      f'{os.path.basename(args.audio)}')
    axes[1].scatter(t, head_arg, s=3, c=head_arg, cmap='turbo')
    axes[1].set_ylabel('head component\n(argmax of softmax)')
    axes[1].set_xlabel('time (s)')
    for ax in axes:
        ax.set_ylim(-1, K)
    fig.tight_layout()
    fig_dir = os.path.join(HERE, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, '03_cluster_vs_gmm.png')
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == '__main__':
    main()
