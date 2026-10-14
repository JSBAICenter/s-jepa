#!/usr/bin/env python3
"""
02_gmm_fitting.py — fit a Gaussian Mixture Model YOURSELF, then meet the trained one.

WHAT YOU WILL LEARN
-------------------
The "S" in S-JEPA stands for *soft clustering*: a Gaussian Mixture Model (GMM)
acts as a learned "vocabulary" over frame features. This script builds that
intuition in two halves.

PART A — hand-rolled, from scratch (the point is to see inside the black box):
  1. Log-mel features: we compute them in plain numpy (25 ms window, 10 ms hop,
     40 mel bands). No torchaudio — you can read every line.
  2. EM (Expectation-Maximization) for a *diagonal-covariance* GMM, also from
     scratch in numpy:
       E-step: responsibility r_ik = P(component k | frame i)   (a softmax over
               Gaussian log-likelihoods — exactly the "posterior" S-JEPA uses
               as its training target)
       M-step: re-estimate means / variances / mixing weights from the soft
               counts (every frame contributes fractionally to every component)
     We fit K=20 on a few clips and watch the log-likelihood converge.

PART B — the real thing:
  3. Load the checkpoint's TRAINED vocabulary (gmm_means/variances/weights,
     K=500 over the 768-d encoder features) and run its posterior on the
     encoder's layer-2 features of the same clip. Compare with Part A.

WHY TWO GMMS? Part A clusters *acoustics* (what a spectrogram looks like) with a
tiny vocabulary you built in seconds. Part B clusters *learned representations*
(what the network thinks is the same sound) with a 500-way vocabulary trained on
billions of frames (gmm_counts sums to ~6e9!). S-JEPA's targets come from Part B.

USAGE
-----
    python3 02_gmm_fitting.py [--ckpt ../checkpoints/portable_step64000.pt]
                              [--gpu 0]
Output: figures/02_em_convergence.png, figures/02_responsibilities.png,
        stdout with convergence numbers and the Part-B statistics.
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
SR = 16000

# ===========================================================================
# PART A0: log-mel features, hand-rolled (read each line — it's the classic
# pipeline: frame -> window -> spectrum -> mel filterbank -> log)
# ===========================================================================

def load_audio(path):
    """16 kHz mono float32 numpy array. wav via stdlib, flac via soundfile."""
    if path.endswith('.wav'):
        import wave
        with wave.open(path) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return x.astype(np.float32) / 32768.0
    import soundfile as sf
    x, sr = sf.read(path, dtype='float32')
    assert sr == SR and x.ndim == 1
    return x


def logmel(x, n_mels=40, win=0.025, hop=0.010):
    """Hand-rolled log-mel spectrogram -> (n_frames, n_mels)."""
    n_fft = 512
    fl, hs = int(win * SR), int(hop * SR)                    # 400 / 160 samples
    # 1. slice the waveform into overlapping frames
    n_frames = 1 + max(0, (len(x) - fl) // hs)
    frames = np.stack([x[i * hs: i * hs + fl] for i in range(n_frames)])
    # 2. window (Hamming) so frames fade in/out at the edges
    frames = frames * np.hamming(fl)
    # 3. power spectrum: |rfft|^2, keep 257 frequency bins up to 8 kHz
    spec = np.abs(np.fft.rfft(frames, n_fft)) ** 2           # (n_frames, 257)
    # 4. mel filterbank: triangular filters, evenly spaced on the MEL scale
    #    (a perceptual pitch scale — more resolution at low frequencies)
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mels = np.linspace(hz2mel(0), hz2mel(SR / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / SR).astype(int)
    fb = np.zeros((n_mels, spec.shape[1]))                   # (40, 257)
    for k in range(n_mels):
        fb[k, bins[k]:bins[k + 1]] = np.linspace(0, 1, bins[k + 1] - bins[k] + 1)[:bins[k + 1] - bins[k]]
        fb[k, bins[k + 1]:bins[k + 2]] = np.linspace(1, 0, bins[k + 2] - bins[k + 1] + 1)[1:]
    # 5. project onto the filterbank, take the log (dB-ish scale)
    return np.log(fb @ spec.T + 1e-10).T                     # (n_frames, 40)


# ===========================================================================
# PART A1: diagonal-covariance GMM via EM, from scratch
# ===========================================================================

def gmm_loglik(X, means, variances, weights):
    """Per-frame joint log p(x_i, k) for every component -> (N, K)."""
    N, D, K = X.shape[0], X.shape[1], means.shape[0]
    # Mahalanobis distance with DIAGONAL covariance: sum_d (x_d - mu_kd)^2 / var_kd
    d2 = ((X[:, None, :] - means[None, :, :]) ** 2 / variances[None, :, :]).sum(-1)
    logdet = np.log(variances).sum(-1)                       # log|Sigma_k| (diagonal)
    log_norm = -0.5 * (d2 + logdet + D * np.log(2 * np.pi))  # Gaussian log-pdf
    return log_norm + np.log(weights)[None, :]               # + log pi_k


def em_fit(X, K=20, n_iter=30, seed=0):
    """EM for a diagonal GMM. Returns (means, variances, weights, ll_history)."""
    rng = np.random.default_rng(seed)
    N, D = X.shape
    # --- init: means = K random frames, variances = global variance, weights uniform
    means = X[rng.choice(N, K, replace=False)].copy()
    variances = np.full((K, D), X.var(0) + 1e-3)
    weights = np.full(K, 1.0 / K)
    ll_hist = []
    for it in range(n_iter):
        # --- E-step: responsibilities = P(k | x_i), via a numerically safe softmax
        joint = gmm_loglik(X, means, variances, weights)     # (N, K)
        m = joint.max(1, keepdims=True)                      # log-sum-exp trick
        logZ = m + np.log(np.exp(joint - m).sum(1, keepdims=True))
        r = np.exp(joint - logZ)                             # (N, K), rows sum to 1
        ll_hist.append(float(logZ.sum()))                    # total log-likelihood
        # --- M-step: re-estimate parameters from SOFT counts
        Nk = r.sum(0) + 1e-9                                 # effective count per component
        means = (r.T @ X) / Nk[:, None]                      # weighted average frame
        # E[x^2] - E[x]^2 per component and dimension (diagonal covariance)
        variances = (r.T @ (X ** 2)) / Nk[:, None] - means ** 2 + 1e-3
        weights = Nk / Nk.sum()                              # mixture weights
    return means, variances, weights, ll_hist


def responsibilities(X, means, variances, weights):
    joint = gmm_loglik(X, means, variances, weights)
    m = joint.max(1, keepdims=True)
    return np.exp(joint - (m + np.log(np.exp(joint - m).sum(1, keepdims=True))))


# ===========================================================================
# PART B: the checkpoint's trained 500-way vocabulary over encoder features
# ===========================================================================

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(AUDIO_ROOT, 'checkpoints', 'portable_step64000.pt'))
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    dev = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    clip = os.path.join(AUDIO_ROOT, 'clips', '13_secs.wav')
    x = load_audio(clip)
    print(f"clip: 13_secs.wav — {len(x)/SR:.1f} s")

    # ---------------- PART A: your own GMM on acoustics ----------------------
    print("\n=== PART A: log-mel + hand-rolled EM, K=20 ===")
    feats13 = logmel(x)                                      # (n_frames, 40)
    print(f"log-mel features: {feats13.shape}  (100 fps: 25 ms window, 10 ms hop)")
    # A couple more clips for a slightly richer fit (LibriSpeech .flac samples)
    import glob
    extra = sorted(glob.glob(os.path.join(AUDIO_ROOT, 'clips', 'librispeech-sample', '*.flac')))[:3]
    X_fit = np.concatenate([feats13] + [logmel(load_audio(p)) for p in extra])
    print(f"fitting set: {X_fit.shape[0]} frames from {1 + len(extra)} clips")

    means, variances, weights, ll_hist = em_fit(X_fit, K=20, n_iter=30)
    print(f"EM log-likelihood: {ll_hist[0]:.1f} -> {ll_hist[-1]:.1f} over {len(ll_hist)} iters "
          f"(should rise monotonically then flatten)")
    resp13 = responsibilities(feats13, means, variances, weights)   # (n_frames, 20)
    ent = -(resp13 * np.log(resp13 + 1e-12)).sum(1).mean()
    print(f"your GMM on 13_secs.wav: mean posterior entropy {ent:.3f} nats, "
          f"{len(np.unique(resp13.argmax(1)))} distinct components used")

    # ---------------- PART B: the trained vocabulary -------------------------
    print("\n=== PART B: the checkpoint's trained GMM, K=500 over 768-d features ===")
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    g_means = ck['gmm_means'].float().to(dev)                # (500, 768)
    g_vars = ck['gmm_variances'].float().to(dev)
    g_w = ck['gmm_weights'].float().to(dev)
    print(f"trained GMM: {tuple(g_means.shape)} means; gmm_counts sum = "
          f"{ck['gmm_counts'].sum():.3e} frames of accumulated soft counts")

    enc = OnlineEncoder(**RECIPE)
    enc.load_state_dict(ck['ema_encoder'], strict=True)
    enc = enc.to(dev).float().eval()
    wav = torch.from_numpy(x)[None, None].to(dev)
    # layer 2 = the layer whose features the GMM was fit on (crispest posterior,
    # see ../../GETTING_STARTED.md section 6)
    f2 = enc.encode_layer(wav, 2)[0]                         # (T_enc, 768) @ 50 fps
    joint = gmm_loglik(f2.cpu().numpy(), g_means.cpu().numpy(),
                       g_vars.cpu().numpy(), g_w.cpu().numpy())
    m = joint.max(1, keepdims=True)
    post = np.exp(joint - (m + np.log(np.exp(joint - m).sum(1, keepdims=True))))
    ent_b = -(post * np.log(post + 1e-12)).sum(1).mean()
    print(f"trained GMM on encoder layer-2 features ({tuple(f2.shape)}): "
          f"mean posterior entropy {ent_b:.4f} nats, "
          f"{len(np.unique(post.argmax(1)))} distinct components used, "
          f"mean max-prob {post.max(1).mean():.3f}")

    print("""
CONTRAST (the teaching point):
 * Your K=20 GMM clusters ACOUSTIC frames (log-mel): 16 components used on
   this clip — silence, vowel-ish and consonant-ish spectral shapes. You built
   it in seconds from ~4.5k frames.
 * The trained K=500 GMM clusters LEARNED 768-d representations: 170 distinct
   components used on the SAME clip. Posterior entropy is similar (0.046 vs
   your 0.056) — crispness is not the difference; the SPACE and VOCABULARY
   are: 500 components in a learned space that encodes 'what the sound IS',
   not 'what the spectrogram looks like'. It was co-adapted with the encoder
   over ~6.1e9 accumulated frames: the GMM EMA-updates on the features the
   encoder produces, WHILE the encoder learns to predict the GMM's
   assignments — a chicken-and-egg loop that online EM resolves.
 * In S-JEPA, Part B's posterior IS the training target: the model learns to
   predict WHICH of the 500 components a masked frame belongs to (see
   03_cluster_head.py and 04_predictor.py).""")

    # ---------------- figures -------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(HERE, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(ll_hist, 'o-', ms=3)
    ax.set_xlabel('EM iteration'); ax.set_ylabel('total log-likelihood')
    ax.set_title('Part A: your EM fit (K=20) — monotone convergence')
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, '02_em_convergence.png'), dpi=120)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    t = np.arange(len(resp13)) * 0.010
    im = ax.imshow(resp13.T, aspect='auto', origin='lower', cmap='viridis',
                   extent=[t[0], t[-1], 0, resp13.shape[1]])
    ax.set_xlabel('time (s)'); ax.set_ylabel('component k')
    ax.set_title('Part A: per-frame responsibilities P(k|x) of YOUR K=20 GMM on 13_secs.wav')
    fig.colorbar(im, ax=ax, label='P(k|x)')
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, '02_responsibilities.png'), dpi=120)
    print(f"\nsaved figures/02_em_convergence.png and figures/02_responsibilities.png")


if __name__ == '__main__':
    main()
