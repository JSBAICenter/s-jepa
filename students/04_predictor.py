#!/usr/bin/env python3
"""
04_predictor.py — the PREDICTOR HEAD and the actual S-JEPA training objective.

WHAT YOU WILL LEARN
-------------------
This is the heart of S-JEPA. Everything the model learns comes from ONE loss
computed in this script:

  1. Mask ~65% of the 20-ms frames (training policy: random 10-frame spans,
     same as train.py's compute_mask_indices).
  2. The ONLINE encoder encodes the masked input (masked frames are zeroed).
  3. The PREDICTOR — a small transformer — receives the visible frames plus a
     learned MASK TOKEN at every masked position, and outputs a *predicted
     representation* for each masked frame (it never sees audio there).
  4. The cluster head turns those predictions into vocabulary logits.
  5. TARGET: the EMA (target) encoder encodes the SAME clip UNMASKED; its
     layer-2 features go through the frozen GMM -> a posterior over the 500
     components for every frame.
  6. LOSS = KL( target_posterior || predicted_softmax ) at masked positions.
     That's it. No audio is ever reconstructed; the model is graded only on
     'WHICH vocabulary items should the hidden frames belong to?'

We also measure how close the predicted FEATURES get to the target features
(per-frame cosine similarity) — a bonus diagnostic; the KL is the real
objective (the predictor is never explicitly trained to match features).

USAGE
-----
    python3 04_predictor.py [--ckpt ../checkpoints/portable_step64000.pt]
                            [--audio ../../clips/13_secs.wav] [--gpu 0]
Output: stdout numbers + figures/04_prediction_quality.png
NOTE: clips must be <= 15 s (the predictor's positional embedding covers at
most 750 frames = 15 s at 50 fps).
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'code'))
from model import OnlineEncoder  # noqa: E402

RECIPE = dict(code_dim=768, conv_dim=512, num_heads=12, ff_dim=3072,
              num_layers=6, K=500, num_experts=1)
AUDIO_ROOT = os.path.join(HERE, '..', '..')
GMM_LAYER = 2          # the layer whose features the GMM vocabulary lives on
MASK_RATIO = 0.65      # training policy (train.py default)
MASK_LENGTH = 10       # frames per span (train.py default)


def load_wav_16k_mono(path):
    import wave
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return torch.from_numpy(x.astype(np.float32) / 32768.0)[None, None]


def make_span_mask(T, ratio=MASK_RATIO, length=MASK_LENGTH, seed=0):
    """Replica of train.py's compute_mask_indices (span version), documented:
    num_spans = T * ratio / length random spans of `length` frames each.
    Returns a BOOL mask where True = MASKED. train.py then converts it to the
    model's float convention with mask = (~mask_bool).float()  (1=visible)."""
    g = torch.Generator().manual_seed(seed)
    num_spans = max(1, int(T * ratio / length))
    starts = torch.randint(0, max(1, T - length), (num_spans,), generator=g)
    m = torch.zeros(T, dtype=torch.bool)
    for s in starts.tolist():
        m[s:s + length] = True
    return m


def gmm_posterior(feats, means, variances, weights):
    D = feats.shape[-1]
    d2 = ((feats[:, None, :] - means[None]) ** 2 / variances[None]).sum(-1)
    logdet = torch.log(variances).sum(-1)
    joint = -0.5 * (d2 + logdet + D * np.log(2 * np.pi)) + torch.log(weights.clamp(min=1e-12))
    return joint.softmax(-1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(AUDIO_ROOT, 'checkpoints', 'portable_step64000.pt'))
    ap.add_argument('--audio', default=os.path.join(AUDIO_ROOT, 'clips', '13_secs.wav'))
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    dev = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)

    # ---- TWO encoders: the student (online) and the teacher (EMA) -----------
    online = OnlineEncoder(**RECIPE)
    online.load_state_dict(ck['online'], strict=True)        # trained by gradients
    online = online.to(dev).float().eval()

    target = OnlineEncoder(**RECIPE)
    target.load_state_dict(ck['ema_encoder'], strict=True)   # slow EMA copy —
    target = target.to(dev).float().eval()                   # produces targets

    wav = load_wav_16k_mono(args.audio).to(dev)
    dur = wav.shape[-1] / 16000
    assert dur <= 15.0, "predictor pos_embed supports at most 750 frames (15 s)"

    # ---- 1. training-style mask ----------------------------------------------
    # The TRUE frame count comes from the conv frontend itself (its exact
    # arithmetic can differ by one from len/hop); spans are laid on that grid.
    T = online.feature_extractor(wav).shape[-1]              # (1, 512, T)
    mask_bool = make_span_mask(T, seed=0).to(dev)            # (T,) True = MASKED
    print(f"clip: {os.path.basename(args.audio)} — {dur:.1f} s, {T} frames; "
          f"masking ~{MASK_RATIO:.0%} in {MASK_LENGTH}-frame (200 ms) spans")

    # ---- 2+3. online encoder on the masked input, then the predictor --------
    # Model convention: mask=1 means VISIBLE, 0 means MASKED (train.py flips
    # the bool mask: mask = (~mask_bool).float()). encode() ZEROES masked
    # frames; the predictor then overwrites them with its mask token.
    mask_float = (~mask_bool).float()[None]                  # (1, T) 1=visible
    ctx, _ = online.encode(wav, mask=mask_float)             # ctx: (1, C, T)
    T = ctx.shape[-1]
    assert mask_bool.shape[0] == T
    # per-sample index lists, exactly like OnlineEncoder.forward() builds them
    vis_idx_list = [mask_float[b].bool().nonzero(as_tuple=True)[0] for b in range(1)]
    mask_idx_list = [(~mask_float[b].bool()).nonzero(as_tuple=True)[0] for b in range(1)]
    mpos = mask_bool                                         # (T,) True = MASKED
    n_masked = int(mpos.sum().item())
    # NOTE: the nominal policy is 65%, but random 10-frame spans OVERLAP, so
    # the realized coverage is lower (~50%). That is exactly what train.py's
    # fast compute_mask_indices does too — faithful to the real training run.
    print(f"visible: {T - n_masked}  masked: {n_masked} ({n_masked / T:.0%}; "
          f"nominal policy {MASK_RATIO:.0%} minus span overlaps)")

    # The predictor sees: visible features at visible positions, the shared
    # MASK TOKEN (+ positional embedding) at masked positions.
    z_pred = online.predictor(ctx, vis_idx_list, mask_idx_list, T)
    # z_pred: (1, C, T) — nonzero ONLY at masked positions (by construction)
    assert z_pred.shape == (1, 768, T)

    # ---- 4. predicted vocabulary distribution --------------------------------
    logits_mask = online.cluster_head(z_pred.permute(0, 2, 1))   # (1, T, 500)
    pred_sm = logits_mask.softmax(-1)
    pred_ent = -(pred_sm * pred_sm.clamp(min=1e-12).log()).sum(-1).mean().item()
    print(f"predicted softmax entropy at masked positions: {pred_ent:.3f} nats")

    # ---- 5. the target: EMA encoder on the UNMASKED clip -> GMM posterior ---
    tgt_feats = target.encode_layer(wav, GMM_LAYER)[0]           # (T, 768)
    post = gmm_posterior(tgt_feats, ck['gmm_means'].float().to(dev),
                         ck['gmm_variances'].float().to(dev),
                         ck['gmm_weights'].float().to(dev))      # (T, 500)
    tgt_ent = -(post * post.clamp(min=1e-12).log()).sum(-1).mean().item()
    print(f"target posterior entropy (unmasked EMA layer {GMM_LAYER}): {tgt_ent:.4f} nats")

    # ---- 6. THE LOSS (identical to train.py's masked-position KL) -----------
    assert logits_mask.shape == (1, T, 500) and post.shape == (T, 500)
    log_pred = F.log_softmax(logits_mask[0][mpos], dim=-1)       # (n_masked, 500)
    tgt = post[mpos]                                             # (n_masked, 500)
    assert log_pred.shape == tgt.shape and log_pred.shape[0] == n_masked
    # F.kl_div(input=log q, target=p) == KL(p || q), reduction='batchmean'
    # divides by the batch (rows) -> this prints the MEAN PER-FRAME KL.
    loss = F.kl_div(log_pred, tgt, reduction='batchmean').item()
    print(f"\nKL( target || predicted ), mean per masked frame = {loss:.4f} nats"
          f"   <-- THIS IS S-JEPA'S ENTIRE TRAINING LOSS")

    # ---- bonus diagnostic: how close are the predicted FEATURES? ------------
    # z_pred lives in the FINAL-layer feature space (the predictor predicts
    # from final-layer context); the GMM target lives at layer 2. We compare
    # against both EMA spaces so you can see the difference.
    pred_f = z_pred[0][:, mpos].T                               # (n_masked, 768)
    tgt_l2 = tgt_feats[mpos]                                    # (n_masked, 768)
    tgt_fin = target.encode_layer(wav, len(online.layers) - 1)[0][mpos]
    assert pred_f.shape == tgt_l2.shape == (n_masked, 768)
    cos_l2 = F.cosine_similarity(pred_f, tgt_l2, dim=-1)         # (n_masked,)
    cos_fin = F.cosine_similarity(pred_f, tgt_fin, dim=-1)       # (n_masked,)
    print(f"per-masked-frame cosine(pred features, EMA layer-{GMM_LAYER} features): "
          f"{cos_l2.mean():.4f}")
    print(f"per-masked-frame cosine(pred features, EMA FINAL features):     "
          f"{cos_fin.mean():.4f}")

    print("""
READING THE NUMBERS:
 * The KL is the only thing gradient descent ever minimized. Predicting a
   500-way distribution — not reconstructing 20 ms of audio — is what forces
   the encoder to learn 'what is being said' rather than 'what it sounds like'.
 * The cosine similarities are DIAGNOSTICS, not objectives. If the KL were
   driven near zero, predicted and target distributions would match; feature
   cosine could still be modest, because many different features can produce
   the same vocabulary distribution.
 * Compare with 03_cluster_head.py: there the head read CLEAN features; here
   it reads the PREDICTOR's outputs — the input distribution it was actually
   trained on. Try raising MASK_RATIO (edit the constant) to 0.9 and watch the
   predicted softmax flatten toward the marginal (uncertainty!), exactly the
   behaviour discussed in ../../GETTING_STARTED.md section 6.""")

    # ---- figure ---------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.4))
    t_masked = mpos.cpu().nonzero().flatten().numpy() * 0.020   # masked frames, s
    assert len(t_masked) == n_masked
    ax.plot(t_masked, cos_l2.cpu().numpy(), 'o-', ms=2, lw=0.8,
            label=f'cos vs EMA layer-{GMM_LAYER} (target space)')
    ax.plot(t_masked, cos_fin.cpu().numpy(), 'o-', ms=2, lw=0.8, alpha=0.6,
            label='cos vs EMA final layer (predictor input space)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('time (s) — masked frames only')
    ax.set_ylabel('cosine similarity')
    ax.set_title(f'Predictor output vs EMA target features — {os.path.basename(args.audio)} '
                 f'(mean KL={loss:.2f} nats/frame, {n_masked} masked frames)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_dir = os.path.join(HERE, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, '04_prediction_quality.png')
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == '__main__':
    main()
