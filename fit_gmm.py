#!/usr/bin/env python3
"""
Diagonal-covariance GMM fitting for speech SSL.

Iteration 1: GMM on 39-dim MFCC features (13 coefficients + delta + delta-delta)
Iteration 2: GMM on 768-dim encoder layer representations

Produces a .pt file with GMM parameters (means, variances, weights)
that the training script uses for soft posterior assignment.

Usage:
    # Iteration 1: fit GMM on MFCC+deltas, K=1024
    CUDA_VISIBLE_DEVICES=0 python fit_gmm.py \
        --jsonl "/path/to/checkpoints/" \
        --out_dir ./gmm_mfcc \
        --K 1024 \
        --feature_type mfcc \
        --target_frames 500000000

    # Iteration 2: fit GMM on encoder layer, K=2048
    CUDA_VISIBLE_DEVICES=0 python fit_gmm.py \
        --jsonl "/path/to/checkpoints/" \
        --out_dir ./gmm_iter2 \
        --K 2048 \
        --feature_type encoder \
        --encoder_ckpt ./model/ckpts/portable_step100000.pt \
        --cluster_layer 9 \
        --target_frames 500000000
"""

import os, json, argparse, random, glob, hashlib

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, IterableDataset


def stable_hash(s):
    return hashlib.md5(s.encode()).hexdigest()


# ============================================================
# Model components for encoder feature extraction
# ============================================================

class Fp32GroupNorm(nn.GroupNorm):
    def forward(self, x):
        return F.group_norm(
            x.float(), self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None, self.eps
        ).to(x.dtype)

class Fp32LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(
            x.float(), self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None, self.eps
        ).to(x.dtype)

class ConvFeatureExtractor(nn.Module):
    def __init__(self, conv_dim=512):
        super().__init__()
        layer_configs = [
            (conv_dim, 10, 5), (conv_dim, 3, 2), (conv_dim, 3, 2),
            (conv_dim, 3, 2), (conv_dim, 3, 2), (conv_dim, 2, 2), (conv_dim, 2, 2),
        ]
        self.conv_layers = nn.ModuleList()
        in_ch = 1
        for i, (out_ch, k, s) in enumerate(layer_configs):
            if i == 0:
                self.conv_layers.append(nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, k, s, bias=False), Fp32GroupNorm(1, out_ch), nn.GELU()))
            else:
                self.conv_layers.append(nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, k, s, bias=False), nn.GELU()))
            in_ch = out_ch

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
        return x

class ConvPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, kernel_size=128, groups=16):
        super().__init__()
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=kernel_size // 2, groups=groups)
        self.num_remove = 1 if kernel_size % 2 == 0 else 0
        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x):
        x_conv = self.conv(x.transpose(1, 2))
        if self.num_remove > 0:
            x_conv = x_conv[:, :, :-self.num_remove]
        return x + F.gelu(x_conv).transpose(1, 2)

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x
    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, ff_dim=3072,
                 dropout=0.0, attention_dropout=0.0, activation_dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.self_attn_layer_norm = Fp32LayerNorm(embed_dim)
        self.final_layer_norm = Fp32LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, T, C = x.shape
        residual = x
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attention_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        x = self.self_attn_layer_norm(residual + out)
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.activation_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.final_layer_norm(residual + x)
        return x

class EncoderForExtraction(nn.Module):
    """Minimal encoder for feature extraction — no cluster head, no predictor."""
    def __init__(self, code_dim=768, conv_dim=512, num_heads=12, ff_dim=3072, num_layers=12,
                 conv_pos_kernel=128, conv_pos_groups=16):
        super().__init__()
        self.feature_extractor = ConvFeatureExtractor(conv_dim)
        self.post_extract_proj = nn.Linear(conv_dim, code_dim)
        self.layer_norm = Fp32LayerNorm(code_dim)
        self.pos_conv = ConvPositionalEncoding(code_dim, conv_pos_kernel, conv_pos_groups)
        self.encoder_layer_norm = Fp32LayerNorm(code_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(code_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

    @torch.no_grad()
    def encode_layerwise(self, wav):
        x = self.feature_extractor(wav)
        x = x.transpose(1, 2)
        x = self.post_extract_proj(x)
        x = self.layer_norm(x)
        x = self.pos_conv(x)
        x = self.encoder_layer_norm(x)
        layer_outputs = []
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)  # [B, T, C]
        return layer_outputs


# ============================================================
# Dataset
# ============================================================

class StreamingDataset(IterableDataset):
    def __init__(self, root, sr=16000, max_sec=15.0, base_path="/scratch/gioannides/granary_data"):
        self.root = root
        self.sr = sr
        self.max_samples = int(sr * max_sec)
        self.base_path = base_path
        self._all_paths = []
        self._preload_all_paths()

    def _preload_all_paths(self):
        all_paths = []
        for root_dir in self.root.split(','):
            root_dir = root_dir.strip()
            for jf in sorted(glob.glob(os.path.join(root_dir, "*.jsonl"))):
                try:
                    with open(jf) as f:
                        for line in f:
                            try:
                                obj = json.loads(line.strip())
                                wp = obj["wav_path"]
                                if not wp.startswith('/'):
                                    wp = os.path.join(self.base_path, wp)
                                all_paths.append(wp)
                            except:
                                continue
                except:
                    continue
        random.shuffle(all_paths)
        self._all_paths = all_paths
        print(f"[Dataset] {len(all_paths)} paths")

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker:
            paths = self._all_paths[worker.id::worker.num_workers]
        else:
            paths = self._all_paths
        for wp in paths:
            try:
                wav, sr = torchaudio.load(wp)
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                if sr != self.sr:
                    wav = torchaudio.functional.resample(wav, sr, self.sr)
                if wav.shape[-1] > self.max_samples:
                    s = random.randint(0, wav.shape[-1] - self.max_samples)
                    wav = wav[..., s:s + self.max_samples]
                yield wav.squeeze(0)
            except:
                continue


def make_collate(hop):
    def collate(batch):
        if not batch:
            return None, None
        lengths = [x.shape[0] for x in batch]
        T = max(lengths)
        T = ((max(T, 4 * hop) + hop - 1) // hop) * hop
        stacked = torch.stack([F.pad(x, (0, T - x.shape[0])) for x in batch]).unsqueeze(1)
        frame_lengths = [l // hop for l in lengths]
        return stacked, frame_lengths
    return collate


# ============================================================
# Feature extraction
# ============================================================

@torch.no_grad()
def extract_mfcc(wav, sr=16000):
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr, n_mfcc=13,
        melkwargs={'n_fft': 400, 'hop_length': 320, 'n_mels': 23}
    ).to(wav.device)
    mfcc = mfcc_transform(wav.squeeze(1))  # [B, 13, T]
    delta1 = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta1)
    return torch.cat([mfcc, delta1, delta2], dim=1)  # [B, 39, T]


@torch.no_grad()
def collect_features(dl, encoder, cluster_layer, target_frames, device,
                     feature_type, reservoir_size=5_000_000):
    """Collect features using reservoir sampling."""
    reservoir = []
    total_frames = 0
    n_seen = 0

    pbar = tqdm(dl, desc=f"Collecting {feature_type} features")
    for wav, frame_lengths in pbar:
        if wav is None:
            continue
        wav = wav.to(device)

        if feature_type == 'mfcc':
            feats = extract_mfcc(wav)  # [B, 39, T]
        elif feature_type == 'encoder':
            layer_outputs = encoder.encode_layerwise(wav)
            feats = layer_outputs[cluster_layer].transpose(1, 2)  # [B, C, T]
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        for b in range(feats.shape[0]):
            real_T = min(frame_lengths[b], feats.shape[2])
            frames = feats[b, :, :real_T].T.cpu()  # [T, D]
            for i in range(frames.shape[0]):
                n_seen += 1
                if len(reservoir) < reservoir_size:
                    reservoir.append(frames[i])
                else:
                    j = random.randint(0, n_seen - 1)
                    if j < reservoir_size:
                        reservoir[j] = frames[i]
            total_frames += real_T

        pbar.set_postfix(frames=f"{total_frames / 1e6:.1f}M", reservoir=len(reservoir))
        if total_frames >= target_frames:
            break

    print(f"[Collect] {total_frames / 1e6:.1f}M total frames, reservoir={len(reservoir)}")
    return torch.stack(reservoir)


# ============================================================
# Diagonal GMM via EM
# ============================================================

@torch.no_grad()
def gmm_init_kmeans(data, K, device, n_iter=5, chunk_size=10000):
    """Initialize GMM with k-means."""
    N, D = data.shape
    print(f"[GMM Init] K-means init: N={N}, D={D}, K={K}")

    # Random init
    idx = torch.randperm(N)[:K]
    centroids = data[idx].to(device)

    for it in range(n_iter):
        all_labels = []
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            chunk = data[s:e].to(device)
            dists = torch.cdist(chunk, centroids)
            all_labels.append(dists.argmin(dim=1))
        labels = torch.cat(all_labels)

        for k in range(K):
            km = labels == k
            if km.sum() > 0:
                members = []
                for s in range(0, N, chunk_size):
                    e = min(s + chunk_size, N)
                    chunk_mask = km[s:e].cpu()
                    if chunk_mask.any():
                        members.append(data[s:e][chunk_mask].to(device))
                members = torch.cat(members)
                centroids[k] = members.mean(0)
            else:
                centroids[k] = data[random.randint(0, N - 1)].to(device)

        alive = (torch.bincount(labels, minlength=K) > 0).sum().item()
        print(f"  k-means iter {it+1}/{n_iter}: alive={alive}/{K}")

    # Compute initial variances and weights from k-means assignments
    means = centroids.clone()
    variances = torch.ones(K, D, device=device)
    weights = torch.zeros(K, device=device)

    for k in range(K):
        km = labels == k
        count = km.sum().item()
        if count > 1:
            members = []
            for s in range(0, N, chunk_size):
                e = min(s + chunk_size, N)
                chunk_mask = km[s:e].cpu()
                if chunk_mask.any():
                    members.append(data[s:e][chunk_mask].to(device))
            members = torch.cat(members)
            variances[k] = members.var(0).clamp(min=1e-6)
            weights[k] = count
        elif count == 1:
            variances[k] = torch.ones(D, device=device)
            weights[k] = 1
        else:
            weights[k] = 1e-6

    weights = weights / weights.sum()

    return means, variances, weights


@torch.no_grad()
def fit_gmm_em(data, K=1024, n_iter=20, device='cuda', chunk_size=10000):
    """Fit diagonal-covariance GMM using EM. Fully chunked to avoid OOM."""
    N, D = data.shape
    print(f"[GMM EM] N={N}, D={D}, K={K}, n_iter={n_iter}")

    # Initialize from k-means
    means, variances, weights = gmm_init_kmeans(data, K, device)

    log_weights = weights.log()  # [K]

    for it in range(n_iter):
        log_2pi = D * np.log(2 * np.pi)
        log_det = variances.log().sum(dim=1)  # [K]

        # Combined E-step + M-step accumulation in chunks
        # Accumulate sufficient statistics without storing full [N, K] resp matrix
        Nk = torch.zeros(K, device=device)
        sum_r_x = torch.zeros(K, D, device=device)
        sum_r_x2 = torch.zeros(K, D, device=device)
        ll = 0.0

        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            chunk = data[s:e].to(device)  # [chunk, D]

            # E-step for this chunk
            diff = chunk.unsqueeze(1) - means.unsqueeze(0)  # [chunk, K, D]
            mahal = (diff ** 2 / variances.unsqueeze(0)).sum(dim=2)  # [chunk, K]
            log_probs = -0.5 * (log_2pi + log_det.unsqueeze(0) + mahal)  # [chunk, K]
            log_joint = log_weights.unsqueeze(0) + log_probs  # [chunk, K]
            log_norm = log_joint.logsumexp(dim=1, keepdim=True)  # [chunk, 1]
            resp = (log_joint - log_norm).exp()  # [chunk, K]

            # Log-likelihood
            ll += log_norm.sum().item()

            # Accumulate M-step statistics
            Nk += resp.sum(dim=0)  # [K]
            sum_r_x += resp.T @ chunk  # [K, D]
            # For variance: sum r * x^2, then var = sum_r_x2/Nk - mean^2
            sum_r_x2 += resp.T @ (chunk ** 2)  # [K, D]

        ll /= N
        Nk = Nk.clamp(min=1e-8)

        # M-step
        new_means = sum_r_x / Nk.unsqueeze(1)
        new_variances = (sum_r_x2 / Nk.unsqueeze(1) - new_means ** 2).clamp(min=1e-6)
        new_weights = Nk / N

        means = new_means
        variances = new_variances
        weights = new_weights
        log_weights = weights.log()

        alive = (weights > 1e-6).sum().item()
        print(f"  EM iter {it+1}/{n_iter}: LL={ll:.4f}, alive={alive}/{K}")

    return means, variances, weights


# ============================================================
# Main
# ============================================================

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    hop = 320

    print("=" * 60)
    print(f"[GMM Fitting] feature_type={args.feature_type}, K={args.K}")
    if args.encoder_ckpt:
        print(f"  encoder={args.encoder_ckpt}, layer={args.cluster_layer}")
    print("=" * 60)

    # Load encoder for iteration 2
    encoder = None
    if args.feature_type == 'encoder':
        assert args.encoder_ckpt, "Need --encoder_ckpt for encoder features"
        ckpt = torch.load(args.encoder_ckpt, map_location='cpu')
        state_dict = ckpt['online'] if 'online' in ckpt else ckpt

        code_dim = state_dict['post_extract_proj.weight'].shape[0]
        conv_dim = state_dict['feature_extractor.conv_layers.0.0.weight'].shape[0]
        num_layers = sum(1 for k in state_dict if k.startswith('layers.') and k.endswith('.q_proj.weight'))
        num_heads = code_dim // (state_dict['layers.0.q_proj.weight'].shape[0] // code_dim)
        ff_dim = state_dict['layers.0.fc1.weight'].shape[0]

        print(f"[Encoder] dim={code_dim}, conv={conv_dim}, layers={num_layers}, heads={num_heads}, ff={ff_dim}")

        encoder = EncoderForExtraction(
            code_dim=code_dim, conv_dim=conv_dim,
            num_heads=num_heads, ff_dim=ff_dim,
            num_layers=num_layers,
        )
        # Load matching keys only
        model_keys = set(encoder.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in model_keys}
        encoder.load_state_dict(filtered, strict=False)
        encoder = encoder.to(device)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        print(f"[Encoder] Loaded {len(filtered)}/{len(model_keys)} keys")

    dl = DataLoader(
        StreamingDataset(args.jsonl, args.sample_rate, args.max_seconds),
        batch_size=args.batch_size, num_workers=4, pin_memory=True,
        collate_fn=make_collate(hop), prefetch_factor=4, timeout=60,
    )

    # Collect features
    data = collect_features(dl, encoder, args.cluster_layer, args.target_frames,
                            device, args.feature_type, args.reservoir_size)
    print(f"[Data] {data.shape[0]} frames, {data.shape[1]} dims")

    # Fit GMM
    means, variances, weights = fit_gmm_em(data, args.K, args.n_iter, device)
    alive = (weights > 1e-6).sum().item()

    # Compute entropy
    counts = weights * data.shape[0]
    p = weights[weights > 1e-6]
    entropy = -(p * p.log()).sum().item() / np.log(args.K)

    # Save
    gmm_path = os.path.join(args.out_dir, "gmm.pt")
    torch.save({
        'means': means.cpu(),
        'variances': variances.cpu(),
        'weights': weights.cpu(),
        'K': args.K,
        'dim': data.shape[1],
        'feature_type': args.feature_type,
        'cluster_layer': args.cluster_layer if args.feature_type == 'encoder' else None,
        'source_ckpt': args.encoder_ckpt,
    }, gmm_path)

    print(f"\n[Done] Saved to {gmm_path}")
    print(f"  K={args.K}, dim={data.shape[1]}, alive={alive}/{args.K}, entropy={entropy:.4f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--jsonl', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--sample_rate', type=int, default=16000)
    p.add_argument('--K', type=int, default=1024)
    p.add_argument('--feature_type', choices=['mfcc', 'encoder'], default='mfcc')
    p.add_argument('--encoder_ckpt', type=str, default=None)
    p.add_argument('--cluster_layer', type=int, default=9)
    p.add_argument('--target_frames', type=int, default=500_000_000)
    p.add_argument('--reservoir_size', type=int, default=5_000_000)
    p.add_argument('--n_iter', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--max_seconds', type=float, default=15.0)
    args = p.parse_args()
    main(args)
