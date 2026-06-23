#!/usr/bin/env python3
"""
JEPA + GMM KL with optional online GMM from EMA encoder.

Iter 1: Frozen MFCC GMM (--gmm_path, soft posteriors from frozen GMM on MFCC features, KL loss)
Iter 2: Online GMM from EMA encoder (--online_gmm, GMM params updated via EMA, KL loss)

In iter 2 mode:
  - EMA encoder is a slow copy of the online encoder (updated every step)
  - GMM means/variances/weights initialized from --gmm_path or randomly
  - Each step: EMA encoder produces features → compute soft posteriors → KL loss
  - After each step: GMM params updated as EMA of sufficient statistics
  - No backprop through GMM — params move via EMA

Architecture unchanged: encoder sees zeros at masked positions,
predictor fills in, KL loss on visible + masked positions.
"""

import os, json, argparse, random, glob, hashlib, math
from collections import deque
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, IterableDataset
import torch.distributed as dist
import deepspeed, random

from model import OnlineEncoder

print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())


def rank0():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)

def unwrap(m):
    return m.module if hasattr(m, "module") else m

def stable_hash(s):
    return hashlib.md5(s.encode()).hexdigest()


# ============================================================
# GMM classes
# ============================================================

class GMMAssigner:
    """Frozen GMM for iter 1 (MFCC features)."""
    def __init__(self, path, device='cuda'):
        data = torch.load(path, map_location=device, weights_only=False)
        self.means = data['means'].to(device)
        self.variances = data['variances'].to(device)
        self.weights = data['weights'].to(device)
        self.K = data['K']
        self.dim = data['dim']
        self.feature_type = data.get('feature_type', 'mfcc')
        self.log_weights = self.weights.log()

    @torch.no_grad()
    def posteriors(self, features):
        if features.dim() == 3:
            B, D, T = features.shape
            features = features.permute(0, 2, 1).reshape(-1, D)
        features = features.float()
        device = features.device
        means = self.means.to(device)
        variances = self.variances.to(device)
        log_weights = self.log_weights.to(device)
        log_2pi = self.dim * math.log(2 * math.pi)
        N = features.shape[0]
        chunk_size = 500
        all_posteriors = []
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            chunk = features[s:e]
            diff = chunk.unsqueeze(1) - means.unsqueeze(0)
            mahal = (diff ** 2 / variances.unsqueeze(0)).sum(dim=2)
            log_det = variances.log().sum(dim=1)
            log_probs = -0.5 * (log_2pi + log_det.unsqueeze(0) + mahal)
            log_joint = log_weights.unsqueeze(0) + log_probs
            log_posterior = log_joint - log_joint.logsumexp(dim=1, keepdim=True)
            all_posteriors.append(log_posterior.exp())
        return torch.cat(all_posteriors, dim=0)


class OnlineGMM:
    """Diagonal GMM with params updated via EMA of sufficient statistics."""
    def __init__(self, K, dim, device, init_gmm_path=None, param_ema=0.999):
        self.K = K
        self.dim = dim
        self.param_ema = param_ema

        if init_gmm_path is not None:
            data = torch.load(init_gmm_path, map_location=device, weights_only=False)
            self.means = data['means'].to(device).float()
            self.variances = data['variances'].to(device).float()
            self.weights = data['weights'].to(device).float()
            self.K = data['K']
        else:
            self.means = torch.randn(K, dim, device=device) * 0.1
            self.variances = torch.ones(K, dim, device=device)
            self.weights = torch.ones(K, device=device) / K

        self.log_weights = self.weights.log()
        self.counts = torch.zeros(K, device=device)

    @torch.no_grad()
    def posteriors(self, features):
        features = features.float()
        device = features.device
        means = self.means.to(device)
        variances = self.variances.to(device)
        log_weights = self.log_weights.to(device)
        log_2pi = self.dim * math.log(2 * math.pi)
        N = features.shape[0]
        chunk_size = 500
        all_posteriors = []
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            chunk = features[s:e]
            diff = chunk.unsqueeze(1) - means.unsqueeze(0)
            mahal = (diff ** 2 / variances.unsqueeze(0)).sum(dim=2)
            log_det = variances.log().sum(dim=1)
            log_probs = -0.5 * (log_2pi + log_det.unsqueeze(0) + mahal)
            log_joint = log_weights.unsqueeze(0) + log_probs
            log_posterior = log_joint - log_joint.logsumexp(dim=1, keepdim=True)
            all_posteriors.append(log_posterior.exp())
        return torch.cat(all_posteriors, dim=0)

    @torch.no_grad()
    def update(self, features, posteriors):
        features = features.float()
        resp = posteriors.float()
        N = features.shape[0]
        Nk = resp.sum(dim=0).clamp(min=1e-8)

        batch_means = (resp.T @ features) / Nk.unsqueeze(1)
        chunk_size = 500 #max(500, 4 * 1024**3 // (self.K * self.dim * 4))
        sum_rv2 = torch.zeros(self.K, self.dim, device=features.device)
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            diff = features[s:e].unsqueeze(1) - batch_means.unsqueeze(0)
            sum_rv2 += (resp[s:e].unsqueeze(2) * diff ** 2).sum(dim=0)
        batch_vars = (sum_rv2 / Nk.unsqueeze(1)).clamp(min=1e-6)
        batch_weights = Nk / Nk.sum()

        self.means = self.param_ema * self.means + (1 - self.param_ema) * batch_means
        self.variances = self.param_ema * self.variances + (1 - self.param_ema) * batch_vars
        self.weights = self.param_ema * self.weights + (1 - self.param_ema) * batch_weights
        self.weights = self.weights / self.weights.sum()
        self.log_weights = self.weights.clamp(min=1e-8).log()
        self.counts += Nk


# ============================================================
# Augmentation
# ============================================================

@torch.no_grad()
def mix_at_snr(clean, noise, snr_db):
    clean_energy = (clean.pow(2).sum() / clean.numel()).clamp(min=1e-8)
    noise_energy = (noise.pow(2).sum() / noise.numel()).clamp(min=1e-8)
    scale = torch.sqrt(clean_energy / (10 ** (snr_db / 10) * noise_energy))
    return clean + scale * noise

@torch.no_grad()
def mix_utterances(wav1, wav2, max_overlap_ratio=0.5):
    L1, L2 = wav1.shape[-1], wav2.shape[-1]
    max_mix_len = int(L1 * max_overlap_ratio)
    mix_len = random.randint(1, max(1, max_mix_len))
    start1 = random.randint(0, max(1, L1 - mix_len))
    start2 = random.randint(0, max(1, L2 - mix_len)) if L2 > mix_len else 0
    actual_mix_len = min(mix_len, L1 - start1, L2 - start2)
    if actual_mix_len <= 0:
        return wav1
    region1 = wav1[..., start1:start1 + actual_mix_len]
    region2 = wav2[..., start2:start2 + actual_mix_len]
    energy1 = (region1.pow(2).sum() / region1.numel()).clamp(min=1e-8)
    energy2 = (region2.pow(2).sum() / region2.numel()).clamp(min=1e-8)
    ratio_db = random.uniform(-5, 5)
    scale = torch.sqrt(energy1 * (10 ** (ratio_db / 10)) / energy2)
    mixed = wav1.clone()
    mixed[..., start1:start1 + actual_mix_len] += scale * region2
    return mixed

class DenoiseAugmentor:
    def __init__(self, p_noise=0.25, p_mix=0.25, snr_range_noise=(-5, 20), snr_range_speech=(-5, 5)):
        self.p_noise = p_noise
        self.p_mix = p_mix
        self.snr_range_noise = snr_range_noise
        self.snr_range_speech = snr_range_speech
        self.utterance_buffer = []
        self.buffer_size = 64

    def update_buffers(self, wav_batch):
        for i in range(wav_batch.shape[0]):
            w = wav_batch[i].detach().cpu().clone()
            if len(self.utterance_buffer) >= self.buffer_size:
                self.utterance_buffer.pop(0)
            self.utterance_buffer.append(w)

    @torch.no_grad()
    def __call__(self, wav_batch):
        if len(self.utterance_buffer) < 4:
            self.update_buffers(wav_batch)
            return wav_batch
        B, C, T = wav_batch.shape
        device = wav_batch.device
        augmented = wav_batch.clone()
        for b in range(B):
            r = random.random()
            if r < self.p_mix:
                other = random.choice(self.utterance_buffer).to(device)
                if other.dim() == 2:
                    other = other.unsqueeze(0)
                if other.shape[-1] < T:
                    other = F.pad(other, (0, T - other.shape[-1]))
                else:
                    start = random.randint(0, other.shape[-1] - T)
                    other = other[..., start:start + T]
                augmented[b] = mix_utterances(wav_batch[b], other.squeeze(0), max_overlap_ratio=0.5)
            elif r < self.p_mix + self.p_noise:
                noise_src = random.choice(self.utterance_buffer).to(device)
                if noise_src.shape[-1] < T:
                    noise_src = F.pad(noise_src, (0, T - noise_src.shape[-1]))
                else:
                    start = random.randint(0, noise_src.shape[-1] - T)
                    noise_src = noise_src[..., start:start + T]
                snr = random.uniform(*self.snr_range_noise)
                augmented[b] = mix_at_snr(wav_batch[b], noise_src, snr)
        self.update_buffers(wav_batch)
        return augmented


# ============================================================
# Dataset
# ============================================================

class StreamingDataset(IterableDataset):
    """
    Reads paths from a pre-built path-index file (one path per line).
    The index is built ONCE on rank 0 (see build_path_index) and shared
    across all ranks via the filesystem — no per-rank globbing of jsonl.

    Sharding across ranks/workers uses a shared epoch seed, so the slices
    are disjoint and reproducible.
    """
    def __init__(self, path_index_file, sr=16000, max_sec=15.0):
        self.path_index_file = path_index_file
        self.sr = sr
        self.max_samples = int(sr * max_sec)

    @staticmethod
    def build_path_index(root, base_path, out_file, seed=0):
        """Run ONCE on rank 0. Globs jsonl files, shuffles globally, writes line-by-line."""
        all_paths = []
        for root_dir in root.split(','):
            root_dir = root_dir.strip()
            for jf in sorted(glob.glob(os.path.join(root_dir, "*.jsonl"))):
                try:
                    with open(jf) as f:
                        for line in f:
                            try:
                                obj = json.loads(line.strip())
                                wp = obj["wav_path"]
                                if not wp.startswith('/'):
                                    wp = os.path.join(base_path, wp)
                                all_paths.append(wp)
                            except (ValueError, KeyError):
                                continue
                except (OSError, IOError):
                    continue
        # Shuffle ONCE, here, on rank 0 only.
        random.Random(seed).shuffle(all_paths)
        # Per-process tmp filename so concurrent calls don't collide on rename.
        tmp = f"{out_file}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write("\n".join(all_paths))
        os.replace(tmp, out_file)
        print(f"[StreamingDataset] Wrote shuffled path index: {len(all_paths)} entries -> {out_file}")
        return len(all_paths)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        rank = int(os.environ.get("RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        if worker:
            wid = rank * worker.num_workers + worker.id
            total = world * worker.num_workers
        else:
            wid, total = rank, world

        # Read every Nth line for this worker — no full list ever loaded.
        # The file was already globally shuffled at build time.
        my_paths = []
        with open(self.path_index_file, "r") as f:
            for i, line in enumerate(f):
                if i % total == wid:
                    line = line.rstrip("\n")
                    if line:
                        my_paths.append(line)

        for wp in my_paths:
            try:
                wav, sr = torchaudio.load(wp)
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                if sr != self.sr:
                    wav = torchaudio.functional.resample(wav, sr, self.sr)
                if wav.shape[-1] > self.max_samples:
                    s = random.randint(0, wav.shape[-1] - self.max_samples)
                    wav = wav[..., s:s + self.max_samples]
                yield wav.squeeze(0), wp
            except (OSError, RuntimeError, EOFError) as e:
                print(f"[SKIP] rank={rank} worker={wid} path={wp} "
                      f"err={type(e).__name__}: {e}", flush=True)
                continue


# ============================================================
# Helpers
# ============================================================

def make_collate(hop):
    def collate(batch):
        batch = [(w, p) for w, p in batch if w is not None]
        if not batch: return None, [], 0.0, None
        wavs, paths = zip(*batch)
        total_secs = sum(x.shape[0] for x in wavs)
        lengths = [x.shape[0] for x in wavs]
        T = max(lengths)
        T = ((max(T, 4 * hop) + hop - 1) // hop) * hop
        stacked = torch.stack([F.pad(x, (0, T - x.shape[0])) for x in wavs]).unsqueeze(1)
        frame_lengths = [l // hop for l in lengths]
        return stacked, list(paths), total_secs, frame_lengths
    return collate

def create_padding_mask(B, T, frame_lengths, device):
    # Vectorized: no per-sample Python loop, no CPU<->GPU sync per element.
    fl = torch.as_tensor(frame_lengths, device=device).unsqueeze(1)  # (B, 1)
    ar = torch.arange(T, device=device).unsqueeze(0)                  # (1, T)
    return (ar < fl).float()

def compute_mask_indices_slow(B, T, frame_lengths, mask_ratio=0.65, mask_length=10, device='cuda'):
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        real_T = frame_lengths[b]
        target_masked = int(real_T * mask_ratio)
        masked = 0
        while masked < target_masked:
            start = random.randint(0, max(0, real_T - mask_length))
            end = min(start + mask_length, real_T)
            new = (~mask[b, start:end]).sum().item()
            mask[b, start:end] = True
            masked += new
    return mask

def compute_mask_indices(B, T, frame_lengths, mask_ratio=0.65, mask_length=10, device='cuda'):
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        real_T = frame_lengths[b]
        num_spans = max(1, int(real_T * mask_ratio / mask_length))
        starts = torch.randint(0, max(1, real_T - mask_length), (num_spans,))
        for s in starts:
            mask[b, s:s + mask_length] = True
    return mask


def build_dataloader(args, hop, path_index):
    """Construct the DataLoader. Used for initial creation and for any rebuild."""
    return DataLoader(
        StreamingDataset(path_index_file=path_index,
                         sr=args.sample_rate, max_sec=args.max_seconds),
        batch_size=args.batch_size, num_workers=8, pin_memory=True,
        collate_fn=make_collate(hop), prefetch_factor=8,
        persistent_workers=True, timeout=300,
    )


# ============================================================
# Training
# ============================================================

def train(args):
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    os.makedirs(args.out_dir, exist_ok=True)
    hop = 320

    # Build path index ONCE on local rank 0 (replaces per-rank jsonl globbing).
    # Globally shuffled once, here — workers will not re-shuffle 13M entries.
    # NOTE: dist isn't initialized yet — use LOCAL_RANK from env (set by deepspeed/torchrun).
    path_index = os.path.join(args.out_dir, "path_index.txt")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0 and not os.path.exists(path_index):
        StreamingDataset.build_path_index(
            args.jsonl, "/scratch/gioannides/granary_data", path_index, seed=0)
    # Other ranks wait for rank 0 to finish writing.
    import time
    waited = 0
    while not os.path.exists(path_index):
        time.sleep(1)
        waited += 1
        if waited > 600:
            raise RuntimeError(f"Timed out waiting for {path_index} after 10 minutes")

    # Initial dataloader (path index is already shuffled).
    dl = build_dataloader(args, hop, path_index)

    use_online_gmm = args.online_gmm
    K = args.K if use_online_gmm else None

    gmm = None
    if not use_online_gmm:
        assert args.gmm_path, "Need --gmm_path for iter 1 (frozen MFCC GMM)"
        gmm = GMMAssigner(args.gmm_path)
        K = gmm.K

    online = OnlineEncoder(
        code_dim=args.code_dim, conv_dim=args.conv_dim,
        num_heads=args.num_heads, ff_dim=args.ff_dim,
        num_layers=args.num_layers, dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        layer_drop=args.layer_drop,
        feature_grad_mult=args.feature_grad_mult,
        conv_pos_kernel=args.conv_pos_kernel,
        conv_pos_groups=args.conv_pos_groups,
        K=K,
    )

    if rank0():
        total_params = sum(p.numel() for p in online.parameters())
        print(f"[Model] Total: {total_params/1e6:.1f}M params")
        if use_online_gmm:
            print(f"[Online GMM] K={K}, ema_layer={args.ema_layer}, "
                  f"ema_decay={args.ema_decay}, param_ema={args.param_ema}")
        else:
            print(f"[GMM] K={K}, feature_type={gmm.feature_type}")

    opt = torch.optim.AdamW(online.parameters(), lr=args.lr, weight_decay=1e-3, betas=(0.9, 0.99))

    with open(args.ds_config) as f:
        ds_cfg = json.load(f)

    global_step = 0
    ema_init_state = None
    if args.pt_ckpt and os.path.exists(args.pt_ckpt):
        if rank0():
            print(f"[Resume] Loading from {args.pt_ckpt}")
        ckpt = torch.load(args.pt_ckpt, map_location='cpu')
        state_dict = ckpt['online'] if 'online' in ckpt else ckpt
        global_step = ckpt.get('step', 0)
        if args.fresh_encoder:
            ema_init_state = state_dict
            if rank0():
                print(f"[Fresh] Online encoder random init, EMA from checkpoint")
        else:
            filtered_sd = {k: v for k, v in state_dict.items()
                           if not k.startswith('cluster_head') or v.shape == online.state_dict()[k].shape}
            online.load_state_dict(filtered_sd, strict=False)

    engine, _, _, _ = deepspeed.initialize(
        args=args, model=online, optimizer=opt,
        model_parameters=online.parameters(), config=ds_cfg
    )
    device = engine.device
    dtype = next(engine.module.parameters()).dtype

    # EMA encoder + online GMM for iter 2
    ema_encoder = None
    online_gm = None
    if use_online_gmm:
        if ema_init_state is not None:
            ema_encoder = deepcopy(unwrap(engine.module))
            filtered_ema = {k: v for k, v in ema_init_state.items()
                           if not k.startswith('cluster_head') or v.shape == ema_encoder.state_dict()[k].shape}
            ema_encoder.load_state_dict(filtered_ema, strict=False)
        else:
            ema_encoder = deepcopy(unwrap(engine.module))
        ema_encoder.to(device)
        ema_encoder.eval()
        for p_ema in ema_encoder.parameters():
            p_ema.requires_grad = False

        online_gm = OnlineGMM(K, args.code_dim, device,
                            init_gmm_path=args.gmm_path,
                            param_ema=args.param_ema)

        # Restore GMM and EMA from pt_ckpt if it's a portable checkpoint
        if args.pt_ckpt and not args.reseed_gmm:
            pt = torch.load(args.pt_ckpt, map_location=device, weights_only=False)
            if 'gmm_means' in pt:
                online_gm.means = pt['gmm_means'].to(device)
                online_gm.variances = pt['gmm_variances'].to(device)
                online_gm.weights = pt['gmm_weights'].to(device)
                online_gm.log_weights = online_gm.weights.clamp(min=1e-8).log()
                online_gm.counts = pt.get('gmm_counts', torch.zeros(K, device=device))
                if rank0(): print(f"[pt_ckpt] Restored GMM")
            if 'ema_encoder' in pt:
                filtered_ema = {k: v for k, v in pt['ema_encoder'].items()
                               if not k.startswith('cluster_head') or v.shape == ema_encoder.state_dict()[k].shape}
                ema_encoder.load_state_dict(filtered_ema, strict=False)
                if rank0(): print(f"[pt_ckpt] Restored EMA encoder")
            del pt
        if rank0() and args.gmm_path:
            print(f"[Online GMM] Init from {args.gmm_path}")

    # MFCC for iter 1
    mfcc_transform = None
    if gmm is not None and gmm.feature_type == 'mfcc':
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=13,
            melkwargs={'n_fft': 400, 'hop_length': 320, 'n_mels': 23}
        ).to(device)

    if args.p_noise < 0 or args.p_mix < 0:
        p_noise = round(random.uniform(0, 0.5), 2)
        p_mix = round(0.5 - p_noise, 2)
    else:
        p_noise = args.p_noise
        p_mix = args.p_mix
    # If both probs are 0 the augmentor still does CPU round-trips per batch
    # via update_buffers — skip it entirely.
    augmentor = None
    if p_noise > 0 or p_mix > 0:
        augmentor = DenoiseAugmentor(p_noise=p_noise, p_mix=p_mix)
    if rank0():
        print(f"[Augment] p_noise={p_noise}, p_mix={p_mix}, "
              f"{'enabled' if augmentor else 'DISABLED (both zero)'}")

    ckpt_dir = os.path.join(args.out_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    total_seconds = 0.0

    # Resolve ema_layer to integer index
    current_layer_idx = args.ema_layer
    if current_layer_idx < 0:
        current_layer_idx = args.num_layers + current_layer_idx

    # Auto layer selection state
    layer_ranks_ema = None
    rank_update_interval = 0
    if args.auto_layer:
        layer_ranks_ema = [0.0] * args.num_layers
        # Floor at 5000 to keep the SVD overhead out of the hot path.
        natural = int(1 / (1 - args.ema_decay)) if args.ema_decay < 1.0 else 0
        rank_update_interval = max(natural, 5000) if natural > 0 else 0
        if rank0():
            print(f"[Auto Layer] Enabled, start layer={current_layer_idx}, checking every {rank_update_interval} steps")

    if rank0():
        print(f"[Layer] Using layer {current_layer_idx}")

    if args.resume:
        global_step = 0
        total_seconds = 0.0
        try:
            _, client_sd = engine.load_checkpoint(ckpt_dir, tag=None, load_module_strict=False)
            if client_sd and 'step' in client_sd:
                global_step = client_sd['step']
                portable_path = os.path.join(ckpt_dir, f"portable_step{global_step}.pt")
                if os.path.exists(portable_path):
                    portable = torch.load(portable_path, map_location=device)
                    total_seconds = portable.get('total_seconds', 0.0)

                    # Always restore EMA encoder
                    if use_online_gmm and 'ema_encoder' in portable:
                        ema_encoder.load_state_dict(portable['ema_encoder'])
                        if rank0():
                            print(f"[Resume] Restored EMA encoder")

                    # Restore GMM params ONLY if not re-seeding
                    if use_online_gmm and 'gmm_means' in portable and not args.reseed_gmm:
                        online_gm.means = portable['gmm_means'].to(device)
                        online_gm.variances = portable['gmm_variances'].to(device)
                        online_gm.weights = portable['gmm_weights'].to(device)
                        online_gm.log_weights = online_gm.weights.clamp(min=1e-8).log()
                        online_gm.counts = portable.get('gmm_counts', torch.zeros(K, device=device))
                        if rank0():
                            print(f"[Resume] Restored online GMM params")
                    elif use_online_gmm and args.reseed_gmm:
                        # Force counts to 0 so seed block below fires
                        online_gm.counts = torch.zeros(K, device=device)
                        if rank0():
                            print(f"[Resume] --reseed_gmm set, will re-seed from EMA encoder")

                    if args.auto_layer and 'auto_layer' in portable:
                        current_layer_idx = portable['auto_layer']
                        layer_ranks_ema = portable.get('layer_ranks_ema', [0.0] * args.num_layers)
                        if rank0():
                            print(f"[Resume] Auto layer={current_layer_idx}")

                if rank0():
                    print(f"[Resume] Step {global_step}, hours={total_seconds/3600:.1f}")

            if hasattr(engine.lr_scheduler, 'num_steps'):
                engine.lr_scheduler.num_steps = global_step
                if rank0():
                    print(f"[Resume] Set scheduler to step {global_step}, lr={engine.lr_scheduler.get_lr()}")
        except Exception as e:
            if rank0():
                print(f"[Resume] Failed: {e}")

    # Seed online GMM from real features if not resumed
    if use_online_gmm:
        need_seed = online_gm.counts.sum() == 0

        if need_seed:
            with torch.no_grad():
                if rank0():
                    print(f"[GMM Init] Collecting up to {args.seed_frames} frames on rank 0...")

                    seed_feats_cpu = []
                    collected_frames = 0
                    seed_iter = iter(dl)
                    while collected_frames < args.seed_frames:
                        try:
                            wav_init, _, _, _ = next(seed_iter)
                        except StopIteration:
                            break
                        if wav_init is None:
                            continue
                        wav_init = wav_init.to(device, dtype)
                        feats = ema_encoder.encode_layer(wav_init, current_layer_idx)
                        seed_feats_cpu.append(feats.reshape(-1, args.code_dim).half().cpu())
                        collected_frames += feats.shape[0] * feats.shape[1]
                        del feats, wav_init

                    all_feats_cpu = torch.cat(seed_feats_cpu, dim=0)
                    del seed_feats_cpu
                    N = all_feats_cpu.shape[0]
                    print(f"[GMM Init] Collected {N} frames "
                          f"({N * args.code_dim * 2 / 1e9:.1f} GB fp16 on CPU)")

                    idx = torch.randperm(N)[:K]
                    centroids = all_feats_cpu[idx].to(device).float()

                    # Pass 2: mini-batch k-means (Sculley 2010) - one pass with
                    # decaying per-cluster learning rate. ~30x faster than full-batch.
                    kmeans_chunk = 200_000
                    total_counts = torch.zeros(K, device=device)
                    num_passes = args.seed_kmeans_iters  # reuse flag; 1-3 is plenty

                    for pass_i in range(num_passes):
                        perm = torch.randperm(N)
                        for s in range(0, N, kmeans_chunk):
                            e = min(s + kmeans_chunk, N)
                            chunk_idx = perm[s:e]
                            chunk = all_feats_cpu[chunk_idx].to(device).float()

                            # Assign to nearest centroid
                            a2 = (chunk * chunk).sum(1, keepdim=True)
                            b2 = (centroids * centroids).sum(1)
                            ab = chunk @ centroids.T
                            dists = a2 + b2.unsqueeze(0) - 2 * ab
                            labels_chunk = dists.argmin(dim=1)

                            # Per-chunk sums and counts
                            chunk_sum = torch.zeros(K, args.code_dim, device=device)
                            chunk_cnt = torch.zeros(K, device=device)
                            chunk_sum.index_add_(0, labels_chunk, chunk)
                            chunk_cnt.index_add_(
                                0, labels_chunk,
                                torch.ones_like(labels_chunk, dtype=torch.float)
                            )

                            # Online update: centroid += (chunk_sum - chunk_cnt * centroid) / (total + chunk_cnt)
                            new_totals = total_counts + chunk_cnt
                            safe_totals = new_totals.clamp(min=1).unsqueeze(1)
                            centroids = centroids + (chunk_sum - chunk_cnt.unsqueeze(1) * centroids) / safe_totals
                            total_counts = new_totals

                            del chunk, a2, b2, ab, dists, labels_chunk, chunk_sum, chunk_cnt

                        if rank0():
                            nonempty = (total_counts > 0).sum().item()
                            print(f"[GMM Init] mini-batch pass {pass_i+1}/{num_passes}: "
                                  f"{nonempty}/{K} clusters populated")

                    # Variance pass (unchanged - one sweep for sums, sum-of-squares, counts)
                    sum_per_cluster = torch.zeros(K, args.code_dim, device=device)
                    sum_sq_per_cluster = torch.zeros(K, args.code_dim, device=device)
                    count_per_cluster = torch.zeros(K, device=device)
                    for s in range(0, N, kmeans_chunk):
                        e = min(s + kmeans_chunk, N)
                        chunk = all_feats_cpu[s:e].to(device).float()
                        a2 = (chunk * chunk).sum(1, keepdim=True)
                        b2 = (centroids * centroids).sum(1)
                        ab = chunk @ centroids.T
                        dists = a2 + b2.unsqueeze(0) - 2 * ab
                        labels_chunk = dists.argmin(dim=1)
                        sum_per_cluster.index_add_(0, labels_chunk, chunk)
                        sum_sq_per_cluster.index_add_(0, labels_chunk, chunk * chunk)
                        count_per_cluster.index_add_(
                            0, labels_chunk,
                            torch.ones_like(labels_chunk, dtype=torch.float)
                        )
                        del chunk, a2, b2, ab, dists, labels_chunk

                    mask_k = count_per_cluster > 1
                    means = centroids.clone()
                    means[mask_k] = sum_per_cluster[mask_k] / count_per_cluster[mask_k].unsqueeze(1)
                    variances = torch.ones(K, args.code_dim, device=device)
                    variances[mask_k] = (
                        sum_sq_per_cluster[mask_k] / count_per_cluster[mask_k].unsqueeze(1)
                        - means[mask_k] ** 2
                    ).clamp(min=1e-6)

                    online_gm.means = means
                    online_gm.variances = variances
                    online_gm.weights = (count_per_cluster / count_per_cluster.sum().clamp(min=1)).clamp(min=1e-8)
                    online_gm.weights = online_gm.weights / online_gm.weights.sum()
                    online_gm.log_weights = online_gm.weights.log()

                    del all_feats_cpu, sum_per_cluster, sum_sq_per_cluster, count_per_cluster
                    torch.cuda.empty_cache()

                    nonempty = (online_gm.weights > 1e-6).sum().item()
                    print(f"[GMM Init] Seeded {nonempty}/{K} clusters from {N} frames")

            # Broadcast seeded GMM params from rank 0 to all other ranks
            if dist.is_initialized():
                if rank0():
                    print(f"[GMM Init] Broadcasting GMM params to all ranks...")
                dist.broadcast(online_gm.means, src=0)
                dist.broadcast(online_gm.variances, src=0)
                dist.broadcast(online_gm.weights, src=0)
                # recompute log_weights locally after broadcast
                online_gm.log_weights = online_gm.weights.clamp(min=1e-8).log()
                if rank0():
                    print(f"[GMM Init] Broadcast complete")

    metrics = {k: deque(maxlen=10) for k in ['loss_vis', 'loss_mask', 'loss']}

    pbar = tqdm(total=args.max_steps, disable=not rank0(), initial=global_step, desc="Training")

    epoch = 0
    dl_iter = iter(dl)

    while global_step < args.max_steps:
        try:
            wav, paths, batch_secs, frame_lengths = next(dl_iter)
        except StopIteration:
            # Clean epoch boundary — coordinated across ranks.
            if not args.loop_forever:
                break
            epoch += 1
            if rank0():
                print(f"[INFO] Epoch {epoch} starting — reshuffling index", flush=True)
            # Re-shuffle the global path index on local rank 0; others wait.
            if local_rank == 0:
                StreamingDataset.build_path_index(
                    args.jsonl, "/scratch/gioannides/granary_data",
                    path_index, seed=epoch)
            if dist.is_initialized():
                dist.barrier()
            dl_iter = iter(dl)
            continue

        if wav is None:
            # Collate filtered an all-bad batch (rare); just try the next.
            continue

        batch_secs_tensor = torch.tensor(batch_secs, device=device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(batch_secs_tensor)
        total_seconds += batch_secs_tensor.item() / args.sample_rate

        wav = wav.to(device, dtype)
        B = wav.shape[0]
        if torch.isnan(wav).any() or torch.isinf(wav).any():
            continue

        # Skip augmentor entirely if disabled (saves CPU round-trips).
        wav_aug = augmentor(wav) if augmentor is not None else wav

        with torch.no_grad():
            # Analytical T_z — saves a full CNN forward pass per step.
            T_z = wav.shape[-1] // hop

            if use_online_gmm:
                ema_feats = ema_encoder.encode_layer(wav, current_layer_idx)
                if ema_feats.shape[1] > T_z:
                    ema_feats = ema_feats[:, :T_z]
                elif ema_feats.shape[1] < T_z:
                    T_z = ema_feats.shape[1]
                ema_feats_flat = ema_feats.reshape(-1, args.code_dim)

                soft_targets = online_gm.posteriors(ema_feats_flat).view(B, T_z, K)

            elif gmm is not None and gmm.feature_type == 'mfcc':
                mfcc = mfcc_transform(wav.squeeze(1).float())
                delta1 = torchaudio.functional.compute_deltas(mfcc)
                delta2 = torchaudio.functional.compute_deltas(delta1)
                features = torch.cat([mfcc, delta1, delta2], dim=1)
                if features.shape[2] > T_z:
                    features = features[:, :, :T_z]
                elif features.shape[2] < T_z:
                    T_z = features.shape[2]
                soft_targets = gmm.posteriors(features).view(B, T_z, K)

            else:
                raise ValueError("Use --online_gmm for iter 2, or --gmm_path with MFCC GMM for iter 1.")

        mask_bool = compute_mask_indices(B, T_z, frame_lengths, mask_ratio=0.65, mask_length=10, device=device)
        mask = (~mask_bool).float().to(dtype)
        pad_mask = create_padding_mask(B, T_z, frame_lengths, device).to(dtype)

        cluster_logits_vis, cluster_logits_mask = engine.module(wav_aug, mask, step=global_step)

        T_out = cluster_logits_mask.shape[1]
        mask = mask[:, :T_out]
        pad_mask = pad_mask[:, :T_out]
        soft_targets = soft_targets[:, :T_out]

        # KL on visible positions (computed for logging only; no_grad since not in total loss)
        with torch.no_grad():
            vis_flat = (mask * pad_mask).view(-1).bool()
            if vis_flat.any():
                log_pred_vis = F.log_softmax(cluster_logits_vis.reshape(-1, K)[vis_flat], dim=-1)
                target_vis = soft_targets.reshape(-1, K)[vis_flat]
                loss_vis = F.kl_div(log_pred_vis, target_vis, reduction='batchmean')
            else:
                loss_vis = torch.tensor(0.0, device=device)

        # KL on masked positions
        mask_flat = ((1 - mask) * pad_mask).view(-1).bool()
        if mask_flat.any():
            log_pred_mask = F.log_softmax(cluster_logits_mask.reshape(-1, K)[mask_flat], dim=-1)
            target_mask = soft_targets.reshape(-1, K)[mask_flat]
            loss_mask = F.kl_div(log_pred_mask, target_mask, reduction='batchmean')
        else:
            loss_mask = torch.tensor(0.0, device=device)

        loss = loss_mask  # + loss_vis (disabled)

        if torch.isnan(loss):
            if rank0():
                print(f"[DEBUG] NaN! mask={loss_mask.item()}")
            continue

        engine.backward(loss)
        engine.step()

        # EMA encoder + online GMM update
        if use_online_gmm:
            with torch.no_grad():
                ema_params = list(ema_encoder.parameters())
                online_params = list(unwrap(engine.module).parameters())
                torch._foreach_mul_(ema_params, args.ema_decay)
                torch._foreach_add_(ema_params, online_params, alpha=1 - args.ema_decay)

                batch_posteriors = soft_targets.reshape(-1, K)
                online_gm.update(ema_feats_flat, batch_posteriors)

        # Accumulate as tensors — defer .item() syncs until we actually log.
        metrics['loss_vis'].append(loss_vis.detach())
        metrics['loss_mask'].append(loss_mask.detach())
        metrics['loss'].append(loss.detach())

        global_step += 1
        if rank0():
            pbar.update(1)

        if global_step % args.log_every == 0:
            # Now sync — once per log interval, not once per step.
            avgs = {k: torch.stack(list(v)).mean().item() for k, v in metrics.items()}
            hours = total_seconds / 3600
            if rank0():
                extra = {}
                if args.auto_layer:
                    extra['layer'] = current_layer_idx
                pbar.set_postfix(vis=f"{avgs['loss_vis']:.4f}", jepa=f"{avgs['loss_mask']:.4f}",
                 total=f"{avgs['loss']:.4f}", hrs=f"{hours:.1f}",
                 lr=f"{engine.optimizer.param_groups[0]['lr']:.2e}", **extra)
                with open(os.path.join(args.out_dir, "log.txt"), "a") as f:
                    f.write(f"{global_step}\t{avgs['loss_vis']:.6f}\t{avgs['loss_mask']:.6f}\t{avgs['loss']:.6f}\t{hours:.2f}\n")

        # Auto layer selection based on effective rank
        if args.auto_layer and rank_update_interval > 0 and global_step % rank_update_interval == 0 and global_step > 0:
            with torch.no_grad():
                rank_encoder = ema_encoder if use_online_gmm else unwrap(engine.module)
                best_rank = -1
                best_layer = current_layer_idx
                for li in range(args.num_layers):
                    feats = rank_encoder.encode_layer(wav, li)
                    feats_flat = feats.reshape(-1, args.code_dim).float()
                    N = min(feats_flat.shape[0], 2000)
                    sub = feats_flat[torch.randperm(feats_flat.shape[0])[:N]]
                    sub = sub - sub.mean(dim=0)
                    sv = torch.linalg.svdvals(sub)
                    p = sv / sv.sum()
                    p = p[p > 1e-10]
                    eff_rank = torch.exp(-(p * p.log()).sum()).item()
                    rank_ema_rate = args.ema_decay ** rank_update_interval
                    layer_ranks_ema[li] = rank_ema_rate * layer_ranks_ema[li] + (1 - rank_ema_rate) * eff_rank
                    if layer_ranks_ema[li] > best_rank:
                        best_rank = layer_ranks_ema[li]
                        best_layer = li
                if rank0():
                    ranks_str = ', '.join([f'L{i}={layer_ranks_ema[i]:.1f}' for i in range(args.num_layers)])
                    print(f"\n[Auto Layer] Step {global_step}: ranks=[{ranks_str}], current={current_layer_idx}")
                if best_layer != current_layer_idx:
                    if rank0():
                        print(f"[Auto Layer] Switching {current_layer_idx} -> {best_layer}")
                    current_layer_idx = best_layer

        if args.save_every > 0 and global_step % args.save_every == 0:
            engine.save_checkpoint(ckpt_dir, tag=f"step{global_step}", client_state={'step': global_step})
            if rank0():
                save_dict = {
                    'online': unwrap(engine.module).state_dict(),
                    'step': global_step,
                    'total_seconds': total_seconds,
                }
                if use_online_gmm:
                    save_dict['ema_encoder'] = ema_encoder.state_dict()
                    save_dict['gmm_means'] = online_gm.means.cpu()
                    save_dict['gmm_variances'] = online_gm.variances.cpu()
                    save_dict['gmm_weights'] = online_gm.weights.cpu()
                    save_dict['gmm_counts'] = online_gm.counts.cpu()
                    if args.auto_layer:
                        save_dict['auto_layer'] = current_layer_idx
                        save_dict['layer_ranks_ema'] = layer_ranks_ema
                # Move state_dicts to CPU here (must hold GIL while reading params),
                # then write to disk in a background thread so training continues.
                save_dict['online'] = {k: v.detach().cpu().clone()
                                       for k, v in save_dict['online'].items()}
                if 'ema_encoder' in save_dict:
                    save_dict['ema_encoder'] = {k: v.detach().cpu().clone()
                                                for k, v in save_dict['ema_encoder'].items()}
                portable_path = os.path.join(ckpt_dir, f"portable_step{global_step}.pt")
                import threading
                threading.Thread(
                    target=torch.save, args=(save_dict, portable_path), daemon=True
                ).start()

                # Log effective ranks (online encoder in Iter 1, EMA encoder in Iter 2)
                with torch.no_grad():
                    rank_encoder = ema_encoder if use_online_gmm else unwrap(engine.module)
                    ranks = []
                    for li in range(args.num_layers):
                        feats = rank_encoder.encode_layer(wav, li)
                        feats_flat = feats.reshape(-1, args.code_dim).float()
                        N = min(feats_flat.shape[0], 2000)
                        sub = feats_flat[torch.randperm(feats_flat.shape[0])[:N]]
                        sub = sub - sub.mean(dim=0)
                        sv = torch.linalg.svdvals(sub)
                        p = sv / sv.sum()
                        p = p[p > 1e-10]
                        ranks.append(torch.exp(-(p * p.log()).sum()).item())
                    ranks_str = '\t'.join([f'{r:.2f}' for r in ranks])
                    with open(os.path.join(args.out_dir, "effective_ranks.txt"), "a") as f:
                        f.write(f"{global_step}\t{ranks_str}\n")

    engine.save_checkpoint(ckpt_dir, tag="final", client_state={'step': global_step})
    if rank0():
        print(f"[Done] {global_step} steps")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--jsonl', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--ds_config', required=True)
    p.add_argument('--local_rank', type=int, default=-1)

    # Iter 1: frozen MFCC GMM
    p.add_argument('--gmm_path', type=str, default=None, help='GMM .pt path. Required for iter 1, optional init for iter 2.')

    # Iter 2: online GMM from EMA encoder
    p.add_argument('--online_gmm', action='store_true', help='Enable online GMM from EMA encoder (iter 2 mode).')
    p.add_argument('--K', type=int, default=500, help='Number of GMM components for online mode.')
    p.add_argument('--ema_decay', type=float, default=1.0, help='EMA decay for target encoder.')
    p.add_argument('--ema_layer', type=int, default=7, help='Layer index for EMA clustering (-1 = last layer).')
    p.add_argument('--param_ema', type=float, default=0.9999, help='EMA decay for GMM param updates.')
    p.add_argument('--auto_layer', action='store_true', help='Enable effective rank auto layer switching')

    p.add_argument('--sample_rate', type=int, default=16000)
    p.add_argument('--code_dim', type=int, default=768)
    p.add_argument('--conv_dim', type=int, default=512)
    p.add_argument('--num_heads', type=int, default=12)
    p.add_argument('--ff_dim', type=int, default=3072)
    p.add_argument('--num_layers', type=int, default=6)

    p.add_argument('--pt_ckpt', type=str, default=None)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_steps', type=int, default=999999999)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--max_seconds', type=float, default=15.0)
    p.add_argument('--log_every', type=int, default=10)
    p.add_argument('--save_every', type=int, default=1000)
    p.add_argument('--resume', action='store_true')

    p.add_argument('--p_noise', type=float, default=-1)
    p.add_argument('--p_mix', type=float, default=-1)

    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--attention_dropout', type=float, default=0.1)
    p.add_argument('--activation_dropout', type=float, default=0.1)
    p.add_argument('--layer_drop', type=float, default=0.05)
    p.add_argument('--feature_grad_mult', type=float, default=0.1)
    p.add_argument('--conv_pos_kernel', type=int, default=128)
    p.add_argument('--conv_pos_groups', type=int, default=16)

    p.add_argument('--loop_forever', action='store_true')
    p.add_argument('--fresh_encoder', action='store_true', help='Random init online encoder, EMA from checkpoint')
    p.add_argument('--seed_frames', type=int, default=50_000_000, help='Min frames for GMM seeding')
    p.add_argument('--seed_kmeans_iters', type=int, default=50, help='K-means iterations for GMM seeding')
    p.add_argument('--reseed_gmm', action='store_true',
               help='On resume, re-run k-means seeding from current EMA encoder, '
                    'discarding saved GMM params.')

    args = p.parse_args()
    from tqdm import tqdm
    if rank0():
        print("=" * 60)
        if args.online_gmm:
            print(f"[JEPA + Online GMM KL] K={args.K}, ema_decay={args.ema_decay}")
            print(f"  ema_layer={args.ema_layer}, param_ema={args.param_ema}")
            if args.auto_layer:
                print(f"  auto_layer=True")
            if args.gmm_path:
                print(f"  init GMM from {args.gmm_path}")
        else:
            print(f"[JEPA + Frozen GMM KL] gmm={args.gmm_path}")
        print(f"  encoder: {args.num_layers}L, dim={args.code_dim}")
        print("=" * 60)
    train(args)
