#!/usr/bin/env python3
"""
JEPA Speech SSL model components.

Shared across training scripts (k-means CE, GMM KL, online k-means, online GMM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 dropout=0.1, attention_dropout=0.1, activation_dropout=0.0):
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


class JEPAPredictor(nn.Module):
    """def __init__(self, dim, num_heads=8, num_layers=1):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(dim) * 0.02)
        self.pos_embed = nn.Embedding(750, dim) # max 15 seconds × 16000 samples/sec = 240,000 samples. 240,000 / 320fps = 750 frames.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, dim_feedforward=dim * 2,
                dropout=0.1, batch_first=True
            ) for _ in range(num_layers)
        ])
        self.proj = nn.Linear(dim, dim)"""
    def __init__(self, dim, num_heads=8, num_layers=1, num_experts=1):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(dim) * 0.02)
        self.pos_embed = nn.Embedding(750, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, dim_feedforward=dim * 2,
                dropout=0.1, batch_first=True
            ) for _ in range(num_layers)
        ])
        self.proj = nn.Linear(dim, dim)
        self.expert_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_experts)
        ])

    #def forward(self, ctx, vis_idx_list, mask_idx_list, T_full):
    def forward(self, ctx, vis_idx_list, mask_idx_list, T_full, expert_idx=0):
        B, C, _ = ctx.shape
        device, dtype = ctx.device, ctx.dtype
        x = ctx.permute(0, 2, 1).clone()
        pos_ids = torch.arange(T_full, device=device).unsqueeze(0).expand(B, -1)
        pos = self.pos_embed(pos_ids).to(dtype)
        for b in range(B):
            x[b, mask_idx_list[b]] = self.mask_token.to(dtype)
        x = x + pos
        for layer in self.layers:
            x = layer(x)
        #x = self.proj(x)
        #z_pred = torch.zeros(B, C, T_full, device=device, dtype=dtype)
        x = self.proj(x)
        x = x + self.expert_projs[expert_idx](x)
        z_pred = torch.zeros(B, C, T_full, device=device, dtype=dtype)
        
        for b in range(B):
            z_pred[b, :, mask_idx_list[b]] = x[b, mask_idx_list[b]].transpose(0, 1)
        return z_pred


class OnlineEncoder(nn.Module):
    def __init__(self, code_dim=768, conv_dim=512, num_heads=12, ff_dim=3072,
                 num_layers=12, dropout=0.1, attention_dropout=0.1,
                 activation_dropout=0.0, layer_drop=0.05,
                 feature_grad_mult=0.1, K=100,
                 conv_pos_kernel=128, conv_pos_groups=16, **kwargs):
        super().__init__()
        self.hop = 320
        self.code_dim = code_dim
        self.feature_grad_mult = feature_grad_mult
        self.layer_drop = layer_drop
        self.feature_extractor = ConvFeatureExtractor(conv_dim)
        self.post_extract_proj = nn.Linear(conv_dim, code_dim)
        self.layer_norm = Fp32LayerNorm(code_dim)
        self.dropout_module = nn.Dropout(dropout)
        self.pos_conv = ConvPositionalEncoding(code_dim, conv_pos_kernel, conv_pos_groups)
        self.encoder_layer_norm = Fp32LayerNorm(code_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(code_dim, num_heads, ff_dim, dropout,
                             attention_dropout, activation_dropout)
            for _ in range(num_layers)
        ])
        self.cluster_head = nn.Sequential(
            nn.Linear(code_dim, code_dim), nn.GELU(),
            nn.Linear(code_dim, code_dim), nn.GELU(),
            nn.Linear(code_dim, K)
        )
        #self.predictor = JEPAPredictor(code_dim)
        self.predictor = JEPAPredictor(code_dim, num_experts=kwargs.get('num_experts', 1))

    def encode(self, wav, mask, step=0):
        x = self.feature_extractor(wav)
        if self.training and self.feature_grad_mult != 1.0:
            x = GradMultiply.apply(x, self.feature_grad_mult)
        x = x.transpose(1, 2)
        x = self.post_extract_proj(x)
        x = self.layer_norm(x)
        x = self.dropout_module(x)
        B, T, C = x.shape
        mask = mask[:, :T]
        x = x * mask.unsqueeze(-1)
        x = self.pos_conv(x)
        x = self.encoder_layer_norm(x)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2), mask

    @torch.no_grad()
    def encode_layer_avg(self, wav):
        x = self.feature_extractor(wav)
        x = x.transpose(1, 2)
        x = self.post_extract_proj(x)
        x = self.layer_norm(x)
        x = self.pos_conv(x)
        x = self.encoder_layer_norm(x)
        accum = torch.zeros_like(x)
        for layer in self.layers:
            x = layer(x)
            accum += x
        return accum / len(self.layers)

    @torch.no_grad()
    def encode_layer(self, wav, layer_idx):
        x = self.feature_extractor(wav)
        x = x.transpose(1, 2)
        x = self.post_extract_proj(x)
        x = self.layer_norm(x)
        x = self.pos_conv(x)
        x = self.encoder_layer_norm(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == layer_idx:
                return x
        return x

    @torch.no_grad()
    def encode_all_layers(self, wav, layer_indices=None):
        """
        Single forward pass, returns hidden states at the requested layer indices.
        If layer_indices is None, returns all layers.

        Returns: list of (B, T, code_dim) tensors, in the order given by layer_indices
                 (or layer order if None).

        Added without modifying encode_layer / encode_layer_avg, so any external script
        relying on those still works exactly as before.
        """
        x = self.feature_extractor(wav)
        x = x.transpose(1, 2)
        x = self.post_extract_proj(x)
        x = self.layer_norm(x)
        x = self.pos_conv(x)
        x = self.encoder_layer_norm(x)

        if layer_indices is None:
            layer_indices = list(range(len(self.layers)))

        wanted = set(int(i) for i in layer_indices)
        captured = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in wanted:
                captured[i] = x  # tensor reference; safe because no further in-place ops

        return [captured[int(i)] for i in layer_indices]

    """def forward(self, wav, mask, step=0):
        ctx, mask = self.encode(wav, mask=mask, step=step)
        B, C, T_z = ctx.shape
        vis_idx_list, mask_idx_list = [], []
        for b in range(B):
            vis_idx_list.append(mask[b].bool().nonzero(as_tuple=True)[0])
            mask_idx_list.append((~mask[b].bool()).nonzero(as_tuple=True)[0])
        z_pred = self.predictor(ctx, vis_idx_list, mask_idx_list, T_z)
        cluster_logits_vis = self.cluster_head(ctx.permute(0, 2, 1))
        cluster_logits_mask = self.cluster_head(z_pred.permute(0, 2, 1))
        return cluster_logits_vis, cluster_logits_mask"""

    def forward(self, wav, mask, step=0):
        ctx, mask = self.encode(wav, mask=mask, step=step)
        B, C, T_z = ctx.shape
        vis_idx_list, mask_idx_list = [], []
        for b in range(B):
            vis_idx_list.append(mask[b].bool().nonzero(as_tuple=True)[0])
            mask_idx_list.append((~mask[b].bool()).nonzero(as_tuple=True)[0])
        z_pred = self.predictor(ctx, vis_idx_list, mask_idx_list, T_z)
        cluster_logits_vis = self.cluster_head(ctx.permute(0, 2, 1))
        cluster_logits_mask = self.cluster_head(z_pred.permute(0, 2, 1))
        return cluster_logits_vis, cluster_logits_mask

    def forward_layer(self, wav, mask, layer_idx):
        x = self.feature_extractor(wav)
        if self.training and self.feature_grad_mult != 1.0:
            x = GradMultiply.apply(x, self.feature_grad_mult)
        x = x.transpose(1, 2)
        x = self.post_extract_proj(x)
        x = self.layer_norm(x)
        x = self.dropout_module(x)
        B, T, C = x.shape
        mask = mask[:, :T]
        x = x * mask.unsqueeze(-1)
        x = self.pos_conv(x)
        x = self.encoder_layer_norm(x)
        x_target = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == layer_idx:
                x_target = x  # save with grad
        # Use layer L output for prediction, but add 0 * final to keep all params in graph
        ctx = (x_target + 0 * x).transpose(1, 2)
        T_z = T
        vis_idx_list, mask_idx_list = [], []
        for b in range(B):
            vis_idx_list.append(mask[b].bool().nonzero(as_tuple=True)[0])
            mask_idx_list.append((~mask[b].bool()).nonzero(as_tuple=True)[0])
        #z_pred = self.predictor(ctx, vis_idx_list, mask_idx_list, T_z)
        z_pred = self.predictor(ctx, vis_idx_list, mask_idx_list, T_z, expert_idx=layer_idx)
        cluster_logits_vis = self.cluster_head(ctx.permute(0, 2, 1))
        cluster_logits_mask = self.cluster_head(z_pred.permute(0, 2, 1))
        return cluster_logits_vis, cluster_logits_mask
