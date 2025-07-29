import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import math
import torch
import torch.nn.functional as F
from torch import nn

from rotary import apply_rotary_emb
#from flash_attn import flash_attn_func
# try:
#     from apex.normalization import FusedRMSNorm as RMSNorm 
# except ModuleNotFoundError:
#     print("No fused RMSNorm")
#     from rms_norm import RMSNorm


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


# class MultiheadDiffAttn(nn.Module):
#     def __init__(
#         self,
#         embed_dim,
#         depth, # current layer index
#         num_heads,
#         num_kv_heads=None,
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
        
#         # arg num_heads set to half of baseline Transformer's num_heads
#         # for e.g., to compare with a baseline Transformer with 16 heads, pass in num_heads=8 for DIFF Transformer
#         self.num_heads = num_heads
        
#         # arg num_kv_heads set to half of baseline Transformer's num_kv_heads if use GQA
#         # for e.g., to compare with a baseline Transformer with 16 heads and 8 kv_heads, 
#         # pass in num_heads=8, num_kv_heads=4 for DIFF Transformer
#         # if use MHA, pass in num_kv_heads=None
#         self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
#         self.n_rep = self.num_heads // self.num_kv_heads
        
#         self.head_dim = embed_dim // num_heads // 2
#         self.scaling = self.head_dim ** -0.5
        
#         self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
#         self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

#         # depth means current layer index
#         self.lambda_init = lambda_init_fn(depth)
#         self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
#         self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
#         self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
#         self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

#         self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
#     def forward(
#         self,
#         x,
#         rel_pos,
#         attn_mask=None,
#     ):
#         bsz, tgt_len, embed_dim = x.size()
#         src_len = tgt_len

#         q = self.q_proj(x)
#         k = self.k_proj(x)
#         v = self.v_proj(x)

#         q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
#         k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
#         v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

#         q = apply_rotary_emb(q, *rel_pos, interleaved=True)
#         k = apply_rotary_emb(k, *rel_pos, interleaved=True)

#         offset = src_len - tgt_len
#         q = q.transpose(1, 2)
#         k = repeat_kv(k.transpose(1, 2), self.n_rep)
#         v = repeat_kv(v.transpose(1, 2), self.n_rep)
#         q *= self.scaling
#         attn_weights = torch.matmul(q, k.transpose(-1, -2))
#         if attn_mask is None:
#             attn_mask = torch.triu(
#                 torch.zeros([tgt_len, src_len])
#                 .float()
#                 .fill_(float("-inf"))
#                 .type_as(attn_weights),
#                 1 + offset,
#             )
#         attn_weights = torch.nan_to_num(attn_weights)
#         attn_weights += attn_mask   

#         attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
#         A1 = attn_weights[:, :, 0]
#         A2 = attn_weights[:, :, 1]
        
#         A1 = F.softmax(A1, dim=-1, dtype=torch.float32).type_as(A1)
#         A2 = F.softmax(A2, dim=-1, dtype=torch.float32).type_as(A2)

        
#         lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
#         lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
#         lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
#         attn_weights = A1 - lambda_full * A2
        
#         attn = torch.matmul(attn_weights, v)
#         attn = self.subln(attn)
#         attn = attn * (1 - self.lambda_init)
#         attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

#         attn = self.out_proj(attn)
#         return attn



class DiffAttn(nn.Module):
    def __init__(self, dim, lambda_init=0.5, dropout=0.0):
        super().__init__()
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))
        self.scale = 1 / math.sqrt(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        B, N, D = Q.shape

        Q1, Q2 = Q.chunk(2, dim=-1)
        K1, K2 = K.chunk(2, dim=-1)

        A1 = torch.matmul(Q1, K1.transpose(-1, -2)) * self.scale
        A2 = torch.matmul(Q2, K2.transpose(-1, -2)) * self.scale

        mask = torch.tril(torch.ones(N, N, device=Q.device)).unsqueeze(0)  # (1, N, N)
        A1 = A1.masked_fill(mask == 0, float("-inf"))
        A2 = A2.masked_fill(mask == 0, float("-inf"))
 
        attn1 = F.softmax(A1, dim=-1)
        attn2 = F.softmax(A2, dim=-1)
        
        lambda_clamped = torch.clamp(self.lambda_param, 0.0, 1.0)
        attn = attn1 - lambda_clamped * attn2
        attn = self.dropout(attn)

        return torch.matmul(attn, V)

class MultiheadDiffAttn(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0, lambda_init=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.lambda_init = lambda_init

        self.W_q = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim * 2, bias=False)  # because V is 2d

        self.attn_heads = nn.ModuleList([
            DiffAttn(self.head_dim, lambda_init, dropout)
            for _ in range(num_heads)
        ])

        self.group_norm = nn.GroupNorm(1, embed_dim * 2)
        self.W_o = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        B, N, _ = x.shape

        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim * 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim * 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim * 2)

        Q = Q.permute(2, 0, 1, 3)  # [h, B, N, D]
        K = K.permute(2, 0, 1, 3)
        V = V.permute(2, 0, 1, 3)

        outs = [self.attn_heads[i](Q[i], K[i], V[i]) for i in range(self.num_heads)]
        out = torch.stack(outs, dim=0)  # [h, B, N, D]
        out = out.permute(1, 2, 0, 3).reshape(B, N, self.embed_dim * 2)
        
        out = out.transpose(1, 2)
        out = self.group_norm(out)
        out = out.transpose(1, 2)

        out = out * (1 - self.lambda_init)
        return self.W_o(out)

