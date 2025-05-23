import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from multihead_diffattn import MultiheadDiffAttn


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(
            torch.empty(ensemble_size, in_features, out_features)
        )
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=np.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias

    def extra_repr(self) -> str:
        return "ensemble_size={} in_features={}, out_features={}, bias={}".format(
            self.ensemble_size,
            self.in_features,
            self.out_features,
            self.bias is not None,
        )


class ResidualBlock(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1, dim2)
        self.fc2 = torch.nn.Linear(dim2, dim2)
        self.activation = nn.GELU()
    def forward(self, x):
        hidden = self.fc1(x)
        residual = hidden
        hidden = self.activation(hidden)
        out = self.fc2(hidden)
        out += residual
        return out


class MLPBlock(nn.Module):
    def __init__(self, dim1, dim2, num_layers=1, use_tanh=False):
        super().__init__()
        model = []
        for _ in range(num_layers-1):
            model.append(nn.Linear(dim1, dim1))
            model.append(nn.GELU())
        model.append(nn.Linear(dim1, dim2))
        if use_tanh:
            model.append(nn.Tanh())
        self.model = nn.Sequential(*model)
    def forward(self, x):
        out = self.model(x)
        return out


def build_rel_pos(seq_len: int, rotary_dim: int, device=None):
    """
    seq_len: the maximum sequence length you’ll ever see
    rotary_dim: the number of channels in each head you want to rotate (must be even)
    """
    assert rotary_dim % 2 == 0, "rotary_dim must be an even number"
    # frequencies for each pair of dims:
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim)
    )
    # positions
    pos_seq = torch.arange(seq_len, device=device).float()
    # outer product -> (seq_len, rotary_dim/2)
    sinusoid_inp = torch.einsum("i,j->ij", pos_seq, inv_freq)
    sin = sinusoid_inp.sin()
    cos = sinusoid_inp.cos()
    return cos, sin




class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        use_diff_att: bool,
        idx: int,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)
        
        if use_diff_att:
          #num_heads = 1 if num_heads == 1 else num_heads // 2
          self.attention = MultiheadDiffAttn(
              embed_dim = embedding_dim, num_heads = num_heads, depth=idx# , batch_first=True
          )
        else:
          self.attention = nn.MultiheadAttention(
              embedding_dim, num_heads, attention_dropout# , batch_first=True
          )

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len
        self.batch_first = True

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
      causal_mask = self.causal_mask[:x.shape[1], :x.shape[1]]
      norm_x = self.norm1(x)
      is_batched = x.dim() == 3

      if self.batch_first and is_batched:
          norm_x = norm_x.transpose(1, 0)  # [seq_len, batch_size, emb_dim] for standard MultiheadAttention

      if isinstance(self.attention, nn.MultiheadAttention):
          attention_out = self.attention(
              query=norm_x,
              key=norm_x,
              value=norm_x,
              attn_mask=causal_mask,
              key_padding_mask=padding_mask,
              need_weights=False,
          )[0]
      else:  # assume it's MultiheadDiffAttn
          if self.batch_first and is_batched:
              norm_x = norm_x.transpose(0, 1)  # [batch_size, seq_len, emb_dim] for MultiheadDiffAttn
          rel_pos = build_rel_pos(norm_x.shape[1],self.attention.head_dim ,norm_x.device)  # <--- YOU NEED TO PROVIDE THIS FUNCTION
          attention_out = self.attention(
              x=norm_x,
              rel_pos=rel_pos,
              attn_mask=causal_mask.float(),  # MultiheadDiffAttn expects float mask
          )

      if self.batch_first and is_batched and isinstance(self.attention, nn.MultiheadAttention):
         attention_out = attention_out.transpose(1, 0)
      
      x = x + self.drop(attention_out)
      x = x + self.mlp(self.norm2(x))
      return x
