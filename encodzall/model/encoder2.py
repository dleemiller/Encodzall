# encoder_refactored.py
import torch
import torch.nn as nn

# Import from torchtune
from torchtune.modules import (
    RMSNorm,
    TransformerSelfAttentionLayer,
    MultiHeadAttention,
    FeedForward,
    RotaryPositionalEmbeddings,
)

from typing import Optional


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        num_kv_heads: int,
        head_dim: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        max_seq_len: int = 4096,
        is_causal: bool = False,
    ):
        """
        Transformer Encoder composed of multiple Self-Attention Layers (Llama2-style).

        Args:
            num_layers (int): Number of Self-Attention Layers.
            d_model (int): Dimension of the input embeddings.
            nhead (int): Number of attention heads.
            num_kv_heads (int): Number of key/value heads (for GQA).
            head_dim (int): Dimension of each attention head.
            dim_feedforward (int): Dimension of the FeedForward network.
            dropout (float, optional): Dropout rate after embeddings and before Encoder Layers. Defaults to 0.1.
            attn_dropout (float, optional): Dropout rate within the attention mechanism. Defaults to 0.0.
            max_seq_len (int, optional): Maximum sequence length for positional embeddings. Defaults to 4096.
            is_causal (bool, optional): Whether the attention is causal. Defaults to False.
        """
        super().__init__()

        # 1) Optional input norm in RMS style
        self.input_layer_norm = RMSNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        # 2) Create a rotary embeddings object (passed into MultiHeadAttention below)
        rotary_pos_emb = RotaryPositionalEmbeddings(
            dim=head_dim,
            max_seq_len=max_seq_len,
        )

        # 3) Build each TransformerSelfAttentionLayer
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # a) Multi-Head Attention module
            attn = MultiHeadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                num_kv_heads=num_kv_heads,  # GQA
                head_dim=head_dim,
                q_proj=nn.Linear(d_model, nhead * head_dim, bias=False),
                k_proj=nn.Linear(d_model, num_kv_heads * head_dim, bias=False),
                v_proj=nn.Linear(d_model, num_kv_heads * head_dim, bias=False),
                output_proj=nn.Linear(nhead * head_dim, d_model, bias=False),
                pos_embeddings=rotary_pos_emb,
                max_seq_len=max_seq_len,
                is_causal=is_causal,
                attn_dropout=attn_dropout,
            )

            # b) FeedForward module (Llama2 style usually has a "gate_proj * σ(up_proj)")
            #    but if you'd like to stick to a simpler MLP, you can do so:
            mlp = FeedForward(
                gate_proj=nn.Linear(d_model, dim_feedforward, bias=False),
                down_proj=nn.Linear(dim_feedforward, d_model, bias=False),
                # up_proj can be None or a second linear of same hidden size:
                up_proj=None,
                # or you can provide an activation if you want something else:
                # activation=nn.SiLU(),
            )

            # c) RMSNorm in place of the typical "pre-layernorm"
            sa_norm = RMSNorm(d_model, eps=1e-6)
            mlp_norm = RMSNorm(d_model, eps=1e-6)

            # d) Put it all together
            layer = TransformerSelfAttentionLayer(
                attn=attn,
                mlp=mlp,
                sa_norm=sa_norm,
                mlp_norm=mlp_norm,
                sa_scale=None,  # Optional gating or scaling (e.g., TanhGate, etc.)
                mlp_scale=None,  # Optional gating or scaling
            )
            self.layers.append(layer)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer Encoder.

        Args:
            x (torch.Tensor): [batch_size, seq_len, d_model].
            attn_mask (Optional[torch.Tensor], optional): [batch_size, seq_len, seq_len] or None.
            input_pos (Optional[torch.Tensor], optional): Position indices tensor or None.

        Returns:
            torch.Tensor: [batch_size, seq_len, d_model].
        """
        # (1) Optional input RMSNorm & dropout
        x = self.input_layer_norm(x)
        x = self.dropout(x)

        # (2) Pass through each transformer layer
        for layer in self.layers:
            x = layer(x, mask=attn_mask, input_pos=input_pos)

        return x
