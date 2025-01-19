import torch
import torch.nn as nn
from typing import Optional

# torchtune modules
from torchtune.modules import (
    RMSNorm,
    MultiHeadAttention,
    FeedForward,
    RotaryPositionalEmbeddings,
    TransformerSelfAttentionLayer,
    TransformerCrossAttentionLayer,
)


class TransformerDecoderLayer(nn.Module):
    """
    A single Transformer decoder layer with:
      (1) Self-attention sublayer (causal),
      (2) Cross-attention sublayer,
      (3) Feed-forward sublayer
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_kv_heads: int,
        head_dim: int,
        dim_feedforward: int,
        max_seq_len: int,
        attn_dropout: float = 0.0,
        is_causal: bool = True,
    ):
        super().__init__()
        """
        Args:
            d_model (int): Dimension of the model (embed_dim).
            nhead (int): Number of heads for self-attention.
            num_kv_heads (int): Number of key/value heads (for GQA).
            head_dim (int): Dimension per head.
            dim_feedforward (int): Hidden dimension of feed-forward.
            max_seq_len (int): Max sequence length for rotary embeddings.
            attn_dropout (float): Dropout inside attention. Defaults to 0.0.
            is_causal (bool): Whether self-attn is causal. Defaults to True (decoder).
        """
        mlp_sa = FeedForward(
            gate_proj=nn.Linear(d_model, dim_feedforward, bias=False),
            down_proj=nn.Linear(dim_feedforward, d_model, bias=False),
            up_proj=None,
        )

        # MLP for the cross-attention block
        mlp_ca = FeedForward(
            gate_proj=nn.Linear(d_model, dim_feedforward, bias=False),
            down_proj=nn.Linear(dim_feedforward, d_model, bias=False),
            up_proj=None,
        )

        # -- 1) Self-attention sublayer (no feed-forward) --
        # Create a MultiHeadAttention for decoder self-attention
        # with Rotary Pos Embeddings
        self_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(d_model, nhead * head_dim, bias=False),
            k_proj=nn.Linear(d_model, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(d_model, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(nhead * head_dim, d_model, bias=False),
            pos_embeddings=RotaryPositionalEmbeddings(
                dim=head_dim, max_seq_len=max_seq_len
            ),
            attn_dropout=attn_dropout,
            max_seq_len=max_seq_len,
            is_causal=is_causal,  # causal self-attention
        )
        for proj in [
            self_attn.q_proj,
            self_attn.k_proj,
            self_attn.v_proj,
            self_attn.output_proj,
        ]:
            nn.init.xavier_uniform_(proj.weight)

        self.self_attn_layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp_sa,
            sa_norm=RMSNorm(d_model, eps=1e-6),
            mlp_norm=RMSNorm(d_model, eps=1e-6),
            sa_scale=None,
            mlp_scale=None,
        )

        # -- 2) Cross-attention sublayer (with feed-forward) --
        # For cross-attention, we typically do not want rotary pos embeddings
        # (it raises AssertionError if pos_embeddings is not None).
        cross_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(d_model, nhead * head_dim, bias=False),
            k_proj=nn.Linear(d_model, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(d_model, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(nhead * head_dim, d_model, bias=False),
            pos_embeddings=None,  # Must be None for cross-attention
            attn_dropout=attn_dropout,
            max_seq_len=max_seq_len,  # Not used here, no pos_embeddings
            is_causal=False,  # cross-attn is not typically causal
        )
        for proj in [
            cross_attn.q_proj,
            cross_attn.k_proj,
            cross_attn.v_proj,
            cross_attn.output_proj,
        ]:
            nn.init.xavier_uniform_(proj.weight)

        self.cross_attn_layer = TransformerCrossAttentionLayer(
            attn=cross_attn,
            mlp=mlp_ca,
            ca_norm=RMSNorm(d_model, eps=1e-6),
            mlp_norm=RMSNorm(d_model, eps=1e-6),
            ca_scale=None,
            mlp_scale=None,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt (torch.Tensor): Target input [batch_size, tgt_seq_len, d_model].
            memory (Optional[torch.Tensor]): Encoder outputs [batch_size, mem_seq_len, d_model].
            tgt_mask (Optional[torch.Tensor]): [batch_size, tgt_seq_len, tgt_seq_len] or None.
            memory_mask (Optional[torch.Tensor]): [batch_size, tgt_seq_len, mem_seq_len] or None.
            tgt_pos (Optional[torch.Tensor]): Position indices for rotary embeddings (decoder).

        Returns:
            torch.Tensor: shape [batch_size, tgt_seq_len, d_model].
        """
        # -- Self-Attention sublayer --
        tgt = self.self_attn_layer(tgt)  # , mask=tgt_mask, input_pos=tgt_pos

        # -- Cross-Attention + FeedForward sublayer --
        tgt = self.cross_attn_layer(
            tgt,
            encoder_input=memory,
            encoder_mask=memory_mask,
        )

        return tgt


class TransformerDecoder(nn.Module):
    """
    A full Transformer decoder stack with multiple layers, each having:
      (1) Self-attn (causal),
      (2) Cross-attn,
      (3) FeedForward.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        num_kv_heads: int,
        head_dim: int,
        dim_feedforward: int,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        is_causal: bool = True,
    ):
        super().__init__()

        """
        Args:
            num_layers (int): Number of decoder layers.
            d_model (int): Model dimension (embed_dim).
            nhead (int): Number of heads for self/cross attention.
            num_kv_heads (int): Number of key/value heads (for GQA).
            head_dim (int): Dimension per head.
            dim_feedforward (int): Hidden dimension for feed-forward.
            max_seq_len (int): Max seq length for rotary embeddings. Defaults to 4096.
            dropout (float): Dropout after input norm. Defaults to 0.1.
            attn_dropout (float): Dropout inside attention. Defaults to 0.0.
            is_causal (bool): Causal self-attention (decoder). Defaults to True.
        """

        # Optional normalization on the raw decoder input
        self.input_norm = RMSNorm(d_model, eps=1e-6)
        self.output_norm = RMSNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        # Build N decoder layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dim_feedforward=dim_feedforward,
                max_seq_len=max_seq_len,
                attn_dropout=attn_dropout,
                is_causal=is_causal,
            )
            self.layers.append(layer)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer Decoder stack.

        Args:
            tgt (torch.Tensor): [batch_size, tgt_seq_len, d_model].
            memory (Optional[torch.Tensor]): [batch_size, mem_seq_len, d_model].
            tgt_mask (Optional[torch.Tensor]): [batch_size, tgt_seq_len, tgt_seq_len] or None.
            memory_mask (Optional[torch.Tensor]): [batch_size, tgt_seq_len, mem_seq_len] or None.
            tgt_pos (Optional[torch.Tensor]): Target token positions for rotary embeddings (optional).

        Returns:
            torch.Tensor: [batch_size, tgt_seq_len, d_model].
        """
        # Optional RMSNorm + dropout on the input
        tgt = self.input_norm(tgt)
        tgt = self.dropout(tgt)

        for layer in self.layers:
            tgt = layer(
                tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_pos=tgt_pos,
            )

        return self.output_norm(tgt)
