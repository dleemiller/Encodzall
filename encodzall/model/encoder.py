# encoder.py
import torch
import torch.nn as nn
from torchtune.modules import MultiHeadAttention, RotaryPositionalEmbeddings
from typing import Optional, Tuple


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        """
        FeedForward network consisting of two linear layers with GELU activation and dropout.

        Args:
            d_model (int): Dimension of the input and output embeddings.
            dim_feedforward (int): Dimension of the hidden layer.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FeedForward network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model].
        """
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(
        self,
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
        Single Transformer Encoder Layer with Multi-Head Attention and FeedForward network.

        Args:
            d_model (int): Dimension of the input embeddings.
            nhead (int): Number of attention heads.
            num_kv_heads (int): Number of key/value heads.
            head_dim (int): Dimension of each attention head.
            dim_feedforward (int): Dimension of the FeedForward network.
            dropout (float, optional): Dropout rate after attention and FeedForward layers. Defaults to 0.1.
            attn_dropout (float, optional): Dropout rate within the attention mechanism. Defaults to 0.0.
            max_seq_len (int, optional): Maximum sequence length for positional embeddings. Defaults to 4096.
            is_causal (bool, optional): Whether the attention is causal. Defaults to False.
        """
        super(EncoderLayer, self).__init__()

        # Rotary Positional Embeddings
        self.rotary_pos_emb = RotaryPositionalEmbeddings(
            dim=head_dim, max_seq_len=max_seq_len
        )

        # Multi-Head Attention
        self.self_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,  # Mapping nhead to num_heads
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(d_model, nhead * head_dim, bias=False),
            k_proj=nn.Linear(d_model, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(d_model, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(nhead * head_dim, d_model, bias=False),
            pos_embeddings=self.rotary_pos_emb,
            q_norm=nn.LayerNorm(head_dim, eps=1e-5, elementwise_affine=False),
            k_norm=nn.LayerNorm(head_dim, eps=1e-5, elementwise_affine=False),
            max_seq_len=max_seq_len,
            is_causal=is_causal,
            attn_dropout=attn_dropout,
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.attn_layer_norm = nn.LayerNorm(d_model, eps=1e-5, elementwise_affine=False)

        # FeedForward Network
        self.ff = FeedForward(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_layer_norm = nn.LayerNorm(d_model, eps=1e-5, elementwise_affine=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the Encoder Layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].
            attn_mask (Optional[torch.Tensor], optional): Attention mask of shape [batch_size, seq_len, seq_len]. Defaults to None.
            input_pos (Optional[torch.Tensor], optional): Position indices tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model].
        """
        # Self-Attention
        residual = x
        x = self.attn_layer_norm(x)
        attn_output = self.self_attn(x, y=x, mask=attn_mask, input_pos=input_pos)
        x = residual + self.attn_dropout(attn_output)

        # FeedForward
        residual = x
        x = self.ff_layer_norm(x)
        ff_output = self.ff(x)
        x = residual + self.ff_dropout(ff_output)

        return x


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
        Transformer Encoder composed of multiple Encoder Layers.

        Args:
            num_layers (int): Number of Encoder Layers.
            d_model (int): Dimension of the input embeddings.
            nhead (int): Number of attention heads.
            num_kv_heads (int): Number of key/value heads.
            head_dim (int): Dimension of each attention head.
            dim_feedforward (int): Dimension of the FeedForward network.
            dropout (float, optional): Dropout rate after embeddings and before Encoder Layers. Defaults to 0.1.
            attn_dropout (float, optional): Dropout rate within the attention mechanism. Defaults to 0.0.
            max_seq_len (int, optional): Maximum sequence length for positional embeddings. Defaults to 4096.
            is_causal (bool, optional): Whether the attention is causal. Defaults to False.
        """
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dim_feedforward = dim_feedforward
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal

        # Input Layer Normalization
        self.input_layer_norm = nn.LayerNorm(
            d_model, eps=1e-5, elementwise_affine=False
        )

        # Encoder Layers
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    max_seq_len=max_seq_len,
                    is_causal=is_causal,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].
            attn_mask (Optional[torch.Tensor], optional): Attention mask of shape [batch_size, seq_len, seq_len]. Defaults to None.
            input_pos (Optional[torch.Tensor], optional): Position indices tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model].
        """
        # Apply input layer normalization and dropout
        x = self.input_layer_norm(x)
        x = self.dropout(x)

        # Pass through each Encoder Layer
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, input_pos=input_pos)

        return x
