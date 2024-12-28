import torch
import torch.nn as nn
from torchtune.modules import MultiHeadAttention, RotaryPositionalEmbeddings


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        ff_dim,
        dropout=0.1,
        attn_dropout=0.0,
        max_seq_len=4096,
        is_causal=False,
    ):
        super(EncoderLayer, self).__init__()

        # Rotary Positional Embeddings
        self.rotary_pos_emb = RotaryPositionalEmbeddings(
            dim=head_dim, max_seq_len=max_seq_len
        )

        # Multi-Head Attention
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False),
            pos_embeddings=self.rotary_pos_emb,
            q_norm=nn.LayerNorm(head_dim),  # Updated to normalize over head_dim
            k_norm=nn.LayerNorm(head_dim),  # Updated to normalize over head_dim
            max_seq_len=max_seq_len,
            is_causal=is_causal,
            attn_dropout=attn_dropout,
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)

        # Feedforward Network
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None, input_pos=None):
        # Self-Attention
        residual = x
        x = self.attn_layer_norm(x)
        attn_output = self.self_attn(
            x, y=x, mask=attn_mask, input_pos=input_pos
        )  # Pass y=x
        x = residual + self.attn_dropout(attn_output)

        # Feedforward
        residual = x
        x = self.ff_layer_norm(x)
        ff_output = self.ff(x)
        x = residual + self.ff_dropout(ff_output)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        ff_dim,
        dropout=0.1,
        attn_dropout=0.0,
        max_seq_len=4096,
        is_causal=False,
    ):
        super(TransformerEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal

        # Input Layer Norm
        self.input_layer_norm = nn.LayerNorm(embed_dim)

        # Encoder Layers
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    max_seq_len=max_seq_len,
                    is_causal=is_causal,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, input_pos=None):
        """
        Parameters:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim]
            attn_mask (torch.Tensor): Attention mask of shape [batch_size * num_heads, seq_len, seq_len]
            input_pos (torch.Tensor): Position ids tensor of shape [batch_size, seq_len]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        # Apply input layer norm
        x = self.input_layer_norm(x)

        # Apply dropout
        x = self.dropout(x)

        # Iterate through encoder layers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, input_pos=input_pos)

        return x
