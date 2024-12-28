import torch
import torch.nn as nn
import math


class RotaryPositionEmbedding:
    def __init__(self, dim):
        self.dim = dim

    def get_rotary_positions(self, seq_len, device):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )
        positions = torch.arange(seq_len, device=device).unsqueeze(1) * inv_freq
        sinusoid = torch.stack((positions.sin(), positions.cos()), dim=-1).reshape(
            seq_len, -1
        )
        return sinusoid

    def apply_rotary_position(self, x, rotary_positions):
        x1, x2 = x[..., ::2], x[..., 1::2]
        rp1, rp2 = rotary_positions[..., ::2], rotary_positions[..., 1::2]
        x_rotated = torch.cat((x1 * rp2 - x2 * rp1, x1 * rp1 + x2 * rp2), dim=-1)
        return x_rotated


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.rotary_embedding = RotaryPositionEmbedding(d_model // nhead)
        self.nhead = nhead
        self.d_model = d_model

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask):
        # Ensure all tensors are on the same device
        device = src.device
        src_mask = src_mask.to(device)

        batch_size, seq_len, d_model = src.size()
        head_dim = d_model // self.nhead

        # Reshape src to (batch_size, seq_len, nhead, head_dim)
        src_reshaped = src.view(batch_size, seq_len, self.nhead, head_dim)
        src_reshaped = src_reshaped.permute(
            0, 2, 1, 3
        )  # (batch_size, nhead, seq_len, head_dim)

        rotary_positions = self.rotary_embedding.get_rotary_positions(seq_len, device)

        # Apply rotary embedding to Q and K
        q, k, v = src_reshaped, src_reshaped, src_reshaped
        q_rotated = self.rotary_embedding.apply_rotary_position(q, rotary_positions)
        k_rotated = self.rotary_embedding.apply_rotary_position(k, rotary_positions)

        q_rotated = q_rotated.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, d_model
        )  # (batch_size, seq_len, d_model)
        k_rotated = k_rotated.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, d_model
        )  # (batch_size, seq_len, d_model)

        # Expand attention mask for nhead
        expanded_mask = src_mask.unsqueeze(1).repeat(
            1, self.nhead, 1, 1
        )  # (batch_size, nhead, seq_len, seq_len)
        expanded_mask = expanded_mask.reshape(batch_size * self.nhead, seq_len, seq_len)

        # Ensure expanded_mask is on the same device
        expanded_mask = expanded_mask.to(device)

        attn_output, _ = self.self_attn(
            q_rotated, k_rotated, src, attn_mask=expanded_mask
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feedforward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask):
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)
