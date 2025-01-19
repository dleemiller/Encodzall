# config.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    # Required Parameters (No default values)
    vocab_size: int
    d_model: int
    nhead: int
    num_encoder1_layers: int
    num_encoder2_layers: int
    num_decoder_layers: int
    activation: str
    dim_feedforward: int
    dropout: float
    attn_dropout: float
    max_seq_length_encoder1: int
    max_seq_length_encoder2: int
    max_seq_length_decoder: int
    pooling_type: str

    # Optional Parameters (Default to None)
    num_kv_heads_encoder1: Optional[int] = None
    num_kv_heads_encoder2: Optional[int] = None
    num_kv_heads_decoder: Optional[int] = None


# Configurations

# Small configuration
encodzall_xs = TransformerConfig(
    vocab_size=256,
    d_model=256,
    nhead=4,
    num_encoder1_layers=2,
    num_encoder2_layers=2,
    num_decoder_layers=1,
    activation="gelu",
    dim_feedforward=1536,
    dropout=0.0,
    attn_dropout=0.05,
    max_seq_length_encoder1=64,
    max_seq_length_encoder2=1024,
    max_seq_length_decoder=4096,
    pooling_type="average",
)

# Small configuration
encodzall_s = TransformerConfig(
    vocab_size=256,
    d_model=384,
    nhead=12,
    num_encoder1_layers=2,
    num_encoder2_layers=2,
    num_decoder_layers=2,
    activation="gelu",
    dim_feedforward=1536,
    dropout=0.0,
    attn_dropout=0.05,
    max_seq_length_encoder1=64,
    max_seq_length_encoder2=1024,
    max_seq_length_decoder=4096,
    pooling_type="average",
)

# Large configuration
encodzall_l = TransformerConfig(
    vocab_size=256,
    d_model=512,
    nhead=8,
    num_encoder1_layers=4,
    num_encoder2_layers=4,
    num_decoder_layers=8,
    activation="gelu",
    dim_feedforward=4096,
    dropout=0.0,
    attn_dropout=0.05,
    max_seq_length_encoder1=64,
    max_seq_length_encoder2=2048,
    max_seq_length_decoder=4096,
    pooling_type="average",
)
