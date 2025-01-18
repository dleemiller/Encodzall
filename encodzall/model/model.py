import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union

from contextlib import contextmanager

from ..config import TransformerConfig
from .encoder2 import TransformerEncoder
from .decoder2 import TransformerDecoder
from .positional_encoding import SinusoidalPositionalEncoding
from .word_pooling import WordPooling
from .unpad_sequences import UnpadSequences


class Encodzall(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=self.d_model
        )

        # Encoder 1
        self.encoder1 = TransformerEncoder(
            num_layers=config.num_encoder1_layers,
            d_model=self.d_model,
            nhead=config.nhead,
            num_kv_heads=config.num_kv_heads_encoder1 or config.nhead // 2,
            head_dim=self.d_model // config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            max_seq_len=config.max_seq_length_encoder1,
            is_causal=False,
        )

        # Encoder 2
        self.encoder2 = TransformerEncoder(
            num_layers=config.num_encoder2_layers,
            d_model=self.d_model,
            nhead=config.nhead,
            num_kv_heads=config.num_kv_heads_encoder2 or config.nhead // 2,
            head_dim=self.d_model // config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            max_seq_len=config.max_seq_length_encoder2,
            is_causal=False,
        )

        # Decoder for sequence-level reconstruction
        self.decoder = TransformerDecoder(
            num_layers=config.num_decoder_layers,
            d_model=self.d_model,
            nhead=config.nhead,
            num_kv_heads=config.num_kv_heads_decoder or config.nhead // 2,
            head_dim=self.d_model // config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            max_seq_len=config.max_seq_length_decoder,
            is_causal=True,
        )

        # Decoder for word-level reconstruction
        self.word_decoder = TransformerDecoder(
            num_layers=1,
            d_model=self.d_model,
            nhead=1,
            num_kv_heads=1,
            head_dim=self.d_model,
            dim_feedforward=config.dim_feedforward,
            dropout=0.0,
            attn_dropout=0.1,
            max_seq_len=64,
            is_causal=True,
        )

        # Pooling, unpadding, output
        self.word_pooling = WordPooling(pooling_type=config.pooling_type)
        self.unpad_sequences = UnpadSequences()
        self.output_layer = nn.Linear(self.d_model, config.vocab_size)

    @contextmanager
    def set_dropout(self, dropout: float):
        """
        Temporarily override dropout rate in all nn.Dropout modules.
        """
        original_dropout_rates = {}
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                original_dropout_rates[module] = module.p
                module.p = dropout
        try:
            yield
        finally:
            for module, original_rate in original_dropout_rates.items():
                module.p = original_rate

    @staticmethod
    def key_padding_to_attention_mask(key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Converts a key_padding_mask (True=pad) into a 2D attention mask (b, s, s).
        """
        inverted_mask = ~key_padding_mask  # True => valid
        attention_mask = inverted_mask.unsqueeze(1) & inverted_mask.unsqueeze(2)
        return attention_mask.bool()

    @staticmethod
    def create_memory_mask(
        memory_key_padding_mask: torch.Tensor,
        target_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create memory mask for the decoder:
            shape => (batch_size, tgt_len, mem_len)
            True in final mask means "allow attend".
        """
        memory_key_padding_mask = memory_key_padding_mask.bool()  # b x mem_len
        target_key_padding_mask = target_key_padding_mask.bool()  # b x tgt_len
        # expand memory_key_padding_mask to match target seq length
        memory_mask = memory_key_padding_mask.unsqueeze(1).expand(
            -1, target_key_padding_mask.size(1), -1
        )
        # PyTorch expects True => attend, so invert if needed
        return ~memory_mask

    @staticmethod
    def flatten_and_filter(tensor, key_padding_mask):
        """
        Flatten (b, s, d) -> (b*s, d) and remove pad rows based on mask.
        """
        if key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.bool()
        valid_mask = ~key_padding_mask  # True => valid
        b, s, d_model = tensor.shape
        flat_t = tensor.view(-1, d_model)  # (b*s, d_model)
        flat_m = valid_mask.view(-1)  # (b*s)
        filtered = flat_t[flat_m]  # (N, d_model)
        return filtered.unsqueeze(dim=1)  # (N, 1, d_model)

    @staticmethod
    def get_eos_mask(sequence_ids: torch.Tensor) -> torch.Tensor:
        """
        Identify final token in each sub-sequence after unpadding,
        to remove them from the 'word memory'.
        """
        # sequence_ids is shape [#words], we want to mark last token in each sequence group
        # This code is a bit unusual, but you can adapt as needed.
        mask = sequence_ids[:-1] != sequence_ids[1:]
        mask = torch.cat([mask, torch.tensor([True], device=sequence_ids.device)])
        # `mask` is True at boundaries => so `~mask` is everything except boundary
        return ~mask

    def average_pool_memory(
        self, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Average pool memory vectors across the unpadded dimension => (batch_size, d_model)
        """
        valid_mask = ~memory_key_padding_mask
        valid_mask_expanded = valid_mask.unsqueeze(-1).float()
        memory_masked = memory * valid_mask_expanded
        sum_memory = memory_masked.sum(dim=1)
        counts = valid_mask_expanded.sum(dim=1).clamp(min=1e-9)
        embeddings = sum_memory / counts
        return embeddings

    def forward(
        self,
        x: torch.Tensor,
        sequence_ids: torch.Tensor,
        seq_target_ids: Optional[torch.Tensor] = None,
        seq_key_padding_mask: Optional[torch.Tensor] = None,
        word_target_ids: Optional[torch.Tensor] = None,
        word_key_padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        word_boundaries: Optional[List[List[Tuple[int, int]]]] = None,
        return_embeddings_only: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass.

        In Stage 2’s “second pass” for contrastive positives,
        call with `return_embeddings_only=True` and no target IDs.

        Args:
            x: Input token IDs => (batch_size, seq_length)
            sequence_ids: Sub-sequence IDs for unpadding => (batch_size,) or a shape that your unpad logic expects
            seq_target_ids: Optional, used for sequence reconstruction => (batch_size, seq_tgt_len)
            seq_key_padding_mask: Optional, True=pad => (batch_size, seq_tgt_len)
            word_target_ids: Optional, used for word-level reconstruction => (batch_size, word_tgt_len)
            word_key_padding_mask: Optional => (batch_size, word_tgt_len)
            attention_mask: (batch_size, seq_len, seq_len) for first encoder
            word_boundaries: data for block-wise attention & pooling
            return_embeddings_only: if True, skip decoders and just return [batch_size, d_model] embeddings

        Returns:
            If return_embeddings_only == True:
                embeddings => (batch_size, d_model)
            Else:
                (embeddings, seq_logits, word_logits)
        """
        device = x.device

        # 1) Embedding
        embedded = self.embedding(x.long())  # (b, s, d_model)

        # 2) First Encoder
        #    e.g. block-diagonal attn for each "word"
        #    attention_mask => (b, s, s)
        enc1_out = self.encoder1(embedded, attn_mask=attention_mask.bool())

        # 3) Word-level pooling
        #    => shape (num_words_total, d_model)
        if word_boundaries is None:
            raise ValueError(
                "word_boundaries must be provided unless you change the pooling logic."
            )
        pooled_word_vectors = self.word_pooling(enc1_out, word_boundaries)

        # 4) Unpad sequences => pass (pooled_word_vectors, sequence_ids)
        enc1_unpadded, memory_key_padding_mask = self.unpad_sequences(
            pooled_word_vectors, sequence_ids
        )
        # => enc1_unpadded shape: (batch_size, *some_length*, d_model)

        # 5) Second Encoder
        mem_attn_mask = self.key_padding_to_attention_mask(memory_key_padding_mask)
        memory = self.encoder2(enc1_unpadded, attn_mask=mem_attn_mask.bool())

        # 6) Average pooling => final "embedding" for each sequence in the batch
        embeddings = self.average_pool_memory(memory, memory_key_padding_mask)
        # => shape (batch_size, d_model)

        # -----------------------------------------------------------------
        # If we only want embeddings (Stage 2 “clean” pass => contrastive positives),
        # skip decoders altogether.
        # -----------------------------------------------------------------
        if return_embeddings_only:
            return embeddings  # shape [b, d_model]

        # ==================
        # Sequence Decoder
        # ==================
        seq_logits = None
        if seq_target_ids is not None:
            # Embed target tokens
            seq_tgt_emb = self.embedding(seq_target_ids.long())
            # Generate causal mask => shape (tgt_len, tgt_len)
            seq_causal_mask = (
                nn.Transformer.generate_square_subsequent_mask(seq_target_ids.size(1))
                .bool()
                .to(device)
            )

            # Memory mask => shape (b, tgt_len, mem_len)
            if seq_key_padding_mask is not None:
                seq_mem_mask = self.create_memory_mask(
                    memory_key_padding_mask.bool(), seq_key_padding_mask.bool()
                )
            else:
                # If no pad mask for the target side, pass something that allows all tokens
                seq_mem_mask = None

            # TransformerDecoder expects True => attend or False => block.
            # By default, PyTorch’s 'mask' means True=ignore, so we often invert.
            # Depending on your usage, you may need to invert seq_causal_mask or not.
            # In your code, you used ~seq_causal_mask.unsqueeze(0).  Just be consistent:
            seq_decoder_out = self.decoder(
                tgt=seq_tgt_emb,
                memory=memory,
                tgt_mask=~seq_causal_mask.unsqueeze(
                    0
                ),  # or however you were applying it
                memory_mask=seq_mem_mask,
            )
            seq_logits = self.output_layer(seq_decoder_out)
            # Exclude last token for teacher-forcing next-token prediction
            seq_logits = seq_logits[:, :-1, :].contiguous()

        # ==================
        # Word Decoder
        # ==================
        word_logits = None
        if word_target_ids is not None:
            word_tgt_emb = self.embedding(word_target_ids.long())
            word_causal_mask = (
                nn.Transformer.generate_square_subsequent_mask(word_target_ids.size(1))
                .bool()
                .to(device)
            )

            # Flatten memory => shape (N, 1, d_model)
            # (where N is total # words after removing pad)
            word_memory = self.flatten_and_filter(memory, memory_key_padding_mask)
            # Also remove sequence-level EOS tokens
            eos_mask = self.get_eos_mask(sequence_ids)
            word_memory = word_memory[eos_mask]

            # Construct memory mask if needed
            # For a single “word sequence” shape mismatch,
            # you might need a custom approach.
            # Below is how you had it, but be sure the shapes align:
            if word_key_padding_mask is not None:
                word_mem_mask = self.create_memory_mask(
                    torch.zeros_like(
                        word_memory[..., 0], dtype=torch.bool, device=device
                    ),
                    word_key_padding_mask.bool(),
                )
            else:
                word_mem_mask = None

            word_decoder_out = self.decoder(
                tgt=word_tgt_emb,
                memory=word_memory,
                tgt_mask=~word_causal_mask.unsqueeze(0),
                memory_mask=word_mem_mask,
            )
            word_logits = self.output_layer(word_decoder_out)
            word_logits = word_logits[:, :-1, :].contiguous()

        return (embeddings, seq_logits, word_logits)
