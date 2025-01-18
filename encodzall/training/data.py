from typing import Optional, Tuple, Any
import torch
from encodzall import ByteLevelTokenizer
from encodzall.config.training_config import TrainingConfig


def prepare_batch(
    batch,
    tokenizer: ByteLevelTokenizer,
    config: TrainingConfig,
    device: torch.device,
    noise_prob: Optional[float] = None,
) -> Tuple:
    """
    Tokenize and pad the batch data.
    """
    texts = batch["text"]
    token_ids, attention_masks, word_boundaries = [], [], []
    sequence_ids, seq_target_ids, word_target_ids = [], [], []

    for j, text in enumerate(texts):
        tokens, mask, boundaries, targets = tokenizer.tokenize(text)
        token_ids.append(tokens)
        attention_masks.append(mask)
        word_boundaries.extend(boundaries)
        seq_target_ids.append([item for sublist in targets for item in sublist])
        word_target_ids.extend(targets)
        sequence_ids.extend([j] * sum(len(x) for x in boundaries))

    tokens_tensor = torch.cat(token_ids).to(device)
    attention_mask_tensor = torch.cat(attention_masks).to(device)

    # Generate target masks for reconstruction
    seq_target_ids, seq_key_padding_mask = tokenizer.pad_targets(seq_target_ids)
    seq_target_ids = seq_target_ids.to(device)
    seq_key_padding_mask = seq_key_padding_mask.to(device)

    word_target_ids, word_key_padding_mask = tokenizer.pad_targets(word_target_ids)
    word_target_ids = word_target_ids.to(device)
    word_key_padding_mask = word_key_padding_mask.to(device)

    return (
        tokens_tensor,
        attention_mask_tensor,
        word_boundaries,
        sequence_ids,
        seq_target_ids,
        seq_key_padding_mask,
        word_target_ids,
        word_key_padding_mask,
    )


def prepare_batch_stage2(
    batch,
    tokenizer: ByteLevelTokenizer,
    config: TrainingConfig,
    device: torch.device,
    # anchor_noise: Optional[float] = None,
) -> Tuple[Tuple, Tuple]:
    """
    Prepare data for Stage-2 contrastive training.

    We produce:
      - anchor_inputs (noisy) => (tokens_tensor, attn_mask, word_boundaries, sequence_ids,
                                  seq_target_ids, seq_key_padding_mask, word_target_ids, word_key_padding_mask)
      - pos_inputs    (clean) => same structure, but with noise=0

    Return: (anchor_inputs, pos_inputs)
      Each is a big tuple of the form we expect in the stage-2 training step.
    """
    texts = batch["text"]

    ## Optionally set anchor noise if provided (or you could rely on your PID outside)
    # old_prob = tokenizer.noise_config.prob
    # if anchor_noise is not None:
    #    tokenizer.noise_config.set_prob(anchor_noise)

    # ---------------------------
    # 1) Build the Anchor Inputs
    # ---------------------------
    anchor_token_ids, anchor_attention_masks, anchor_word_boundaries = [], [], []
    anchor_sequence_ids, anchor_seq_target_ids, anchor_word_target_ids = [], [], []

    for j, text in enumerate(texts):
        # "tokenize" is your custom method that respects noise_config.prob
        tokens, mask, boundaries, targets = tokenizer.tokenize(text)

        anchor_token_ids.append(tokens)
        anchor_attention_masks.append(mask)
        anchor_word_boundaries.extend(boundaries)

        # Flatten for sequence-level
        anchor_seq_target_ids.append([item for sublist in targets for item in sublist])
        # Keep the list-of-lists for word-level
        anchor_word_target_ids.extend(targets)
        # Expand sequence_ids
        anchor_sequence_ids.extend([j] * sum(len(x) for x in boundaries))

    anchor_tokens_tensor = torch.cat(anchor_token_ids).to(device)
    anchor_attention_mask_tensor = torch.cat(anchor_attention_masks).to(device)

    # Pad sequence-level
    anchor_seq_target_ids, anchor_seq_key_padding_mask = tokenizer.pad_targets(
        anchor_seq_target_ids
    )
    anchor_seq_target_ids = anchor_seq_target_ids.to(device)
    anchor_seq_key_padding_mask = anchor_seq_key_padding_mask.to(device)

    # # Pad word-level (though stage-2 might not use word recon, we match the shape anyway)
    # anchor_word_target_ids, anchor_word_key_padding_mask = tokenizer.pad_targets(anchor_word_target_ids)
    # anchor_word_target_ids = anchor_word_target_ids.to(device)
    # anchor_word_key_padding_mask = anchor_word_key_padding_mask.to(device)

    anchor_inputs = (
        anchor_tokens_tensor,
        anchor_attention_mask_tensor,
        anchor_word_boundaries,
        anchor_sequence_ids,
        anchor_seq_target_ids,
        anchor_seq_key_padding_mask,
        None,
        None,
    )

    # ----------------------------------
    # 2) Build the Positive (Clean) pass
    # ----------------------------------
    # Force noise=0 for the clean pass
    tokenizer.noise_config.set_prob(0.0)

    pos_token_ids, pos_attention_masks, pos_word_boundaries = [], [], []
    pos_sequence_ids, pos_seq_target_ids, pos_word_target_ids = [], [], []

    for j, text in enumerate(texts):
        tokens, mask, boundaries, targets = tokenizer.tokenize(text)

        pos_token_ids.append(tokens)
        pos_attention_masks.append(mask)
        pos_word_boundaries.extend(boundaries)

        pos_seq_target_ids.append([item for sublist in targets for item in sublist])
        pos_word_target_ids.extend(targets)
        pos_sequence_ids.extend([j] * sum(len(x) for x in boundaries))

    pos_tokens_tensor = torch.cat(pos_token_ids).to(device)
    pos_attention_mask_tensor = torch.cat(pos_attention_masks).to(device)

    pos_seq_target_ids, pos_seq_key_padding_mask = tokenizer.pad_targets(
        pos_seq_target_ids
    )
    pos_seq_target_ids = pos_seq_target_ids.to(device)
    pos_seq_key_padding_mask = pos_seq_key_padding_mask.to(device)

    # pos_word_target_ids, pos_word_key_padding_mask = tokenizer.pad_targets(pos_word_target_ids)
    # pos_word_target_ids = pos_word_target_ids.to(device)
    # pos_word_key_padding_mask = pos_word_key_padding_mask.to(device)

    pos_inputs = (
        pos_tokens_tensor,
        pos_attention_mask_tensor,
        pos_word_boundaries,
        pos_sequence_ids,
        None,
        None,
        None,
        None,
    )

    # Restore noise prob
    # tokenizer.noise_config.set_prob(old_prob)

    return anchor_inputs, pos_inputs
