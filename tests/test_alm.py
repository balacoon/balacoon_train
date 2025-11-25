"""
Copyright 2025 Balacoon

Tests for ALM model
"""

import pytest
import torch
import torch.nn.functional as F
from balacoon_train.alm import ALMModel
from balacoon_train.config import Config
from balacoon_train.data.container import Container


@pytest.fixture
def config():
    """Create a minimal config for testing to ensure tests run fast."""
    conf = Config()
    # Encoder settings
    conf.enc_layers_num = 2
    conf.enc_dim = 32
    conf.enc_heads_num = 4
    conf.enc_vocab_size = 50
    conf.phonemes_num = 50
    conf.pitch_num = 20

    # Decoder settings
    conf.dec_layers_num = 2
    conf.dec_dim = 32
    conf.dec_heads_num = 4
    conf.dec_vocab_size = 50
    conf.dec_vocabs_num = 2  # 2 acoustic tokens per step

    conf.max_position_embeddings = 128
    conf.learning_rate = 1e-4
    return conf


@pytest.fixture
def model(config):
    """Initialize model with test config."""
    return ALMModel(config)


@pytest.fixture
def batch_data(config):
    """Create a dummy batch."""
    batch_size = 2
    text_seq_len = 10
    acoustic_seq_len = 10

    text_tokens = torch.randint(0, config.phonemes_num, (batch_size, text_seq_len))
    pitch = torch.randint(0, config.pitch_num, (batch_size, text_seq_len))
    acoustic_tokens = torch.randint(
        0, config.dec_vocab_size, (batch_size, acoustic_seq_len, config.dec_vocabs_num)
    )

    container = Container(["dummy_id_1", "dummy_id_2"])
    container["text_tokens"] = text_tokens
    container["phoneme"] = text_tokens
    container["pitch"] = pitch
    container["acoustic_tokens"] = acoustic_tokens
    container["acoustic_tokens_len"] = torch.tensor([acoustic_seq_len] * batch_size)
    container["phoneme_len"] = torch.tensor([text_seq_len] * batch_size)
    container["prompt_len"] = torch.tensor([5] * batch_size)
    container["text_attention_mask"] = torch.ones_like(text_tokens)
    container["acoustic_attention_mask"] = torch.ones(batch_size, acoustic_seq_len)

    return container


def test_initialization(model, config):
    """Test if model initializes with correct submodules."""
    assert hasattr(model, "encoder")
    assert hasattr(model, "decoder")
    assert hasattr(model, "acoustic_embedding")
    assert len(model.acoustic_embedding) == config.dec_vocabs_num
    assert hasattr(model, "text_projection")
    assert hasattr(model, "acoustic_projection")


def test_forward_shape(model, batch_data, config):
    """Test forward pass output shapes."""
    outputs = model.forward(
        acoustic_tokens=batch_data["acoustic_tokens"],
        text_tokens=batch_data["text_tokens"],
        pitch=batch_data["pitch"],
        sequence_lens=batch_data["acoustic_tokens_len"],
    )

    logits = outputs["logits"]
    batch_size = batch_data["text_tokens"].size(0)
    acoustic_seq_len = batch_data["acoustic_tokens"].size(1)

    # Expected shape: [batch, seq_len, num_vocabs, vocab_size]
    expected_shape = (batch_size, acoustic_seq_len, config.dec_vocabs_num, config.dec_vocab_size)
    assert logits.shape == expected_shape


def test_encoder_bidirectionality(model, batch_data):
    """
    Test if encoder is non-causal (bidirectional).
    Output at position T should depend on input at T+k.
    """
    text_tokens = batch_data["text_tokens"]
    pitch = batch_data["pitch"]
    batch_size, seq_len = text_tokens.shape

    # We need embeddings to compute gradients w.r.t. them
    text_embeds = model.phonemes_embedding(text_tokens) + model.pitch_embedding(pitch)
    text_embeds.retain_grad()

    # Manually construct mask as in forward()
    bidirectional_mask = batch_data["text_attention_mask"].unsqueeze(1).unsqueeze(2)
    bidirectional_mask = bidirectional_mask.expand(batch_size, 1, seq_len, seq_len)
    encoder_attention_mask = torch.where(
        bidirectional_mask.bool(),
        torch.zeros_like(bidirectional_mask, dtype=torch.float),
        torch.full_like(bidirectional_mask, float("-inf"), dtype=torch.float),
    )

    encoder_outputs = model.encoder(
        inputs_embeds=text_embeds, attention_mask=encoder_attention_mask, return_dict=True
    )
    last_hidden_state = encoder_outputs.last_hidden_state

    # Check dependency of output at pos 0 on input at pos 5 (future)
    target = last_hidden_state[0, 0].sum()
    target.backward()

    grad = text_embeds.grad
    # Gradient at pos 5 should be non-zero if bidirectional
    is_bidirectional = torch.norm(grad[0, 5]) > 0
    assert (
        is_bidirectional
    ), "Encoder should be bidirectional (output at pos 0 depends on input at pos 5)"


def test_decoder_causality(model, batch_data):
    """
    Test if decoder is causal.
    Output at position T should NOT depend on input at T+k.
    """
    text_tokens = batch_data["text_tokens"]
    pitch = batch_data["pitch"]
    acoustic_tokens = batch_data["acoustic_tokens"]
    batch_size, acoustic_seq_len, _ = acoustic_tokens.shape

    # Prepare encoder output
    with torch.no_grad():
        text_embeds = model.phonemes_embedding(text_tokens) + model.pitch_embedding(pitch)
        # We simplify mask creation here as we just need encoder output
        encoder_outputs = model.encoder(inputs_embeds=text_embeds, return_dict=True)
        encoder_projected = model.text_projection(encoder_outputs.last_hidden_state)

    # Prepare acoustic embeddings manually to check gradients
    acoustic_embeds_list = []
    for i, emb in enumerate(model.acoustic_embedding):
        acoustic_embeds_list.append(emb(acoustic_tokens[:, :, i]))
    acoustic_embeds = torch.stack(acoustic_embeds_list, dim=2).sum(dim=2)
    acoustic_embeds.retain_grad()

    # Combine with encoder (resize encoder if needed, here just simplified logic matching forward)
    if encoder_projected.size(1) != acoustic_seq_len:
        encoder_projected = torch.nn.functional.adaptive_avg_pool1d(
            encoder_projected.transpose(1, 2), acoustic_seq_len
        ).transpose(1, 2)

    combined_embeds = acoustic_embeds + encoder_projected

    # Create causal mask
    attn_mask = torch.ones((batch_size, acoustic_seq_len), device=text_tokens.device)

    # Run decoder
    decoder_outputs = model.decoder.model(
        inputs_embeds=combined_embeds, attention_mask=attn_mask, return_dict=True
    )
    dec_hidden = decoder_outputs.last_hidden_state

    # Check dependency: Output at pos 0 should NOT depend on Input at pos 1
    target_dec = dec_hidden[0, 0].sum()
    target_dec.backward()

    grad_acoustic = acoustic_embeds.grad
    grad_at_future = torch.norm(grad_acoustic[0, 1])

    assert (
        grad_at_future == 0
    ), f"Decoder should be causal, but gradient at future pos is {grad_at_future}"


def test_training_step(model, batch_data):
    """Test training step returns a valid loss."""
    loss = model.training_step(batch_data, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_validation_step(model, batch_data):
    """Test validation step returns a valid loss."""
    loss = model.validation_step(batch_data, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert not torch.isnan(loss)


def test_generate(model, batch_data, config):
    """Test autoregressive generation."""
    batch_size = 2
    text_seq_len = 20
    prompt_len = 5

    # Full text tokens
    text_tokens = torch.randint(0, config.phonemes_num, (batch_size, text_seq_len))
    pitch = torch.randint(0, config.pitch_num, (batch_size, text_seq_len))

    # Prompt acoustic tokens
    acoustic_prompt = torch.randint(
        0, config.dec_vocab_size, (batch_size, prompt_len, config.dec_vocabs_num)
    )

    container = Container(["gen_id_1", "gen_id_2"])
    container["phoneme"] = text_tokens
    container["pitch"] = pitch
    container["ref_acoustic_tokens"] = acoustic_prompt
    container["phoneme_len"] = torch.tensor([text_seq_len] * batch_size)

    # Set generation params in config
    model.config.temperature = 1.0
    model.config.top_k = 10

    generated_tokens = model.generate(container)

    # Check shape
    # generate returns only new tokens (after prompt)
    expected_len = text_seq_len - prompt_len
    assert generated_tokens.shape == (batch_size, expected_len, config.dec_vocabs_num)
