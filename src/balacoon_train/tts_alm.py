"""
Copyright 2025 Balacoon

Llama-based acoustic token prediction.
"""

from typing import Any, Dict, Optional, Tuple, cast
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from balacoon_train.config import Config
from balacoon_train.data.container import Container


class TTSALMModel(LightningModule):
    """
    Text-to-speech acoustic Language Model using Llama3 architecture.

    Components:
    - Non-causal Llama3 encoder (100M params): encodes text tokens
    - Causal Llama3 decoder (300M params): takes acoustic tokens, adds encoder embeddings,
      and predicts multiple acoustic tokens per step
    """

    def __init__(self, config: Config):
        """
        Initialize the ALM model.

        Parameters
        ----------
        config: Config
            Configuration object containing model hyperparameters
        """
        super().__init__()
        config.add_missing(TTSALMConfig)
        self.config = config

        self.phonemes_emb = nn.Embedding(self.config.phonemes_num, self.config.enc_dim)

        # Encoder configuration (100M params, non-causal)
        encoder_config = LlamaConfig(
            vocab_size=1,  # Dummy value, as we provide custom embeddings
            hidden_size=self.config.enc_dim,
            intermediate_size=4 * self.config.enc_dim,
            num_hidden_layers=self.config.enc_layers_num,
            num_attention_heads=self.config.enc_heads_num,
            num_key_value_heads=self.config.enc_heads_num,
            max_position_embeddings=self.config.max_position_embeddings,
            rms_norm_eps=1e-6,
            use_cache=False,
        )

        # Decoder configuration (300M params, causal)
        decoder_config = LlamaConfig(
            vocab_size=1,  # Dummy value, as we provide custom embeddings
            hidden_size=self.config.dec_dim,
            intermediate_size=4 * self.config.dec_dim,
            num_hidden_layers=self.config.dec_layers_num,
            num_attention_heads=self.config.dec_heads_num,
            num_key_value_heads=self.config.dec_heads_num,
            max_position_embeddings=self.config.max_position_embeddings,
            rms_norm_eps=1e-6,
            use_cache=True,
        )

        # Initialize encoder (non-causal, bidirectional attention)
        self.encoder = LlamaModel(encoder_config)
        self.encoder.config.use_cache = False

        # Initialize decoder (causal)
        self.decoder = LlamaForCausalLM(decoder_config)

        # Acoustic token embedding (separate from decoder's embedding)
        self.acoustic_embedding = nn.ModuleList(
            [
                nn.Embedding(self.config.dec_vocab_size + 1, self.config.dec_dim)
                for _ in range(self.config.dec_vocabs_num)
            ]
        )

        # Projection layer to combine encoder and acoustic embeddings
        self.text_projection = nn.Linear(self.config.enc_dim, self.config.dec_dim)

        # Output projection for multi-token prediction
        self.acoustic_projection = nn.Linear(
            self.config.dec_dim, (self.config.dec_vocab_size + 1) * self.config.dec_vocabs_num
        )
        self.eop_projection = nn.Linear(self.config.dec_dim, 1)

    def forward(
        self,
        acoustic_tokens: torch.Tensor,
        phonemes: torch.Tensor,
        phoneme_indices: torch.Tensor,
        tokens_len: Optional[torch.Tensor] = None,
        phonemes_len: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Parameters
        ----------
        acoustic_tokens: torch.Tensor
            Acoustic token indices of shape [batch_size, max_tokens_len, vocabs_num]
        phonemes: torch.Tensor
            Phoneme ids [batch_size, max_phonemes_len]
        phoneme_indices: torch.Tensor
            indices which specifies which phoneme to use for each token.
            we upsample the encoder output using this tensor in `torch.gather`.
            shape: [batch_size, max_tokens_len]
        tokens_len: torch.Tensor
            number of acoustic tokens in every sequence, sequences are padded on the left
        phonemes_len: torch.Tensor
            number of phonemes in sequences, padded on the right

        Returns
        -------
        logits: predicted acoustic tokens logits of shape
            [batch_size, max_tokens_len, dec_vocabs_num, dec_vocab_size + 1]
        eop_logits: logits for binary end-of-phoneme prediction
            [batch_size, max_tokens_len, 1]
        """
        device = phonemes.device

        # 1. create a mask for the phonemes to be used in the encoder
        if phonemes_len is None:
            # no phonemes length, dummy maks
            phonemes_mask = torch.ones_like(phonemes)
            # will be created automatically by the encoder
            position_ids = None
        else:
            batch_size, seq_len = phonemes.shape
            # Create mask [batch, seq_len]
            idx = torch.arange(seq_len, device=device).expand(batch_size, seq_len)
            # Left padding: valid tokens are at the end of the sequence
            # indices [L-len, ..., L-1] should be 1
            phonemes_mask = (idx >= (seq_len - phonemes_len.unsqueeze(1))).long()
            # Create position_ids that ignore left padding
            # valid tokens should start at position 0
            # mask: 0 0 1 1 -> cumsum: 0 0 1 2 -> sub 1: -1 -1 0 1 -> relu: 0 0 0 1
            position_ids = phonemes_mask.cumsum(dim=1) - 1
            position_ids.clamp_(min=0)

        # Create bidirectional mask: [batch_size, num_heads, seq_len, seq_len]
        phonemes_mask = phonemes_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        phonemes_mask = phonemes_mask.expand(batch_size, 1, seq_len, seq_len)
        phonemes_attention_mask = torch.where(
            phonemes_mask.bool(),
            torch.zeros_like(phonemes_mask, dtype=torch.float),
            torch.full_like(phonemes_mask, float("-inf"), dtype=torch.float),
        )

        # 2. run the bidirectional encoder and process phonemes
        phonemes_embeds = self.phonemes_emb(phonemes)  # [batch_size, seq_len, enc_dim]
        encoder_outputs = self.encoder(
            inputs_embeds=phonemes_embeds,
            attention_mask=phonemes_attention_mask,  # 4d mask overwrites internal transformers mask
            position_ids=position_ids,
            return_dict=True,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state  # [batch_size, seq_len, enc_dim]
        encoder_projected = self.text_projection(
            encoder_hidden_states
        )  # [batch_size, seq_len, dec_dim]

        # 3. upsample the encoder output using torch.gather
        indices_expanded = phoneme_indices.unsqueeze(-1).expand(-1, -1, encoder_projected.size(-1))
        phonemes_up_embeds = torch.gather(encoder_projected, dim=1, index=indices_expanded)

        # 4. prepare mask for decoder, tokens are on the left
        batch_size, seq_len, _ = acoustic_tokens.shape
        if tokens_len is None:
            # no tokens length, dummy maks
            tokens_mask = torch.ones_like(acoustic_tokens)
            # will be created automatically by the decoder
            position_ids = None
        else:
            # Create mask [batch, seq_len]
            idx = torch.arange(seq_len, device=device).expand(batch_size, seq_len)
            # Left padding: valid tokens are at the end of the sequence
            # indices [L-len, ..., L-1] should be 1
            tokens_mask = (idx >= (seq_len - tokens_len.unsqueeze(1))).long()
            # Create position_ids that ignore left padding
            # valid tokens should start at position 0
            # mask: 0 0 1 1 -> cumsum: 0 0 1 2 -> sub 1: -1 -1 0 1 -> relu: 0 0 0 1
            position_ids = tokens_mask.cumsum(dim=1) - 1
            position_ids.clamp_(min=0)

        # 5. prepare acoustic tokens embeddings as input for decoder
        # Embed acoustic tokens
        # acoustic_tokens shape: [batch, acoustic_seq_len, vocabs_num]
        # Embed each vocabulary separately and sum
        acoustic_embeds_list = []
        for i, emb in enumerate(self.acoustic_embedding):
            acoustic_embeds_list.append(emb(acoustic_tokens[:, :, i]))
        acoustic_embeds = torch.stack(acoustic_embeds_list, dim=2).sum(dim=2)

        # 6. combine upsampled phonemes and acoustic tokens embeddings,
        # creating input for the decoder
        combined_embeds = phonemes_up_embeds + acoustic_embeds

        # 7. finally run decoder
        decoder_outputs = self.decoder.model(
            inputs_embeds=combined_embeds,
            attention_mask=tokens_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        decoder_hidden_states = decoder_outputs.last_hidden_state

        # Project to output logits
        logits = self.acoustic_projection(decoder_hidden_states)
        logits = logits.view(
            batch_size, seq_len, self.config.dec_vocabs_num, self.config.dec_vocab_size + 1
        )

        eop_logits = self.eop_projection(decoder_hidden_states)
        return logits, eop_logits

    @torch.no_grad()
    def generate(
        self,
        batch: Container,
        out_dir: str,
    ) -> None:
        """
        Autoregressive generation of acoustic tokens.

        Parameters
        ----------
        batch: Container
            Container with inputs. Must contain:
            - phoneme: [batch, total_seq_len, vocab_size] - combines reference and target phonemes
            - pitch: [batch, total_seq_len] - combines reference and target pitch
            - ref_phoneme_len: [batch,] - length of reference sequences without left padding
            - phoneme_len [batch,] - length of target sequences without right padding
            - ref_acoustic_tokens: [batch, prompt_len, vocabs_num] - contains only reference acoustic tokens
        """
        self.eval()
        batch = batch.to(self.device)
        phoneme_probs = cast(
            torch.Tensor, batch["phoneme"]
        )  # phoneme probabilities for ref + target
        pitch = cast(torch.Tensor, batch["pitch"])  # total pitch
        acoustic_prompt = cast(torch.Tensor, batch["ref_acoustic_tokens"])
        device = phoneme_probs.device

        # Setup logits processors
        logits_processor = LogitsProcessorList()
        if self.config.temperature != 1.0:
            logits_processor.append(TemperatureLogitsWarper(self.config.temperature))
        if self.config.top_k > 0:
            logits_processor.append(TopKLogitsWarper(self.config.top_k))
        if self.config.top_p < 1.0:
            logits_processor.append(TopPLogitsWarper(self.config.top_p))

        batch_size, total_len, vocab_size = phoneme_probs.shape
        prompt_len = acoustic_prompt.shape[1]

        # 1. Encode full text once
        target_lens = cast(
            torch.Tensor, batch["phoneme_len"]
        )  # targets length wihtput padding on right
        source_lens = cast(
            torch.Tensor, batch["ref_phoneme_len"]
        )  # source length without padding on the left
        # left_padding[i] = prompt_len - source_lens[i]
        # right_padding[i] = total_len - prompt_len - target_lens[i]
        # Create mask [B, L]
        idx = torch.arange(total_len, device=device).unsqueeze(0)
        left_pad = prompt_len - source_lens
        right_pad = total_len - prompt_len - target_lens
        valid_mask = (idx >= left_pad.unsqueeze(1)) & (idx < (total_len - right_pad.unsqueeze(1)))
        position_ids = valid_mask.cumsum(dim=1) - 1
        position_ids.clamp_(min=0)

        bidirectional_mask = valid_mask.unsqueeze(1).unsqueeze(2)
        bidirectional_mask = bidirectional_mask.expand(batch_size, 1, total_len, total_len)
        encoder_attention_mask = torch.where(
            bidirectional_mask.bool(),
            torch.zeros_like(bidirectional_mask, dtype=torch.float),
            torch.full_like(bidirectional_mask, float("-inf"), dtype=torch.float),
        )
        text_embeds = self.phonemes_proj(phoneme_probs) + self.pitch_embedding(pitch)
        encoder_outputs = self.encoder(
            inputs_embeds=text_embeds,
            position_ids=position_ids,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        # [batch, total_len, dec_dim]
        text_embeddings = self.text_projection(encoder_outputs.last_hidden_state)

        # 2. Setup generation
        generated = acoustic_prompt.clone()  # [batch, prompt_len, vocabs_num]
        past_key_values = None

        # 3. Prefill prompt
        # We feed the prompt. The last token of prompt + aligned text predicts the first new token.
        # Embed prompt
        acoustic_embeds_list = []
        for i, emb in enumerate(self.acoustic_embedding):
            acoustic_embeds_list.append(emb(acoustic_prompt[:, :, i]))
        acoustic_embeds = torch.stack(acoustic_embeds_list, dim=2).sum(
            dim=2
        )  # [B, prompt_len, dec_dim]

        # Align text embeddings: use corresponding slice
        current_text_embeds = text_embeddings[:, :prompt_len, :]

        # Combine
        inputs_embeds = acoustic_embeds + current_text_embeds

        # Forward prompt
        outputs = self.decoder.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=valid_mask[:, :prompt_len],
            position_ids=position_ids[:, :prompt_len],
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        last_hidden = outputs.last_hidden_state[:, -1:, :]  # [B, 1, hidden]

        # 4. Autoregressive loop
        # We start generating from index `prompt_len` up to `total_len`
        for i in range(prompt_len, total_len):
            # Project to logits
            logits = self.acoustic_projection(last_hidden)  # [B, 1, num_vocabs * vocab_size]
            logits = logits.view(
                batch_size, 1, self.config.dec_vocabs_num, self.config.dec_vocab_size + 1
            )

            # Sample next token (all codebooks in parallel)
            next_tokens = []
            for v in range(self.config.dec_vocabs_num):
                v_logits = logits[:, 0, v, :]  # batch x vocab_size
                v_logits = logits_processor(generated[:, :, v], v_logits)
                probs = F.softmax(v_logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)  # [B, 1]
                next_tokens.append(token)

            next_acoustic_token = torch.cat(next_tokens, dim=1).unsqueeze(1)  # [B, 1, vocabs_num]

            # Append to result
            generated = torch.cat([generated, next_acoustic_token], dim=1)

            # Prepare input for next step
            # Next input is: newly generated token + text embedding at current pos `i`

            # Embed new acoustic token
            next_acoustic_embeds_list = []
            for v_idx, emb in enumerate(self.acoustic_embedding):
                next_acoustic_embeds_list.append(emb(next_acoustic_token[:, :, v_idx]))
            next_acoustic_embed = torch.stack(next_acoustic_embeds_list, dim=2).sum(
                dim=2
            )  # [B, 1, dec_dim]

            # Get corresponding text embedding
            next_text_embed = text_embeddings[:, i : i + 1, :]  # [B, 1, dec_dim]

            inputs_embeds = next_acoustic_embed + next_text_embed

            # Forward single step
            outputs = self.decoder.model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                attention_mask=valid_mask[:, : (i + 1)],
                position_ids=position_ids[:, i : (i + 1)],
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            last_hidden = outputs.last_hidden_state

        batch["generated_acoustic_tokens"] = generated[:, prompt_len:, :].transpose(
            1, 2
        )  # B x 8 x time
        batch.save_to_npz(
            streams=["generated_acoustic_tokens"],
            out_dir=out_dir,
            seq_axis=[1],
            actual_lens=["phoneme_len"],
            rename=["acoustic_tokens"],  # save with standard name
        )

    def _calculate_loss(
        self,
        logits: torch.Tensor,
        eop_logits: torch.Tensor,
        acoustic_tokens: torch.Tensor,
        eop_flags: torch.Tensor,
        prompt_len: torch.Tensor,
        sequence_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate loss for next token prediction (no staggered shift).

        Parameters
        ----------
        logits: torch.Tensor
            Model output logits of shape [batch, seq_len, vocabs_num, vocab_size]
        eop_logits: torch.Tensor
            Model output logits for end-of-phoneme prediction of shape [batch, seq_len, 1]
        acoustic_tokens: torch.Tensor
            Target tokens of shape [batch, seq_len, vocabs_num]
        eop_flags: torch.Tensor
            Binary flags indicating end-of-phoneme of shape [batch, seq_len]
        prompt_len: torch.Tensor
            Prompt length specifying regions of tokens that should not be used for loss computation of shape [batch,]
        sequence_lens: torch.Tensor
            Sequence length of shape [batch,] specifies regions of acoustic tokens that are padded

        Returns
        -------
        loss: torch.Tensor
            Cross entropy loss
        """
        # Remove the last logit as we don't have a target for it (it predicts T+1)
        logits = logits[:, :-1, :, :]  # [B, T-1, V_num, V_size]
        eop_logits = eop_logits[:, :-1, :]  # [B, T-1, 1]

        # Remove the first token from targets as it's the input for the first prediction
        # Targets are shifted by 1 relative to input
        targets = acoustic_tokens[:, 1:, :]  # [B, T-1, V_num]
        eop_flags = eop_flags[:, 1:]  # [B, T-1]

        batch_size, seq_len, vocabs_num, vocab_size = logits.shape
        valid_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=logits.device)
        for i in range(batch_size):
            # Mask out the prompt region in targets so we don't compute loss on it.
            # The model is not trained to predict the prompt, but to continue from it.
            # With left padding:
            # Padding: [0, ..., seq_len - sequence_lens[i]]
            # Prompt starts after padding: [seq_len - sequence_lens[i], ..., seq_len - sequence_lens[i] + prompt_len[i]]
            # We mask everything from 0 up to prompt end.
            padding_len = seq_len - sequence_lens[i]
            mask_end = padding_len + prompt_len[i]
            targets[i, :mask_end, :] = -1
            valid_mask[i, mask_end:] = True

        # Flatten for loss computation
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1).long()
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)

        valid_mask_flat = valid_mask.reshape(-1)
        eop_logits_flat = eop_logits.reshape(-1)[valid_mask_flat]
        eop_flags_flat = eop_flags.reshape(-1)[valid_mask_flat].float()
        eop_loss = F.binary_cross_entropy_with_logits(eop_logits_flat, eop_flags_flat)

        return loss + eop_loss

    def _step(self, batch: Container) -> torch.Tensor:
        batch = batch.to(self.device)
        phonemes = cast(torch.Tensor, batch["phonemes"])
        phoneme_indices = cast(torch.Tensor, batch["phoneme_indices"])
        acoustic_tokens = cast(torch.Tensor, batch["acoustic_tokens"])
        tokens_len = cast(torch.Tensor, batch["tokens_len"])
        phonemes_len = cast(torch.Tensor, batch["phonemes_len"])

        # Forward pass
        logits, eop_logits = self.forward(
            acoustic_tokens=acoustic_tokens,
            phonemes=phonemes,
            phoneme_indices=phoneme_indices,
            tokens_len=tokens_len,
            phonemes_len=phonemes_len,
        )

        prompt_lens = cast(torch.Tensor, batch["prompt_len"])
        eop_flags = cast(torch.Tensor, batch["end_of_phoneme"])
        loss = self._calculate_loss(
            logits,
            eop_logits,
            acoustic_tokens.clone(),
            eop_flags,
            prompt_lens,
            tokens_len,
        )
        return loss

    def training_step(self, batch: Container, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Parameters
        ----------
        batch: Container
            Batch containing text_tokens, acoustic_tokens, etc.
        batch_idx: int
            Batch index

        Returns
        -------
        loss: torch.Tensor
            Training loss
        """
        loss = self._step(batch)
        batch_size = batch["phonemes"].shape[0]
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        return loss

    def validation_step(self, batch: Container, batch_idx: int) -> torch.Tensor:
        """
        Validation step.

        Parameters
        ----------
        batch: Container
            Batch containing text_tokens, acoustic_tokens, etc.
        batch_idx: int
            Batch index

        Returns
        -------
        loss: torch.Tensor
            Validation loss
        """
        loss = self._step(batch)
        batch_size = batch["phonemes"].shape[0]
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns
        -------
        optimizer: torch.optim.Optimizer
            Optimizer for training
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_steps,
            eta_min=self.config.min_learning_rate,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


@dataclass
class TTSALMConfig:
    cls: str = TTSALMModel.__module__ + "." + TTSALMModel.__name__
    # 768 / 21.5 = 35 seconds
    max_position_embeddings: int = 768

    # encoder (text) architecture
    enc_layers_num: int = 16
    enc_dim: int = 768
    enc_heads_num: int = 12
    phonemes_num: int = 47  # 47 phonemes for libritts alignment

    # decoder (acoustic) architecture
    dec_layers_num: int = 24
    dec_dim: int = 1024
    dec_heads_num: int = 16
    dec_vocab_size: int = 2024  # vocab size is the same for each codebook
    dec_vocabs_num: int = 8

    # optimization params
    learning_rate: float = 5e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.1
    num_steps: int = 100000
    precision: str = "bf16-mixed"

    # generation params
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
