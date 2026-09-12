"""
slurp_audio_model.py — End-to-end speech intent classification model for SLURP.

Architecture: pretrained speech encoder (wav2vec2 / HuBERT / WavLM, anything
that returns `last_hidden_state` over time) + masked mean pooling + linear
heads for scenario / action / intent classification directly from raw audio.

This is the standard "E2E SLU" setup used in most SLURP speech baselines
(e.g. Lugosch et al., "Speech Model Pre-training for End-to-End Spoken
Language Understanding"). It classifies intent directly from audio — no ASR
transcript needed for this part.

Slot/entity extraction is NOT done end-to-end here (see the note in
slurp_audio_dataset.py about why: no audio-aligned span labels exist in
SLURP). Use cascade_asr_nlu.py for slots.

Pick a real pretrained checkpoint when you actually train, e.g.:
    "facebook/wav2vec2-base-960h"
    "facebook/wav2vec2-large-960h-lv60-self"
    "microsoft/wavlm-base-plus"
Downloading these requires internet access to the HF Hub in your own environment.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class SlurpAudioIntentModel(nn.Module):
    def __init__(self, encoder_name, num_scenarios, num_actions, num_intents,
                 dropout=0.1, freeze_feature_encoder=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)

        if freeze_feature_encoder and hasattr(self.encoder, "feature_extractor"):
            # Freeze the CNN feature extractor (standard practice — it's already
            # a good low-level audio feature extractor after pretraining, and
            # freezing it makes fine-tuning much cheaper/more stable).
            for p in self.encoder.feature_extractor.parameters():
                p.requires_grad = False

        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.scenario_head = nn.Linear(hidden, num_scenarios)
        self.action_head = nn.Linear(hidden, num_actions)
        self.intent_head = nn.Linear(hidden, num_intents)

    def _masked_mean_pool(self, hidden_states, feature_mask):
        # hidden_states: [B, T', H], feature_mask: [B, T'] (1 = real frame, 0 = padding)
        mask = feature_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, T', 1]
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def forward(self, input_values, attention_mask):
        out = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden_states = out.last_hidden_state  # [B, T', H] (T' = downsampled frame count)

        # project the raw-waveform attention_mask down to the encoder's
        # downsampled frame count, so pooling ignores padded frames
        feature_mask = self.encoder._get_feature_vector_attention_mask(
            hidden_states.shape[1], attention_mask
        ) if hasattr(self.encoder, "_get_feature_vector_attention_mask") else \
            torch.ones(hidden_states.shape[:2], device=hidden_states.device)

        pooled = self.dropout(self._masked_mean_pool(hidden_states, feature_mask))
        scenario_logits = self.scenario_head(pooled)
        action_logits = self.action_head(pooled)
        intent_logits = self.intent_head(pooled)
        return scenario_logits, action_logits, intent_logits


class SlurpAudioLoss(nn.Module):
    """Sum of CE losses over scenario, action, and intent heads.
    (intent is scenario_action combined, so this is a little redundant by
    design — it mirrors how most SLURP baselines report all three; drop the
    scenario/action terms if you only care about the 101-way intent head.)"""

    def __init__(self, scenario_weight=0.0, action_weight=0.0, intent_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.w_s = scenario_weight
        self.w_a = action_weight
        self.w_i = intent_weight

    def forward(self, scenario_logits, action_logits, intent_logits,
                scenario_labels, action_labels, intent_labels):
        loss = 0.0
        if self.w_s > 0:
            loss = loss + self.w_s * self.ce(scenario_logits, scenario_labels)
        if self.w_a > 0:
            loss = loss + self.w_a * self.ce(action_logits, action_labels)
        loss = loss + self.w_i * self.ce(intent_logits, intent_labels)
        return loss
