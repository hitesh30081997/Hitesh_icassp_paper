"""
slurp_model.py — Joint intent classification + slot tagging model for SLURP.

Standard "joint NLU" architecture (Chen et al. 2019 style, as used in most
SLURP baselines): a pretrained transformer encoder with two heads on top:
    - intent head: linear layer on the [CLS] (first-token) pooled vector
    - slot head:   linear layer on every token's hidden state (token classification)

Works with any encoder-only or encoder-decoder-encoder HF model that returns
`last_hidden_state`, e.g. "bert-base-uncased", "distilbert-base-uncased",
"roberta-base", "microsoft/mdeberta-v3-base", etc. Pick a real pretrained
checkpoint name when you actually train (this requires internet access to
the HF Hub in your own environment).
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class SlurpJointModel(nn.Module):
    def __init__(self, encoder_name, num_intents, num_tags, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden, num_intents)
        self.slot_head = nn.Linear(hidden, num_tags)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state              # [B, T, H]
        pooled = self.dropout(seq[:, 0])          # first-token ([CLS]-like) pooling
        seq = self.dropout(seq)
        intent_logits = self.intent_head(pooled)  # [B, num_intents]
        slot_logits = self.slot_head(seq)         # [B, T, num_tags]
        return intent_logits, slot_logits


class SlurpJointLoss(nn.Module):
    """Combined loss = intent CE + slot_loss_weight * slot CE (ignoring -100)."""

    def __init__(self, slot_loss_weight=1.0):
        super().__init__()
        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.slot_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.slot_loss_weight = slot_loss_weight

    def forward(self, intent_logits, slot_logits, intent_labels, slot_labels):
        intent_loss = self.intent_loss_fn(intent_logits, intent_labels)
        slot_loss = self.slot_loss_fn(
            slot_logits.view(-1, slot_logits.size(-1)),
            slot_labels.view(-1),
        )
        total = intent_loss + self.slot_loss_weight * slot_loss
        return total, intent_loss, slot_loss
