"""
slurp_dataset.py — PyTorch Dataset for the official SLURP corpus.

Source dataset:
    Bastianelli, Vanzo, Swietojanski, Rieser.
    "SLURP: A Spoken Language Understanding Resource Package" (EMNLP 2020).
    https://github.com/pswietojanski/slurp

This wraps the official train.jsonl / train_synthetic.jsonl / devel.jsonl /
test.jsonl annotation files (the text/NLU portion of SLURP — audio is a
separate ~6GB download from Zenodo and isn't needed for a text NLU model)
and prepares SLURP's standard 3-way joint task:

    1. scenario classification   (18 classes)
    2. action classification     (54 classes)
    3. intent classification     (101 classes = scenario_action, given directly
                                   in the data as the "intent" field)
    4. entity/slot tagging       (56 slot types, BIO scheme) over subword tokens

Predicted slot tags can be converted back into {"type", "filler"} entities
with `decode_slots_to_entities`, which is exactly the format `slurp_f1.py`
expects for scoring (SLU-F1).

Requires a HuggingFace tokenizer (any AutoTokenizer). Any subword tokenizer
works as long as it exposes `is_split_into_words=True` + `.word_ids()`,
which all `transformers` "fast" tokenizers do.
"""

import json
from functools import partial

import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --------------------------------------------------------------------------
# Label vocabulary (build once from the training file(s), reuse everywhere)
# --------------------------------------------------------------------------

class SlurpLabelVocab:
    def __init__(self, scenarios, actions, intents, slot_types):
        self.scenario2id = {s: i for i, s in enumerate(sorted(scenarios))}
        self.action2id = {a: i for i, a in enumerate(sorted(actions))}
        self.intent2id = {it: i for i, it in enumerate(sorted(intents))}
        self.id2scenario = {i: s for s, i in self.scenario2id.items()}
        self.id2action = {i: a for a, i in self.action2id.items()}
        self.id2intent = {i: it for it, i in self.intent2id.items()}

        tags = ["O"]
        for t in sorted(slot_types):
            tags.append(f"B-{t}")
            tags.append(f"I-{t}")
        self.tag2id = {t: i for i, t in enumerate(tags)}
        self.id2tag = {i: t for t, i in self.tag2id.items()}

    @classmethod
    def build(cls, jsonl_paths):
        """Build the label vocabulary from one or more SLURP jsonl files.
        Typically pass both train.jsonl and train_synthetic.jsonl so the
        vocabulary covers every label that can appear."""
        scenarios, actions, intents, slots = set(), set(), set(), set()
        for p in jsonl_paths:
            for r in load_jsonl(p):
                scenarios.add(r["scenario"])
                actions.add(r["action"])
                intents.add(r["intent"])
                for e in r.get("entities", []):
                    slots.add(e["type"])
        return cls(scenarios, actions, intents, slots)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({
                "scenario2id": self.scenario2id,
                "action2id": self.action2id,
                "intent2id": self.intent2id,
                "tag2id": self.tag2id,
            }, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        obj = cls.__new__(cls)
        obj.scenario2id = d["scenario2id"]
        obj.action2id = d["action2id"]
        obj.intent2id = d["intent2id"]
        obj.tag2id = d["tag2id"]
        obj.id2scenario = {i: s for s, i in obj.scenario2id.items()}
        obj.id2action = {i: a for a, i in obj.action2id.items()}
        obj.id2intent = {i: it for it, i in obj.intent2id.items()}
        obj.id2tag = {i: t for t, i in obj.tag2id.items()}
        return obj

    @property
    def num_scenarios(self):
        return len(self.scenario2id)

    @property
    def num_actions(self):
        return len(self.action2id)

    @property
    def num_intents(self):
        return len(self.intent2id)

    @property
    def num_tags(self):
        return len(self.tag2id)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class SlurpDataset(Dataset):
    """
    One example = one (sentence, scenario, action, intent, entities) record
    from an official SLURP jsonl file.

    Note: SLURP has multiple audio `recordings` per sentence (different
    speakers/mics), but for text-NLU training you only need one example per
    sentence, which is what this class gives you.
    """

    def __init__(self, jsonl_path, label_vocab, tokenizer, max_length=64):
        self.records = load_jsonl(jsonl_path)
        self.vocab = label_vocab
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _word_bio_tags(record):
        tokens = record["tokens"]
        tags = ["O"] * len(tokens)
        for ent in record.get("entities", []):
            span = ent["span"]           # list of token ids belonging to this entity
            etype = ent["type"]
            for i, tok_id in enumerate(span):
                tags[tok_id] = ("B-" if i == 0 else "I-") + etype
        return tags

    def __getitem__(self, idx):
        r = self.records[idx]
        words = [t["surface"] for t in r["tokens"]]
        if not words:
            words = r["sentence"].split() or [""]
        word_tags = self._word_bio_tags(r) if r["tokens"] else ["O"] * len(words)

        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = enc.word_ids()

        # Project word-level BIO tags onto subwords: only the FIRST subword
        # of each word is supervised (standard practice for BERT-style NER);
        # special tokens and continuation subwords get -100 (ignored by loss).
        slot_labels = []
        prev_word = None
        for wid in word_ids:
            if wid is None:
                slot_labels.append(-100)
            elif wid != prev_word:
                slot_labels.append(self.vocab.tag2id[word_tags[wid]])
            else:
                slot_labels.append(-100)
            prev_word = wid

        gold_entities = [
            {"type": e["type"], "filler": " ".join(words[i] for i in e["span"])}
            for e in r.get("entities", [])
        ]

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "slot_labels": slot_labels,
            "scenario_label": self.vocab.scenario2id[r["scenario"]],
            "action_label": self.vocab.action2id[r["action"]],
            "intent_label": self.vocab.intent2id[r["intent"]],
            "sentence": r["sentence"],
            "words": words,
            "word_ids": word_ids,
            "gold_entities": gold_entities,
        }


# --------------------------------------------------------------------------
# Collation (dynamic padding per batch)
# --------------------------------------------------------------------------

def _collate(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids, attn, slot_labels = [], [], []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        attn.append(x["attention_mask"] + [0] * pad_len)
        slot_labels.append(x["slot_labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "slot_labels": torch.tensor(slot_labels, dtype=torch.long),
        "scenario_label": torch.tensor([x["scenario_label"] for x in batch], dtype=torch.long),
        "action_label": torch.tensor([x["action_label"] for x in batch], dtype=torch.long),
        "intent_label": torch.tensor([x["intent_label"] for x in batch], dtype=torch.long),
        # kept for decoding predictions back into entities / SLU-F1 scoring:
        "sentences": [x["sentence"] for x in batch],
        "words": [x["words"] for x in batch],
        "word_ids": [x["word_ids"] for x in batch],
        "gold_entities": [x["gold_entities"] for x in batch],
    }


def get_collate_fn(tokenizer):
    """Bind the tokenizer's pad_token_id into a ready-to-use collate_fn."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    return partial(_collate, pad_token_id=pad_id)


# --------------------------------------------------------------------------
# Turning model predictions back into SLURP-style entities (for slurp_f1.py)
# --------------------------------------------------------------------------

def decode_slots_to_entities(pred_tag_ids, word_ids, words, vocab):
    """
    pred_tag_ids: list[int] — argmax tag id per subword token (same length as word_ids)
    word_ids:     list[Optional[int]] — from tokenizer output .word_ids()
    words:        list[str] — original word tokens for this example
    vocab:        SlurpLabelVocab

    Returns: list[{"type": ..., "filler": ...}] in the exact format slurp_f1.py expects.
    """
    word_level_tag = {}
    for tag_id, wid in zip(pred_tag_ids, word_ids):
        if wid is None or wid in word_level_tag:
            continue
        word_level_tag[wid] = vocab.id2tag[int(tag_id)]
    tags = [word_level_tag.get(i, "O") for i in range(len(words))]

    entities = []
    cur_type, cur_words = None, []
    for w, tag in zip(words, tags):
        if tag.startswith("B-"):
            if cur_type is not None:
                entities.append({"type": cur_type, "filler": " ".join(cur_words)})
            cur_type, cur_words = tag[2:], [w]
        elif tag.startswith("I-") and tag[2:] == cur_type:
            cur_words.append(w)
        else:
            if cur_type is not None:
                entities.append({"type": cur_type, "filler": " ".join(cur_words)})
            cur_type, cur_words = None, []
    if cur_type is not None:
        entities.append({"type": cur_type, "filler": " ".join(cur_words)})
    return entities
