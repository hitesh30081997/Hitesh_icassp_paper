"""
slurp_audio_dataset.py — PyTorch Dataset for SLURP with SPEECH (audio) input.

Source dataset:
    Bastianelli, Vanzo, Swietojanski, Rieser.
    "SLURP: A Spoken Language Understanding Resource Package" (EMNLP 2020).
    https://github.com/pswietojanski/slurp

IMPORTANT — audio is a separate download from the text annotations:
    The jsonl files (train.jsonl etc.) only contain text + a list of audio
    filenames per sentence (the "recordings" field). The actual .flac audio
    (~6GB) is hosted separately on Zenodo and is NOT included in this repo's
    text annotations. To get it:

        git clone https://github.com/pswietojanski/slurp.git
        cd slurp
        bash scripts/download_audio.sh
        # downloads:
        #   https://zenodo.org/record/4274930/files/slurp_real.tar.gz
        #   https://zenodo.org/record/4274930/files/slurp_synth.tar.gz
        # and extracts them to audio/slurp_real/ and audio/slurp_synth/

    After that you'll have a directory that looks like:
        audio/
          slurp_real/     <- audio for train.jsonl, devel.jsonl, test.jsonl
             audio-1501754435.flac
             audio-1501407267-headset.flac
             ...
          slurp_synth/    <- audio for train_synthetic.jsonl
             audio-1590160754048Ccloa-synth.flac
             ...

    Point `audio_dir` at that `audio/` folder (the one containing the two
    subfolders) and this Dataset resolves filenames automatically.

IMPORTANT — about slot/entity labels with audio input:
    SLURP's entity annotations are token-character spans over the TEXT
    transcript (see slurp_dataset.py), with no forced-alignment timestamps
    into the audio. So there is no ground truth to train a model to predict
    "this slot value spans these audio frames" directly. In practice (and in
    the SLURP paper itself) speech SLU on this dataset is done one of two ways:

      1. End-to-end INTENT classification from audio (scenario/action/intent) —
         fully supported, no alignment needed. That's what SlurpAudioDataset +
         SlurpAudioIntentModel (in slurp_audio_model.py) implement below.

      2. Cascade for SLOTS: run ASR on the audio to get a transcript, then
         run the text slot-filling model (slurp_dataset.py + slurp_model.py)
         on that transcript, and score with slurp_f1.py. This is exactly why
         slurp_f1's dist-F1 metric uses fuzzy word/char distance matching
         instead of exact match — it's designed to tolerate ASR transcription
         errors. See cascade_asr_nlu.py for this pipeline end-to-end.

This module gives you (1): audio -> waveform tensor, aligned with the same
scenario/action/intent label vocabulary used by the text pipeline, so you
can train/evaluate a speech intent classifier and, if you want, feed its
ASR-derived transcript into the existing text pipeline for slots.
"""

import json
import os
import random

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _resolve_audio_path(audio_dir, filename):
    """slurp_real/ holds real recordings (train/devel/test); slurp_synth/
    holds synthetic TTS recordings (train_synthetic, filenames end in
    '-synth.flac')."""
    subdir = "slurp_synth" if filename.endswith("-synth.flac") else "slurp_real"
    return os.path.join(audio_dir, subdir, filename)


def _mic_of(filename):
    if "-headset" in filename:
        return "headset"
    if "-synth" in filename:
        return "synthetic"
    return "close-talk"


class SlurpAudioDataset(Dataset):
    """
    Each item = one (audio recording, scenario/action/intent label) pair.

    A SLURP sentence has MULTIPLE recordings (different mics/speakers). By
    default this expands every jsonl record into one dataset item per
    matching recording, which is the usual way to use SLURP for speech
    training (more audio examples per unique sentence).

    Args:
        jsonl_path: path to train.jsonl / train_synthetic.jsonl / devel.jsonl / test.jsonl
        audio_dir: path to the extracted `audio/` folder (contains slurp_real/, slurp_synth/)
        label_vocab: a SlurpLabelVocab (from slurp_dataset.py) — reuse the SAME
                     vocab object you built for the text pipeline, so audio and
                     text models share label ids.
        sample_rate: target sample rate, everything is resampled to this (default 16000,
                     what wav2vec2/whisper expect)
        mic: "all" | "headset" | "close-talk" — filter which physical microphone's
             recordings to use. "all" uses both (more data, more robust model).
        status_filter: only keep recordings whose annotation status is in this set
                       (default: keep "correct" and "synthetic", drop e.g. "corrupt")
        max_duration_sec: drop recordings longer than this (None = no limit)
    """

    def __init__(self, jsonl_path, audio_dir, label_vocab, sample_rate=16000,
                 mic="all", status_filter=("correct", "synthetic"),
                 max_duration_sec=None):
        self.audio_dir = audio_dir
        self.vocab = label_vocab
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec

        records = load_jsonl(jsonl_path)
        self.items = []  # flattened (record, recording) pairs
        for r in records:
            for rec in r.get("recordings", []):
                if rec.get("status") not in status_filter:
                    continue
                m = _mic_of(rec["file"])
                if mic != "all" and m != mic:
                    continue
                self.items.append((r, rec))

        if not self.items:
            raise ValueError(
                "No recordings matched your filters. Common causes: audio_dir "
                "doesn't point at the extracted SLURP audio/ folder, or the "
                "mic/status filters excluded everything. See the module "
                "docstring for how to download the audio."
            )

    def __len__(self):
        return len(self.items)

    def _load_waveform(self, filename):
        path = _resolve_audio_path(self.audio_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Audio file not found: {path}\n"
                f"Did you run scripts/download_audio.sh from the official SLURP repo "
                f"and point audio_dir at the resulting audio/ folder? "
                f"See the top of slurp_audio_dataset.py for exact instructions."
            )
        wave, sr = sf.read(path, dtype="float32")  # [num_samples] or [num_samples, channels]
        wave_t = torch.from_numpy(wave)
        if wave_t.ndim == 2:  # stereo -> mono
            wave_t = wave_t.mean(dim=1)
        if sr != self.sample_rate:
            wave_t = torchaudio.functional.resample(wave_t.unsqueeze(0), sr, self.sample_rate).squeeze(0)
        return wave_t

    def __getitem__(self, idx):
        r, rec = self.items[idx]
        waveform = self._load_waveform(rec["file"])

        if self.max_duration_sec is not None:
            max_samples = int(self.max_duration_sec * self.sample_rate)
            waveform = waveform[:max_samples]

        gold_entities = [
            {"type": e["type"], "filler": " ".join(t["surface"] for t in r["tokens"] if t["id"] in e["span"])}
            for e in r.get("entities", [])
        ]

        return {
            "waveform": waveform,                      # 1D float tensor, raw audio @ self.sample_rate
            "scenario_label": self.vocab.scenario2id[r["scenario"]],
            "action_label": self.vocab.action2id[r["action"]],
            "intent_label": self.vocab.intent2id[r["intent"]],
            "sentence": r["sentence"],                  # reference transcript, for ASR/WER comparison
            "gold_entities": gold_entities,              # reference slots, for cascade SLU-F1 scoring
            "file": rec["file"],
            "mic": _mic_of(rec["file"]),
        }


class SlurpAudioCollator:
    """
    Pads a batch of variable-length waveforms and (optionally) applies the
    zero-mean/unit-variance normalization wav2vec2-style models expect.

    Uses transformers' Wav2Vec2FeatureExtractor directly (constructed with
    default settings, NOT from_pretrained — no download needed) purely as a
    padding + normalization utility.
    """

    def __init__(self, sample_rate=16000, do_normalize=True):
        from transformers import Wav2Vec2FeatureExtractor
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=sample_rate,
            padding_value=0.0,
            do_normalize=do_normalize,
            return_attention_mask=True,
        )

    def __call__(self, batch):
        waveforms = [x["waveform"].numpy() for x in batch]
        features = self.feature_extractor(
            waveforms, sampling_rate=self.feature_extractor.sampling_rate,
            padding=True, return_tensors="pt",
        )
        return {
            "input_values": features["input_values"],         # [B, T]
            "attention_mask": features["attention_mask"],     # [B, T]
            "scenario_label": torch.tensor([x["scenario_label"] for x in batch], dtype=torch.long),
            "action_label": torch.tensor([x["action_label"] for x in batch], dtype=torch.long),
            "intent_label": torch.tensor([x["intent_label"] for x in batch], dtype=torch.long),
            "sentences": [x["sentence"] for x in batch],
            "gold_entities": [x["gold_entities"] for x in batch],
            "files": [x["file"] for x in batch],
        }
