"""
cascade_asr_nlu.py — Slot filling from SPEECH via ASR -> text NLU cascade.

Why a cascade: SLURP's entity/slot labels are spans over the TEXT transcript,
with no timestamps into the audio, so there's no direct supervision for
"predict this slot from these audio frames". The standard (and the SLURP
paper's own) approach is:

    audio --[ASR]--> hypothesis text --[text NLU model]--> intent + slots

and then score against the gold annotation with slurp_f1.py's dist-F1 metric
— which uses fuzzy word/char distance matching specifically so that small
ASR transcription errors (e.g. "seven am" -> "seven a.m.") don't zero out an
otherwise-correct slot prediction.

This script wires that together:
    1. ASR: any HF automatic-speech-recognition pipeline (wav2vec2-CTC,
       whisper, etc.) turns a waveform into text.
    2. NLU: the text model from slurp_dataset.py / slurp_model.py / train.py
       (load a checkpoint you already trained with train.py) predicts intent
       + BIO slot tags on that ASR text.
    3. decode_slots_to_entities turns predicted tags back into entities.
    4. slurp_f1.compute_slu_f1 scores predicted vs. gold entities.

Usage:
    python cascade_asr_nlu.py \
        --data_dir data --audio_dir /path/to/slurp/audio --split devel \
        --asr_model facebook/wav2vec2-base-960h \
        --text_checkpoint checkpoints/best_model.pt \
        --text_encoder_name bert-base-uncased \
        --label_vocab checkpoints/label_vocab.json
"""

import argparse
import os

import torch

from slurp_dataset import SlurpLabelVocab, SlurpDataset, decode_slots_to_entities, load_jsonl
from slurp_audio_dataset import _resolve_audio_path
from slurp_model import SlurpJointModel
from slurp_f1 import compute_slu_f1, compute_intent_metrics


def build_asr_backend(asr_model_name, device):
    """Returns a callable(waveform_1d_tensor, sample_rate) -> str transcript.
    Swap this out for any ASR system you like (Whisper, your own model, a
    cloud ASR API, etc.) as long as it matches that signature."""
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=asr_model_name,
                    device=0 if device.type == "cuda" else -1)

    def _transcribe(waveform, sample_rate):
        result = asr({"array": waveform.numpy(), "sampling_rate": sample_rate})
        return result["text"].strip().lower()

    return _transcribe


@torch.no_grad()
def run_cascade(records, audio_dir, asr_fn, text_model, tokenizer, vocab, device,
                 sample_rate=16000, prefer_mic="close-talk"):
    """
    records: list of raw jsonl records (from load_jsonl on e.g. devel.jsonl)
    Picks ONE recording per sentence (preferring `prefer_mic`, falling back to
    any "correct" recording) — for evaluation you typically don't need every
    mic variant, just a representative one per sentence.
    """
    import soundfile as sf
    import torchaudio

    gold_records, pred_records = [], []
    intent_gold, intent_pred = [], []

    for r in records:
        recordings = [rec for rec in r.get("recordings", []) if rec.get("status") == "correct"]
        if not recordings:
            continue
        chosen = next((rec for rec in recordings if prefer_mic in rec["file"]), recordings[0])

        path = _resolve_audio_path(audio_dir, chosen["file"])
        if not os.path.exists(path):
            continue  # skip if audio wasn't downloaded for this particular file

        wave, sr = sf.read(path, dtype="float32")
        wave_t = torch.from_numpy(wave)
        if wave_t.ndim == 2:
            wave_t = wave_t.mean(dim=1)
        if sr != sample_rate:
            wave_t = torchaudio.functional.resample(wave_t.unsqueeze(0), sr, sample_rate).squeeze(0)

        # 1. ASR
        hyp_text = asr_fn(wave_t, sample_rate)
        hyp_words = hyp_text.split() or [""]

        # 2. text NLU on the ASR hypothesis
        enc = tokenizer(hyp_words, is_split_into_words=True, truncation=True,
                         max_length=64, return_tensors="pt")
        word_ids = enc.word_ids(0)
        intent_logits, slot_logits = text_model(
            enc["input_ids"].to(device), enc["attention_mask"].to(device)
        )
        pred_intent_id = intent_logits.argmax(dim=-1).item()
        pred_tag_ids = slot_logits.argmax(dim=-1)[0].cpu().tolist()

        # 3. decode back to entities
        pred_entities = decode_slots_to_entities(pred_tag_ids, word_ids, hyp_words, vocab)

        gold_entities = [
            {"type": e["type"], "filler": " ".join(t["surface"] for t in r["tokens"] if t["id"] in e["span"])}
            for e in r.get("entities", [])
        ]

        gold_records.append({"entities": gold_entities})
        pred_records.append({"entities": pred_entities})
        intent_gold.append({"intent": r["intent"]})
        intent_pred.append({"intent": vocab.id2intent[pred_intent_id]})

    return gold_records, pred_records, intent_gold, intent_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--split", default="devel", choices=["devel", "test"])
    parser.add_argument("--asr_model", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--text_checkpoint", required=True, help="Path to best_model.pt from train.py")
    parser.add_argument("--text_encoder_name", default="bert-base-uncased")
    parser.add_argument("--label_vocab", required=True, help="Path to label_vocab.json from train.py")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = SlurpLabelVocab.load(args.label_vocab)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name)

    text_model = SlurpJointModel(args.text_encoder_name, vocab.num_intents, vocab.num_tags).to(device)
    text_model.load_state_dict(torch.load(args.text_checkpoint, map_location=device))
    text_model.eval()

    asr_fn = build_asr_backend(args.asr_model, device)

    records = load_jsonl(os.path.join(args.data_dir, f"{args.split}.jsonl"))
    gold, pred, ig, ip = run_cascade(records, args.audio_dir, asr_fn, text_model, tokenizer, vocab, device)

    slu = compute_slu_f1(gold, pred)
    intent_metrics = compute_intent_metrics(ig, ip)

    print(f"=== Cascade ASR->NLU results on {args.split} ({len(gold)} utterances scored) ===")
    print(f"SLU-F1: P={slu['overall']['slu_precision']:.4f} "
          f"R={slu['overall']['slu_recall']:.4f} F1={slu['overall']['slu_f1']:.4f}")
    print(f"Intent accuracy: {intent_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
