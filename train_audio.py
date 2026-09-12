"""
train_audio.py — train an end-to-end speech intent classifier on SLURP audio.

Usage:
    python train_audio.py \
        --data_dir data \
        --audio_dir /path/to/slurp/audio \
        --encoder_name facebook/wav2vec2-base-960h \
        --output_dir checkpoints_audio \
        --epochs 10 --batch_size 8 --lr 1e-4

Prerequisites:
    1. Text annotations (already in ./data): train.jsonl, train_synthetic.jsonl,
       devel.jsonl, test.jsonl.
    2. Audio (separate ~6GB download, see slurp_audio_dataset.py docstring):
           git clone https://github.com/pswietojanski/slurp.git
           cd slurp && bash scripts/download_audio.sh
       --audio_dir should point at the resulting `audio/` folder.

This trains scenario/action/intent classification directly from raw audio.
It does NOT train slot filling (see slurp_audio_dataset.py for why SLURP's
audio can't be used for E2E slot supervision) — for slots, run
cascade_asr_nlu.py after this.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from slurp_dataset import SlurpLabelVocab
from slurp_audio_dataset import SlurpAudioDataset, SlurpAudioCollator
from slurp_audio_model import SlurpAudioIntentModel, SlurpAudioLoss


def build_dataloaders(args, vocab):
    collate_fn = SlurpAudioCollator(sample_rate=args.sample_rate)

    train_paths = [os.path.join(args.data_dir, "train.jsonl")]
    synth_path = os.path.join(args.data_dir, "train_synthetic.jsonl")
    if os.path.exists(synth_path) and not args.no_synthetic:
        train_paths.append(synth_path)

    train_ds = torch.utils.data.ConcatDataset([
        SlurpAudioDataset(p, args.audio_dir, vocab, sample_rate=args.sample_rate,
                           mic=args.mic, max_duration_sec=args.max_duration_sec)
        for p in train_paths
    ])
    devel_ds = SlurpAudioDataset(os.path.join(args.data_dir, "devel.jsonl"), args.audio_dir, vocab,
                                  sample_rate=args.sample_rate, mic=args.mic,
                                  max_duration_sec=args.max_duration_sec)
    test_ds = SlurpAudioDataset(os.path.join(args.data_dir, "test.jsonl"), args.audio_dir, vocab,
                                 sample_rate=args.sample_rate, mic=args.mic,
                                 max_duration_sec=args.max_duration_sec)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    devel_loader = DataLoader(devel_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, devel_loader, test_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        s_labels = batch["scenario_label"].to(device)
        a_labels = batch["action_label"].to(device)
        i_labels = batch["intent_label"].to(device)

        optimizer.zero_grad()
        s_logits, a_logits, i_logits = model(input_values, attention_mask)
        loss = loss_fn(s_logits, a_logits, i_logits, s_labels, a_labels, i_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct_intent, total = 0, 0
    for batch in loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        _, _, i_logits = model(input_values, attention_mask)
        preds = i_logits.argmax(dim=-1).cpu()
        correct_intent += (preds == batch["intent_label"]).sum().item()
        total += len(preds)
    return correct_intent / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--audio_dir", required=True,
                         help="Path to the extracted SLURP audio/ folder (contains slurp_real/, slurp_synth/).")
    parser.add_argument("--encoder_name", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--output_dir", default="checkpoints_audio")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--mic", default="all", choices=["all", "headset", "close-talk"])
    parser.add_argument("--max_duration_sec", type=float, default=10.0)
    parser.add_argument("--no_synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_paths = [os.path.join(args.data_dir, "train.jsonl")]
    synth_path = os.path.join(args.data_dir, "train_synthetic.jsonl")
    if os.path.exists(synth_path):
        vocab_paths.append(synth_path)
    vocab = SlurpLabelVocab.build(vocab_paths)
    vocab.save(os.path.join(args.output_dir, "label_vocab.json"))
    print(f"Labels -> scenarios={vocab.num_scenarios} actions={vocab.num_actions} intents={vocab.num_intents}")

    train_loader, devel_loader, test_loader = build_dataloaders(args, vocab)

    model = SlurpAudioIntentModel(
        args.encoder_name, vocab.num_scenarios, vocab.num_actions, vocab.num_intents
    ).to(device)
    loss_fn = SlurpAudioLoss(scenario_weight=0.5, action_weight=0.5, intent_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        devel_acc = evaluate(model, devel_loader, device)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} devel_intent_acc={devel_acc:.4f}")
        if devel_acc > best_acc:
            best_acc = devel_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> new best devel_intent_acc={best_acc:.4f}, checkpoint saved")

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f"\n=== Final TEST intent accuracy (best devel checkpoint): {test_acc:.4f} ===")


if __name__ == "__main__":
    main()
