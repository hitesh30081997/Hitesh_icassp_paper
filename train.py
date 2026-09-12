"""
train.py — end-to-end training script for SLURP joint intent + slot model.

Usage:
    python train.py \
        --data_dir data \
        --encoder_name bert-base-uncased \
        --output_dir checkpoints \
        --epochs 5 --batch_size 32 --lr 3e-5

Files expected in --data_dir (the official SLURP annotation files):
    train.jsonl, train_synthetic.jsonl (optional but recommended), devel.jsonl, test.jsonl

What it does each epoch:
    1. Trains on train.jsonl (+ train_synthetic.jsonl if present) with the
       joint intent + slot-tagging loss.
    2. Evaluates on devel.jsonl:
         - intent accuracy / F1
         - SLU-F1 (via slurp_f1.compute_slu_f1), by decoding predicted slot
           tags back into {"type","filler"} entities with decode_slots_to_entities
    3. Saves the best checkpoint (by devel SLU-F1) plus the label vocabulary,
       so you can reload everything later for inference/leaderboard scoring
       on test.jsonl with slurp_f1.py directly.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from slurp_dataset import SlurpLabelVocab, SlurpDataset, get_collate_fn, decode_slots_to_entities
from slurp_model import SlurpJointModel, SlurpJointLoss
from slurp_f1 import compute_slu_f1, compute_intent_metrics


def build_dataloaders(args, tokenizer, vocab):
    train_paths = [os.path.join(args.data_dir, "train.jsonl")]
    synth_path = os.path.join(args.data_dir, "train_synthetic.jsonl")
    if os.path.exists(synth_path) and not args.no_synthetic:
        train_paths.append(synth_path)

    train_ds = torch.utils.data.ConcatDataset([
        SlurpDataset(p, vocab, tokenizer, max_length=args.max_length) for p in train_paths
    ])
    devel_ds = SlurpDataset(os.path.join(args.data_dir, "devel.jsonl"), vocab, tokenizer, args.max_length)
    test_ds = SlurpDataset(os.path.join(args.data_dir, "test.jsonl"), vocab, tokenizer, args.max_length)

    collate_fn = get_collate_fn(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    devel_loader = DataLoader(devel_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, devel_loader, test_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        slot_labels = batch["slot_labels"].to(device)
        intent_labels = batch["intent_label"].to(device)

        optimizer.zero_grad()
        intent_logits, slot_logits = model(input_ids, attention_mask)
        loss, intent_loss, slot_loss = loss_fn(intent_logits, slot_logits, intent_labels, slot_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, vocab, device):
    model.eval()
    gold_records, pred_records = [], []
    intent_gold, intent_pred = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        intent_logits, slot_logits = model(input_ids, attention_mask)

        intent_pred_ids = intent_logits.argmax(dim=-1).cpu().tolist()
        slot_pred_ids = slot_logits.argmax(dim=-1).cpu().tolist()

        for i in range(len(batch["sentences"])):
            decoded = decode_slots_to_entities(
                slot_pred_ids[i], batch["word_ids"][i], batch["words"][i], vocab
            )
            gold_records.append({"entities": batch["gold_entities"][i]})
            pred_records.append({"entities": decoded})

            gold_intent_id = batch["intent_label"][i].item()
            intent_gold.append({"intent": vocab.id2intent[gold_intent_id]})
            intent_pred.append({"intent": vocab.id2intent[intent_pred_ids[i]]})

    slu = compute_slu_f1(gold_records, pred_records)
    intent_metrics = compute_intent_metrics(intent_gold, intent_pred)
    return slu, intent_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--encoder_name", default="bert-base-uncased",
                         help="Any HF encoder checkpoint, e.g. bert-base-uncased, "
                              "distilbert-base-uncased, roberta-base.")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--slot_loss_weight", type=float, default=1.0)
    parser.add_argument("--no_synthetic", action="store_true",
                         help="Don't include train_synthetic.jsonl in training.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. label vocabulary — build from BOTH real and synthetic train data so
    #    the vocab covers every label that could appear
    vocab_paths = [os.path.join(args.data_dir, "train.jsonl")]
    synth_path = os.path.join(args.data_dir, "train_synthetic.jsonl")
    if os.path.exists(synth_path):
        vocab_paths.append(synth_path)
    vocab = SlurpLabelVocab.build(vocab_paths)
    vocab.save(os.path.join(args.output_dir, "label_vocab.json"))
    print(f"Labels -> scenarios={vocab.num_scenarios} actions={vocab.num_actions} "
          f"intents={vocab.num_intents} tags={vocab.num_tags}")

    # 2. tokenizer + data
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    train_loader, devel_loader, test_loader = build_dataloaders(args, tokenizer, vocab)

    # 3. model + loss + optimizer
    model = SlurpJointModel(args.encoder_name, vocab.num_intents, vocab.num_tags).to(device)
    loss_fn = SlurpJointLoss(slot_loss_weight=args.slot_loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 4. train / eval loop, keep best checkpoint by devel SLU-F1
    best_slu_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        slu, intent_metrics = evaluate(model, devel_loader, vocab, device)

        print(f"[epoch {epoch}] train_loss={train_loss:.4f} | "
              f"devel SLU-F1={slu['overall']['slu_f1']:.4f} "
              f"(P={slu['overall']['slu_precision']:.4f} R={slu['overall']['slu_recall']:.4f}) | "
              f"devel intent_acc={intent_metrics['accuracy']:.4f}")

        if slu["overall"]["slu_f1"] > best_slu_f1:
            best_slu_f1 = slu["overall"]["slu_f1"]
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> new best devel SLU-F1={best_slu_f1:.4f}, checkpoint saved")

    # 5. final test-set evaluation with the best checkpoint
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device))
    test_slu, test_intent = evaluate(model, test_loader, vocab, device)
    print("\n=== Final TEST results (best devel checkpoint) ===")
    print(f"SLU-F1: P={test_slu['overall']['slu_precision']:.4f} "
          f"R={test_slu['overall']['slu_recall']:.4f} F1={test_slu['overall']['slu_f1']:.4f}")
    print(f"Intent accuracy: {test_intent['accuracy']:.4f}")


if __name__ == "__main__":
    main()
