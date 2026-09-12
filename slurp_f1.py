#!/usr/bin/env python3
"""
slurp_f1.py — SLU-F1 (SLURP) metric implementation.

Reproduces the "dist-F1" entity-scoring algorithm from:
    Bastianelli, Vanzo, Swietojanski, Rieser.
    "SLURP: A Spoken Language Understanding Resource Package" (EMNLP 2020).
    https://github.com/pswietojanski/slurp

What it computes
-----------------
For each utterance, gold entities and predicted entities are lists of
{"type": <slot label>, "filler": <slot value text>}.

Predicted entities are greedily matched to gold entities of the SAME label,
choosing (per prediction) the gold candidate with the smallest text distance.
- Perfect match (dist == 0):      TP += 1
- Same label, wrong value:        TP += 1, FP += dist, FN += dist   (partial credit)
- Predicted label never in gold:  FP += 1
- Leftover unmatched gold:        FN += 1

This is done twice — once using a WORD-level normalized edit distance
(-> Word-F1) and once using a CHARACTER-level normalized edit distance
(-> Char-F1). The two confusion matrices (summed TP/FP/FN over the whole
dataset) are then ADDED together and Precision/Recall/F1 computed on that
sum. That final number is SLU-F1.

Also included: plain intent (scenario_action) accuracy/F1, since SLURP
papers usually report that alongside SLU-F1.

Usage
-----
CLI (gold and prediction files are JSONL, one JSON object per line):

    python slurp_f1.py -g gold.jsonl -p pred.jsonl

Expected JSON fields per line (matching SLURP's own format):
    {
      "file": "audio-1234.flac",       # optional, used to align gold/pred
      "sentence": "wake me up at seven am",
      "scenario": "alarm",             # optional, for intent metrics
      "action": "set",                 # optional, for intent metrics
      "entities": [{"type": "time", "filler": "seven am"}]
    }

If "file" is present in both gold and prediction lines, records are aligned
by that key. Otherwise they are aligned by line order (so gold.jsonl and
pred.jsonl must have the same number of lines, in the same order).

You can also import and call the functions directly from Python:

    from slurp_f1 import compute_slu_f1, compute_intent_metrics
    result = compute_slu_f1(gold_records, pred_records)
    print(result["overall"]["slu_f1"])
"""

import argparse
import json
import sys
from collections import defaultdict


# --------------------------------------------------------------------------
# Edit distance primitives
# --------------------------------------------------------------------------

def _levenshtein(a, b):
    """Classic Levenshtein edit distance between two sequences (list-like)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost # substitution
            )
        prev = curr
    return prev[m]


def word_distance(ref, hyp):
    """Normalized word-level edit distance (WER-style) between two strings."""
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def char_distance(ref, hyp):
    """Normalized character-level edit distance (CER-style) between strings."""
    ref = ref.strip()
    hyp = hyp.strip()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _levenshtein(list(ref), list(hyp)) / len(ref)


# --------------------------------------------------------------------------
# Core dist-F1 algorithm (Algorithm 1 in the SLURP paper)
# --------------------------------------------------------------------------

def _sentence_dist_confusion(gold_entities, pred_entities, dist_fn):
    """
    Run the dist-F1 matching algorithm for a single utterance.

    gold_entities / pred_entities: list of dicts with "type" and "filler".
    dist_fn: function(ref_str, hyp_str) -> float distance in [0, ~1+]

    Returns (tp, fp, fn) as floats (tp/fp/fn accumulate fractional credit).
    """
    # Work on mutable copies so we can "remove" matched items.
    gold = list(gold_entities)
    pred = list(pred_entities)

    gold_labels = {e["type"] for e in gold}

    tp = fp = fn = 0.0

    for pe in list(pred):
        if pe["type"] in gold_labels:
            # candidates: remaining gold entities with the same label
            candidates = [ge for ge in gold if ge["type"] == pe["type"]]
            if candidates:
                # pick the closest-matching gold entity by text distance
                best = min(candidates, key=lambda ge: dist_fn(ge["filler"], pe["filler"]))
                d = dist_fn(best["filler"], pe["filler"])
                tp += 1.0
                fp += d
                fn += d
                gold.remove(best)
                gold_labels = {e["type"] for e in gold}
            else:
                # label was in gold overall but no remaining instance of it
                fp += 1.0
        else:
            fp += 1.0

    # anything left in gold was never predicted
    fn += len(gold)

    return tp, fp, fn


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------
# Dataset-level aggregation
# --------------------------------------------------------------------------

def compute_slu_f1(gold_records, pred_records, breakdown_by_label=False):
    """
    gold_records / pred_records: lists of dicts, aligned 1:1 (same order,
    or matched beforehand). Each dict needs an "entities" key: a list of
    {"type": str, "filler": str}.

    Returns a dict:
        {
          "word_f1":  {"precision":..., "recall":..., "f1":..., "tp":..., "fp":..., "fn":...},
          "char_f1":  {...},
          "overall":  {"slu_precision":..., "slu_recall":..., "slu_f1":...},
          "per_label": {...}   # only if breakdown_by_label=True
        }
    """
    assert len(gold_records) == len(pred_records), (
        f"Mismatched number of records: {len(gold_records)} gold vs "
        f"{len(pred_records)} predictions."
    )

    word_tp = word_fp = word_fn = 0.0
    char_tp = char_fp = char_fn = 0.0

    per_label_word = defaultdict(lambda: [0.0, 0.0, 0.0])  # label -> [tp, fp, fn]
    per_label_char = defaultdict(lambda: [0.0, 0.0, 0.0])

    for g, p in zip(gold_records, pred_records):
        g_ents = g.get("entities", [])
        p_ents = p.get("entities", [])

        wtp, wfp, wfn = _sentence_dist_confusion(g_ents, p_ents, word_distance)
        ctp, cfp, cfn = _sentence_dist_confusion(g_ents, p_ents, char_distance)

        word_tp += wtp; word_fp += wfp; word_fn += wfn
        char_tp += ctp; char_fp += cfp; char_fn += cfn

        if breakdown_by_label:
            # recompute per label so we can report a breakdown too
            labels = {e["type"] for e in g_ents} | {e["type"] for e in p_ents}
            for lab in labels:
                g_sub = [e for e in g_ents if e["type"] == lab]
                p_sub = [e for e in p_ents if e["type"] == lab]
                wtp2, wfp2, wfn2 = _sentence_dist_confusion(g_sub, p_sub, word_distance)
                ctp2, cfp2, cfn2 = _sentence_dist_confusion(g_sub, p_sub, char_distance)
                a = per_label_word[lab]; a[0] += wtp2; a[1] += wfp2; a[2] += wfn2
                b = per_label_char[lab]; b[0] += ctp2; b[1] += cfp2; b[2] += cfn2

    word_scores = precision_recall_f1(word_tp, word_fp, word_fn)
    char_scores = precision_recall_f1(char_tp, char_fp, char_fn)

    # SLU-F1: sum the two confusion matrices, then compute P/R/F1
    combined_tp = word_tp + char_tp
    combined_fp = word_fp + char_fp
    combined_fn = word_fn + char_fn
    overall = precision_recall_f1(combined_tp, combined_fp, combined_fn)

    result = {
        "word_f1": {**word_scores, "tp": word_tp, "fp": word_fp, "fn": word_fn},
        "char_f1": {**char_scores, "tp": char_tp, "fp": char_fp, "fn": char_fn},
        "overall": {
            "slu_precision": overall["precision"],
            "slu_recall": overall["recall"],
            "slu_f1": overall["f1"],
        },
    }

    if breakdown_by_label:
        per_label = {}
        for lab in set(per_label_word) | set(per_label_char):
            wtp, wfp, wfn = per_label_word[lab]
            ctp, cfp, cfn = per_label_char[lab]
            lab_scores = precision_recall_f1(wtp + ctp, wfp + cfp, wfn + cfn)
            per_label[lab] = lab_scores
        result["per_label"] = per_label

    return result


def compute_intent_metrics(gold_records, pred_records):
    """
    Simple intent (scenario_action) accuracy + micro F1.
    Expects "scenario" and "action" fields (or a combined "intent" field).
    """
    def intent_of(rec):
        if "intent" in rec:
            return rec["intent"]
        return f"{rec.get('scenario','')}_{rec.get('action','')}"

    total = len(gold_records)
    correct = 0
    tp = fp = fn = 0

    for g, p in zip(gold_records, pred_records):
        gi, pi = intent_of(g), intent_of(p)
        if gi == pi:
            correct += 1
            tp += 1
        else:
            fp += 1
            fn += 1

    accuracy = correct / total if total else 0.0
    scores = precision_recall_f1(tp, fp, fn)
    return {"accuracy": accuracy, **scores}


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def align_records(gold, pred):
    """Align by 'file' key if both sides have it; else assume same order."""
    if all("file" in r for r in gold) and all("file" in r for r in pred):
        pred_by_file = {r["file"]: r for r in pred}
        aligned_gold, aligned_pred = [], []
        missing = 0
        for g in gold:
            p = pred_by_file.get(g["file"])
            if p is None:
                missing += 1
                continue
            aligned_gold.append(g)
            aligned_pred.append(p)
        if missing:
            print(f"Warning: {missing} gold records had no matching prediction "
                  f"(matched by 'file') and were skipped.", file=sys.stderr)
        return aligned_gold, aligned_pred
    return gold, pred


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute SLURP SLU-F1 metrics.")
    parser.add_argument("-g", "--gold", required=True, help="Path to gold JSONL file.")
    parser.add_argument("-p", "--pred", required=True, help="Path to predictions JSONL file.")
    parser.add_argument("--per-label", action="store_true",
                         help="Also print a per-entity-label F1 breakdown.")
    parser.add_argument("--no-intent", action="store_true",
                         help="Skip scenario/action intent metrics.")
    args = parser.parse_args()

    gold = load_jsonl(args.gold)
    pred = load_jsonl(args.pred)
    gold, pred = align_records(gold, pred)

    result = compute_slu_f1(gold, pred, breakdown_by_label=args.per_label)

    print("=== SLU-F1 (entity metric) ===")
    print(f"Word-F1 : P={result['word_f1']['precision']:.4f}  "
          f"R={result['word_f1']['recall']:.4f}  F1={result['word_f1']['f1']:.4f}")
    print(f"Char-F1 : P={result['char_f1']['precision']:.4f}  "
          f"R={result['char_f1']['recall']:.4f}  F1={result['char_f1']['f1']:.4f}")
    print(f"SLU-F1  : P={result['overall']['slu_precision']:.4f}  "
          f"R={result['overall']['slu_recall']:.4f}  F1={result['overall']['slu_f1']:.4f}")

    if args.per_label:
        print("\n--- Per-label F1 ---")
        for label, scores in sorted(result["per_label"].items()):
            print(f"{label:20s} P={scores['precision']:.4f}  "
                  f"R={scores['recall']:.4f}  F1={scores['f1']:.4f}")

    if not args.no_intent:
        try:
            intent = compute_intent_metrics(gold, pred)
            print("\n=== Intent (scenario_action) ===")
            print(f"Accuracy={intent['accuracy']:.4f}  "
                  f"P={intent['precision']:.4f}  R={intent['recall']:.4f}  F1={intent['f1']:.4f}")
        except Exception:
            pass  # scenario/action/intent fields not present; skip silently


if __name__ == "__main__":
    main()
