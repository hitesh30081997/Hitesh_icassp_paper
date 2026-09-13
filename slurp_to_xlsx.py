"""
Convert a SLURP dataset jsonl file into a structured Excel sheet.

Usage:
    python slurp_to_xlsx.py <input.jsonl> <output.xlsx>

One row is written per (slurp_id, recording file) pair, so slurp_ids with
multiple audio recordings produce multiple rows -- all sharing the same
sentence/entities/scenario/action/intent, differing only in file name.

Columns:
    slurp id | sentence | entities (type: words) | entities (type: spans) |
    scenario | action | intent | file name
"""
import json
import sys
import pandas as pd


def format_entities_words(entities, tokens):
    """Build {type: word word, type: word} using token surfaces resolved from span indices."""
    if not entities:
        return ""
    surface_by_id = {t["id"]: t["surface"] for t in tokens}
    parts = []
    for ent in entities:
        words = " ".join(surface_by_id[i] for i in ent["span"] if i in surface_by_id)
        parts.append(f"{ent['type']}: {words}")
    return "{" + ", ".join(parts) + "}"


def format_entities_spans(entities):
    """Build {type: [idx, idx], type: [idx]} using raw token-index spans."""
    if not entities:
        return ""
    parts = []
    for ent in entities:
        span_str = "[" + ",".join(str(i) for i in ent["span"]) + "]"
        parts.append(f"{ent['type']}: {span_str}")
    return "{" + ", ".join(parts) + "}"


def convert(input_path, output_path):
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            slurp_id = rec.get("slurp_id")
            sentence = rec.get("sentence", "")
            tokens = rec.get("tokens", [])
            entities = rec.get("entities", [])
            scenario = rec.get("scenario", "")
            action = rec.get("action", "")
            intent = rec.get("intent", "")
            # SLURP paper defines intent as scenario_action (60 unique classes).
            # The raw "intent" field in the jsonl is occasionally missing the
            # scenario prefix (e.g. "sendemail" instead of "email_sendemail"),
            # which inflates the unique-intent count if used as-is. This
            # reconstructs the canonical label from scenario + action.
            intent_corrected = f"{scenario}_{action}"
            recordings = rec.get("recordings", [])

            ents_words = format_entities_words(entities, tokens)
            ents_spans = format_entities_spans(entities)

            if recordings:
                for r in recordings:
                    rows.append({
                        "slurp id": slurp_id,
                        "sentence": sentence,
                        "entities (type: words)": ents_words,
                        "entities (type: spans)": ents_spans,
                        "scenario": scenario,
                        "action": action,
                        "intent (raw)": intent,
                        "intent (corrected)": intent_corrected,
                        "file name": r.get("file", ""),
                    })
            else:
                # no recordings listed -- still emit one row so the utterance isn't lost
                rows.append({
                    "slurp id": slurp_id,
                    "sentence": sentence,
                    "entities (type: words)": ents_words,
                    "entities (type: spans)": ents_spans,
                    "scenario": scenario,
                    "action": action,
                    "intent (raw)": intent,
                    "intent (corrected)": intent_corrected,
                    "file name": "",
                })

    df = pd.DataFrame(rows, columns=[
        "slurp id", "sentence", "entities (type: words)", "entities (type: spans)",
        "scenario", "action", "intent (raw)", "intent (corrected)", "file name",
    ])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="slurp")
        ws = writer.sheets["slurp"]
        # basic formatting: professional font, sensible column widths, frozen header
        from openpyxl.styles import Font
        widths = {"A": 10, "B": 45, "C": 40, "D": 30, "E": 14, "F": 14, "G": 18, "H": 18, "I": 35}
        for col, w in widths.items():
            ws.column_dimensions[col].width = w
        for row in ws.iter_rows():
            for cell in row:
                cell.font = Font(name="Arial", size=10, bold=(cell.row == 1))
                # A handful of SLURP sentences are literally the text "#NAME?"
                # (a data-quality artifact from the original dataset). openpyxl
                # auto-detects that pattern as an Excel *error* value instead of
                # text -- force it back to a plain string so it displays/reads
                # correctly instead of showing as a broken formula.
                if cell.data_type == "e":
                    cell.data_type = "s"
        ws.freeze_panes = "A2"

    print(f"Wrote {len(df)} rows ({df['slurp id'].nunique()} unique slurp ids) to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python slurp_to_xlsx.py <input.jsonl> <output.xlsx>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
