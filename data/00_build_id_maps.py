"""
Step 0: build the id -> index maps for the gold-chain evidence subset.

Scans MMQA_train.jsonl / MMQA_dev.jsonl for the text and table doc_ids that
appear as supporting context, then assigns each a contiguous index:
  - map_text_ids.json   : text doc_id         -> index
  - map_table_ids.json  : "{table_id}#{row}"  -> index

These maps define the row order of embeddings_text.npy / embeddings_table.npy,
so this step must run before embeddings/embed_text_table.py. Fully
deterministic (sorted text ids + table file order; no model, no randomness).
"""

import os
import json

from tqdm import tqdm

# Data directory. Defaults to the repo root; override with MMQA_DATA_DIR.
BASE_DIR = os.environ.get("MMQA_DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_FILE = os.path.join(BASE_DIR, "MMQA_train.jsonl")
DEV_FILE = os.path.join(BASE_DIR, "MMQA_dev.jsonl")
TABLES_FILE = os.path.join(BASE_DIR, "MMQA_tables.jsonl")

MAP_TEXT_IDS = os.path.join(BASE_DIR, "map_text_ids.json")
MAP_TABLE_IDS = os.path.join(BASE_DIR, "map_table_ids.json")


def collect_needed_ids():
    """Collect the text/table doc_ids referenced in train/dev supporting context."""
    needed_tables = set()
    needed_texts = set()

    for path in [TRAIN_FILE, DEV_FILE]:
        if not os.path.exists(path):
            continue
        print(f"Scanning {path}...")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if "table_id" in item.get("metadata", {}):
                    needed_tables.add(item["metadata"]["table_id"])
                for ctx in item.get("supporting_context", []):
                    if ctx.get("doc_part") == "table":
                        needed_tables.add(ctx["doc_id"])
                    elif ctx.get("doc_part") == "text":
                        needed_texts.add(ctx["doc_id"])

    print(f"Found {len(needed_tables)} unique tables and {len(needed_texts)} unique texts.")
    return needed_tables, needed_texts


def create_mappings(needed_tables, needed_texts):
    """Build deterministic id -> index maps for texts and table rows."""
    # Text: sorted doc_ids -> contiguous index.
    text_map = {tid: i for i, tid in enumerate(sorted(list(needed_texts)))}

    # Table: "{table_id}#{row_idx}" -> contiguous index, in table-file order.
    table_row_map = {}
    idx = 0
    print(f"Loading {TABLES_FILE} to enumerate rows...")
    with open(TABLES_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Mapping table rows"):
            item = json.loads(line)
            tid = item["id"]
            if tid in needed_tables:
                rows = item["table"].get("table_rows", [])
                for r_idx in range(len(rows)):
                    table_row_map[f"{tid}#{r_idx}"] = idx
                    idx += 1

    return text_map, table_row_map


def main():
    needed_tables, needed_texts = collect_needed_ids()
    text_map, table_row_map = create_mappings(needed_tables, needed_texts)

    with open(MAP_TEXT_IDS, "w", encoding="utf-8") as f:
        json.dump(text_map, f, indent=2)
    with open(MAP_TABLE_IDS, "w", encoding="utf-8") as f:
        json.dump(table_row_map, f, indent=2)

    print(f"Saved {len(text_map)} texts to {MAP_TEXT_IDS}")
    print(f"Saved {len(table_row_map)} table rows to {MAP_TABLE_IDS}")


if __name__ == "__main__":
    main()
