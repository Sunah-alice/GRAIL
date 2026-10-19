import os
import json
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, TapasTokenizer, TapasModel
import pandas as pd


# Data directory. Defaults to the repo root; override with MMQA_DATA_DIR.
BASE_DIR = os.environ.get("MMQA_DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEXTS_FILE = os.path.join(BASE_DIR, "MMQA_texts.jsonl")
TABLES_FILE = os.path.join(BASE_DIR, "MMQA_tables.jsonl")
GOLD_CHAINS_FILE = os.path.join(BASE_DIR, "enriched_gold_chains.jsonl")

MAP_TEXT_IDS = os.path.join(BASE_DIR, "map_text_ids.json")
MAP_TABLE_IDS = os.path.join(BASE_DIR, "map_table_ids.json")

OUT_TEXT_NPY = os.path.join(BASE_DIR, "embeddings_text.npy")
OUT_TABLE_NPY = os.path.join(BASE_DIR, "embeddings_table.npy")

# Text encoder: BGE-Large (same as question / text evidences)
TEXT_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Table encoder: TAPAS-base (row-level evidence)
# Change the checkpoint here if needed.
TABLE_MODEL_NAME = "google/tapas-base-finetuned-wtq"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_MAX_LEN = 256
BATCH_SIZE = 32

# TAPAS uses table + queries; max_length follows the model config.
BATCH_SIZE = 32


def load_json_map(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def invert_id_map(id_map: Dict[str, int]) -> List[str]:
    """Invert a doc_id -> idx map into an idx -> doc_id list."""
    max_idx = max(id_map.values())
    id_list = [None] * (max_idx + 1)
    for doc_id, idx in id_map.items():
        id_list[idx] = doc_id
    if any(x is None for x in id_list):
        missing = [i for i, x in enumerate(id_list) if x is None][:10]
        raise ValueError(f"id_list has missing entries at indices: {missing}")
    return id_list  # index = chain id, value = original doc_id


def build_text_corpus_for_gold(text_id_list: List[str]) -> List[str]:
    """
    Read the content of gold-chain text docs and return texts in index order.
    """
    needed = set(text_id_list)
    doc_text: Dict[str, str] = {}

    print(f"[TEXT] Loading MMQA_texts.jsonl and filtering {len(needed)} gold docs...")
    with open(TEXTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            _id = item["id"]
            if _id in needed:
                # Use the 'text' (content) field only.
                doc_text[_id] = item.get("text", "")

    missing = [d for d in needed if d not in doc_text]
    if missing:
        print(f"[TEXT] WARNING: {len(missing)} doc_ids not found in MMQA_texts.jsonl (showing up to 5): {missing[:5]}")

    texts: List[str] = []
    for doc_id in text_id_list:
        txt = doc_text.get(doc_id, "")
        if not txt:
            # Avoid an empty string by falling back to the id.
            txt = f"[EMPTY] {doc_id}"
        texts.append(txt)
    return texts


def collect_table_rows_from_gold() -> Dict[str, List[int]]:
    """
    Collect, per table (doc_id), the row indices used by gold-chain contexts
    across enriched_gold_chains.jsonl.

    Returns:
        table_id -> sorted unique row indices
    """

    def walk(obj, acc: Dict[str, set]):
        if isinstance(obj, dict):
            # table context pattern: {"doc_id": ..., "doc_part": "table", "rows": [...]}
            if obj.get("doc_part") == "table" and "doc_id" in obj and "rows" in obj:
                doc_id = obj["doc_id"]
                rows = obj.get("rows") or []
                if isinstance(rows, list):
                    for r in rows:
                        try:
                            acc[doc_id].add(int(r))
                        except Exception:
                            continue
            # recurse
            for v in obj.values():
                walk(v, acc)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, acc)

    acc: Dict[str, set] = defaultdict(set)
    print("[TABLE] Collecting evidence rows from enriched_gold_chains.jsonl ...")
    with open(GOLD_CHAINS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            walk(item.get("gold_chain", {}), acc)

    result: Dict[str, List[int]] = {}
    for k, v in acc.items():
        result[k] = sorted(v)
    print(f"[TABLE] Collected row indices for {len(result)} tables.")
    return result


def load_tables_for_gold(table_id_list: List[str]) -> Dict[str, Dict]:
    """
    Load the raw table structures for gold-chain tables.
    Returns: doc_id -> {"title": ..., "table": {...}}
    """
    needed = set(table_id_list)
    result: Dict[str, Dict] = {}

    print(f"[TABLE] Loading MMQA_tables.jsonl for {len(needed)} gold tables...")
    with open(TABLES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            doc_id = item["id"]
            if doc_id in needed:
                result[doc_id] = item

    missing = [d for d in needed if d not in result]
    if missing:
        print(f"[TABLE] WARNING: {len(missing)} table_ids not found in MMQA_tables.jsonl (showing up to 5): {missing[:5]}")
    return result


def encode_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_len: int,
) -> np.ndarray:
    """
    Compute CLS embeddings with BGE-Large, L2-normalize, return a numpy matrix.
    """
    model.eval()
    all_vecs: List[np.ndarray] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding texts"):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :]  # (B, H)
            cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
            all_vecs.append(cls.cpu().numpy())

    return np.concatenate(all_vecs, axis=0)


def build_pandas_table(table_obj: Dict) -> pd.DataFrame:
    """
    Build a pandas.DataFrame for TAPAS input from the table's header and rows.
    """
    header = table_obj.get("header", [])
    rows = table_obj.get("table_rows", [])

    col_names: List[str] = []
    used = set()
    for i, col in enumerate(header):
        name = col.get("column_name", "").strip()
        if not name:
            name = f"col_{i}"
        # TAPAS requires unique column names; add a suffix on collision.
        base = name
        k = 1
        while name in used:
            name = f"{base}_{k}"
            k += 1
        used.add(name)
        col_names.append(name)

    data = []
    for row in rows:
        cells = [c.get("text", "") for c in row]
        # Pad/trim to the header length.
        if len(cells) < len(col_names):
            cells = cells + [""] * (len(col_names) - len(cells))
        elif len(cells) > len(col_names):
            cells = cells[: len(col_names)]
        data.append(cells)

    if not col_names:
        # Fallback to a single column for malformed tables.
        col_names = ["col_0"]
        data = [[str(row)] for row in rows]

    df = pd.DataFrame(data, columns=col_names)
    return df


def encode_tables_with_tapas(
    tapas_model: TapasModel,
    tapas_tokenizer: TapasTokenizer,
    table_id_list: List[str],
    table_raw_map: Dict[str, Dict],
    table_rows_map: Dict[str, List[int]],
) -> np.ndarray:
    """
    Build table embeddings with TAPAS-base. For each table, run each gold
    evidence row as a TAPAS query and mean-pool the per-row CLS embeddings
    into one table embedding. Result: (N_table, D_tapas).
    """
    tapas_model.eval()

    all_embs: List[np.ndarray] = []

    print("[TABLE] Encoding tables with TAPAS (header + evidence rows)...")
    for doc_id in tqdm(table_id_list, desc="Encoding tables"):
        item = table_raw_map.get(doc_id)
        if item is None:
            # Dummy vector for missing tables.
            if all_embs:
                all_embs.append(np.zeros_like(all_embs[0]))
            else:
                # First table missing: temporary zero vector (TAPAS hidden size).
                all_embs.append(np.zeros((tapas_model.config.hidden_size,), dtype=np.float32))
            continue

        table = item["table"]
        df = build_pandas_table(table)

        all_rows = table.get("table_rows", [])
        rows_to_use = table_rows_map.get(doc_id)
        if not rows_to_use:
            rows_to_use = list(range(len(all_rows)))

        row_embs: List[np.ndarray] = []

        with torch.no_grad():
            for r_idx in rows_to_use:
                if 0 <= r_idx < len(all_rows):
                    cells = all_rows[r_idx]
                    cell_texts = [c.get("text", "") for c in cells]
                    row_str = " | ".join(cell_texts).strip()
                else:
                    row_str = ""

                if not row_str:
                    row_str = f"row_{r_idx}"

                enc = tapas_tokenizer(
                    table=df,
                    queries=row_str,
                    padding="max_length",
                    return_tensors="pt",
                )
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                out = tapas_model(**enc)
                cls = out.last_hidden_state[:, 0, :]  # (1, H)
                cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
                row_embs.append(cls.cpu().numpy()[0])

        if row_embs:
            table_emb = np.mean(np.stack(row_embs, axis=0), axis=0)
        else:
            table_emb = np.zeros((tapas_model.config.hidden_size,), dtype=np.float32)

        all_embs.append(table_emb.astype(np.float32))

    return np.stack(all_embs, axis=0)


def main():
    print(f"Using device: {DEVICE}")

    # 1) Load id maps (gold-chain subset).
    text_id_map = load_json_map(MAP_TEXT_IDS)
    table_id_map = load_json_map(MAP_TABLE_IDS)

    text_id_list = invert_id_map(text_id_map)
    table_id_list = invert_id_map(table_id_map)

    print(f"[TEXT] #gold text docs:  {len(text_id_list)}")
    print(f"[TABLE] #gold table docs: {len(table_id_list)}")

    # 2) Build the text corpus (content only).
    gold_texts = build_text_corpus_for_gold(text_id_list)

    # 3) Load raw tables and evidence row indices.
    table_raw_map = load_tables_for_gold(table_id_list)
    table_rows_map = collect_table_rows_from_gold()

    # 4) Text embeddings: BGE-Large.
    print(f"Loading text encoder: {TEXT_MODEL_NAME}")
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(DEVICE)

    print("[TEXT] Encoding text evidences (content only)...")
    text_matrix = encode_texts(text_model, text_tokenizer, gold_texts, max_len=TEXT_MAX_LEN)
    print(f"[TEXT] Embedding matrix shape: {text_matrix.shape}")

    # 5) Table embeddings: TAPAS-base (header + evidence rows).
    print(f"Loading table encoder (TAPAS): {TABLE_MODEL_NAME}")
    tapas_tokenizer = TapasTokenizer.from_pretrained(TABLE_MODEL_NAME)
    tapas_model = TapasModel.from_pretrained(TABLE_MODEL_NAME).to(DEVICE)

    table_matrix = encode_tables_with_tapas(
        tapas_model=tapas_model,
        tapas_tokenizer=tapas_tokenizer,
        table_id_list=table_id_list,
        table_raw_map=table_raw_map,
        table_rows_map=table_rows_map,
    )
    print(f"[TABLE] Embedding matrix shape: {table_matrix.shape}")

    # 6) Save as the (N, D) matrix format expected by multimodal_encoder.py
    np.save(OUT_TEXT_NPY, text_matrix.astype(np.float32))
    np.save(OUT_TABLE_NPY, table_matrix.astype(np.float32))

    print(f"[SAVE] Text embeddings -> {OUT_TEXT_NPY}")
    print(f"[SAVE] Table embeddings -> {OUT_TABLE_NPY}")
    print("Done.")


if __name__ == "__main__":
    main()


