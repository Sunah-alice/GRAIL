import json
import os
from typing import List, Dict, Any

# Data directory. Defaults to the repo root; override with MMQA_DATA_DIR.
BASE_DIR = os.environ.get("MMQA_DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_FILE = os.path.join(BASE_DIR, "MMQA_train.jsonl")
DEV_FILE = os.path.join(BASE_DIR, "MMQA_dev.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "mmqa_strict_multi_hop_chains.jsonl")

def extract_strict_steps(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    seen = set()
    steps = []
    meta = item.get("metadata", {})
    table_id = meta.get("table_id")
    
    # Extract from intermediate_answers + final answers only.
    sources = []
    for step in meta.get("intermediate_answers", []):
        sources.extend(step)
    sources.extend(item.get("answers", []))
    
    for src in sources:
        # Text
        for inst in src.get("text_instances", []):
            ev_id = str(inst["doc_id"])
            if ("text", ev_id) not in seen:
                steps.append({"modality": "text", "id": ev_id})
                seen.add(("text", ev_id))
        # Image
        for inst in src.get("image_instances", []):
            ev_id = str(inst["doc_id"])
            if ("image", ev_id) not in seen:
                steps.append({"modality": "image", "id": ev_id})
                seen.add(("image", ev_id))
        # Table
        for row in src.get("table_indices", []):
            full_id = f"{table_id}#{row[0]}"
            if ("table", full_id) not in seen:
                steps.append({"modality": "table", "id": full_id})
                seen.add(("table", full_id))
    return steps

def build_strict_dataset():
    all_chains = []
    
    def process_file(path):
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                steps = extract_strict_steps(item)
                # Keep multi-hop chains (>=2 steps).
                if len(steps) >= 2:
                    chain = {
                        "qid": item["qid"],
                        "question": item["question"],
                        "type": item["metadata"].get("type"),
                        "answer": [a.get("answer") for a in item.get("answers", [])],
                        "steps": steps
                    }
                    all_chains.append(chain)
                    count += 1
        return count

    print("Processing Train...")
    train_count = process_file(TRAIN_FILE)
    print("Processing Dev...")
    dev_count = process_file(DEV_FILE)
    
    print(f"Total multi-hop chains found: {len(all_chains)}")

    # Masking and the train/val/test split are handled by 04_build_splits.py.
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for c in all_chains:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_chains)} chains to {OUTPUT_FILE}")


if __name__ == "__main__":
    build_strict_dataset()







