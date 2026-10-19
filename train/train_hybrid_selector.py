"""
Stage 3: Task-Adaptive Hybrid query selector (paper Section 4.3).

Trains a self-supervised, performance-driven router that chooses, per question,
between the Additive specialist (Option B) and the GRAIL specialist (Option C):

  1. Load the validation split.
  2. Compute per-sample target ranks for B and C in exclude-context mode.
  3. Build pseudo-labels (Eq. 13):  y = 0 if rank_B <= rank_C else 1.
  4. Fine-tune a BERT-base classifier on question text -> pseudo-label.
  5. Save the selector to hybrid_selector.pt.
  6. Report hybrid routing performance on the test split (vs. B, C, and oracle).

Requires the trained specialists:
  biencoder_3B_freeze.pt       (Additive, Option B)
  biencoder_3C_freeze_dy.pt    (GRAIL,    Option C)
"""

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Repo root (parent of this train/ folder) holds the shared library modules.
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT_DIR)
# Data/checkpoint directory. Defaults to repo root; override with MMQA_DATA_DIR.
BASE_DIR = os.environ.get("MMQA_DATA_DIR", _ROOT_DIR)

from grail_model import UnifiedBiEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Question-type grouping for per-category reporting.
CATEGORIES = {
    "Uni-modal": ["TableQ", "TextQ", "ImageListQ"],
    "Intersect": ["Intersect(ImageListQ,TableQ)", "Intersect(ImageListQ,TextQ)", "Intersect(TableQ,TextQ)"],
    "Compose": [
        "Compose(TextQ,TableQ)", "Compose(ImageQ,TableQ)", "Compose(TextQ,ImageListQ)",
        "Compose(TableQ,ImageListQ)", "Compose(TableQ,TextQ)", "Compose(ImageQ,TextQ)"
    ],
    "Compare": [
        "Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))",
        "Compare(Compose(TableQ,ImageQ),TableQ)",
        "Compare(TableQ,Compose(TableQ,TextQ))"
    ],
}


def get_category(qtype: str) -> str:
    if qtype in CATEGORIES["Uni-modal"]:
        return "Uni-modal"
    if qtype in CATEGORIES["Intersect"]:
        return "Intersect"
    if qtype in CATEGORIES["Compose"]:
        return "Compose"
    if "Compare" in qtype or qtype in CATEGORIES["Compare"]:
        return "Compare"
    return "Uni-modal"


def build_global_pool(encoder):
    """Project every corpus item into the shared space once (batched)."""
    encoder.eval()
    id_maps, offsets = {}, {}
    current_offset = 0
    all_embs = []
    with torch.no_grad():
        for mod in ["text", "table", "image"]:
            store = getattr(encoder, f"{mod}_store")
            ids = store.ids
            id_maps[mod] = {str(_id): i for i, _id in enumerate(ids)}
            offsets[mod] = current_offset

            weights = getattr(encoder, f"{mod}_embedding").weight
            proj_layer = getattr(encoder, f"{mod}_proj")
            ln_layer = getattr(encoder, f"layer_norm_{mod}")

            chunk_size = 50000
            mod_proj_embs = []
            for i in range(0, weights.shape[0], chunk_size):
                out = proj_layer(weights[i:i + chunk_size])
                if encoder.use_layernorm:
                    out = ln_layer(out)
                out = torch.nn.functional.normalize(out, p=2, dim=-1)
                mod_proj_embs.append(out.cpu())

            all_embs.append(torch.cat(mod_proj_embs, dim=0))
            current_offset += len(ids)

    global_pool = torch.cat(all_embs, dim=0).cpu()
    return global_pool, id_maps, offsets


def get_ranks(model, global_pool, id_maps, offsets, data, exclude_ctx=True):
    """Target rank per sample under the model's steering (exclude-context mode)."""
    model.eval()
    questions = [item["question"] for item in data]
    q_vecs_list = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(questions), batch_size):
            q_vecs_list.append(model.question_encoder(questions[i:i + batch_size]))
        q_vecs = torch.cat(q_vecs_list, dim=0)

    final_qs_list = []
    with torch.no_grad():
        for i_item, item in enumerate(data):
            q_vec = q_vecs[i_item]
            ev_reps_indices = []
            for e in item.get("evidences", []):
                ev_mod, ev_id = e["modality"], str(e["id"])
                if ev_mod in id_maps and ev_id in id_maps[ev_mod]:
                    ev_reps_indices.append(offsets[ev_mod] + id_maps[ev_mod][ev_id])

            ev_reps = [global_pool[idx].to(DEVICE) for idx in ev_reps_indices] if ev_reps_indices else []
            h_sat_i = model.get_h_sat(q_vec, ev_reps)

            if model.step2_option == "A":
                final_q = q_vec
            elif model.step2_option == "B":
                h_sat_norm = torch.nn.functional.normalize(h_sat_i, p=2, dim=-1) if h_sat_i.norm() > 1e-6 else h_sat_i
                final_q = torch.nn.functional.normalize(q_vec + h_sat_norm, p=2, dim=-1)
            else:  # Option C (GRAIL): use the shared steering logic
                final_q, _ = model.steer_query(q_vec, h_sat_i)
            final_qs_list.append(final_q.cpu())

        final_qs = torch.stack(final_qs_list)
        raw_scores = final_qs @ global_pool.T

    ranks = []
    for i_item, item in enumerate(data):
        target = item["target"]
        scores = raw_scores[i_item].clone()
        if exclude_ctx:
            for e in item.get("evidences", []):
                ev_mod, ev_id = e["modality"], str(e["id"])
                if ev_mod in id_maps and ev_id in id_maps[ev_mod]:
                    scores[offsets[ev_mod] + id_maps[ev_mod][ev_id]] = -1e9
        tgt_idx = offsets[target["modality"]] + id_maps[target["modality"]][str(target["id"])]
        target_score = raw_scores[i_item, tgt_idx].item()
        ranks.append(int((scores > target_score).sum().item()) + 1)
    return ranks


class QDataset(Dataset):
    def __init__(self, questions, labels, tokenizer, max_len=64):
        self.samples = [{"question": q, "label": l} for q, l in zip(questions, labels)]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item['question'], truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long),
        }


def train_selector(train_questions, train_labels, save_path, epochs=4):
    print("Fine-tuning BERT-base hybrid selector...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_loader = DataLoader(QDataset(train_questions, train_labels, tokenizer), batch_size=32, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                batch['input_ids'].to(DEVICE),
                attention_mask=batch['attention_mask'].to(DEVICE),
                labels=batch['labels'].to(DEVICE),
            )
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
        print(f"  Epoch {epoch}/{epochs} - loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved hybrid selector: {save_path}")


def get_selector_predictions(model_path, data):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    questions = [item["question"] for item in data]
    preds = []
    with torch.no_grad():
        for i in range(0, len(questions), 128):
            inputs = tokenizer(questions[i:i + 128], truncation=True, padding=True, max_length=64, return_tensors="pt").to(DEVICE)
            preds.extend(torch.argmax(model(**inputs).logits, dim=1).cpu().numpy().tolist())
    del model
    torch.cuda.empty_cache()
    return preds


def compute_cat_metrics(ranks_list, data):
    cat_ranks = defaultdict(list)
    for rank, item in zip(ranks_list, data):
        cat_ranks[get_category(item.get("question_type", "unknown"))].append(rank)

    res = {}
    r1_list, r10_list, mrr_list = [], [], []
    for cat in ["Uni-modal", "Intersect", "Compare", "Compose"]:
        rs = np.array(cat_ranks[cat])
        r1 = float(np.mean(rs <= 1) * 100)
        r10 = float(np.mean(rs <= 10) * 100)
        mrr = float(np.mean(1.0 / rs))
        res[cat] = {"count": len(rs), "R@1": r1, "R@10": r10, "MRR": mrr}
        r1_list.append(r1); r10_list.append(r10); mrr_list.append(mrr)

    overall = np.array(ranks_list)
    res["Micro Avg"] = {"count": len(overall), "R@1": float(np.mean(overall <= 1) * 100),
                        "R@10": float(np.mean(overall <= 10) * 100), "MRR": float(np.mean(1.0 / overall))}
    res["Macro Avg"] = {"count": len(overall), "R@1": float(np.mean(r1_list)),
                        "R@10": float(np.mean(r10_list)), "MRR": float(np.mean(mrr_list))}
    return res


def main():
    val_file = os.path.join(BASE_DIR, "mmqa_masked_renewal_val.jsonl")
    test_file = os.path.join(BASE_DIR, "mmqa_masked_renewal_test.jsonl")
    with open(val_file) as f:
        val_data = [json.loads(line) for line in f]
    with open(test_file) as f:
        test_data = [json.loads(line) for line in f]
    print(f"Loaded {len(val_data)} val / {len(test_data)} test samples.")

    # Load the two specialists.
    model_b = UnifiedBiEncoder(base_dir=BASE_DIR, step2_option="B")
    model_b.load_state_dict(torch.load(os.path.join(BASE_DIR, "biencoder_3B_freeze.pt"), map_location="cpu"), strict=False)
    model_b.to(DEVICE)

    model_c = UnifiedBiEncoder(base_dir=BASE_DIR, step2_option="C")
    model_c.load_state_dict(torch.load(os.path.join(BASE_DIR, "biencoder_3C_freeze_dy.pt"), map_location="cpu"), strict=False)
    model_c.to(DEVICE)

    print("Building global representation pool...")
    global_pool, id_maps, offsets = build_global_pool(model_b.evidence_encoder)

    # 1. Performance-driven pseudo-labels on the validation split (Eq. 13).
    print("Computing validation ranks for B and C...")
    val_ranks_b = get_ranks(model_b, global_pool, id_maps, offsets, val_data, exclude_ctx=True)
    val_ranks_c = get_ranks(model_c, global_pool, id_maps, offsets, val_data, exclude_ctx=True)
    val_questions = [item["question"] for item in val_data]
    val_labels = [0 if rb <= rc else 1 for rb, rc in zip(val_ranks_b, val_ranks_c)]

    # 2. Train the selector.
    save_path = os.path.join(BASE_DIR, "hybrid_selector.pt")
    train_selector(val_questions, val_labels, save_path)

    # 3. Evaluate routing on the test split.
    print("Computing test ranks for B and C...")
    test_ranks_b = get_ranks(model_b, global_pool, id_maps, offsets, test_data, exclude_ctx=True)
    test_ranks_c = get_ranks(model_c, global_pool, id_maps, offsets, test_data, exclude_ctx=True)

    preds = get_selector_predictions(save_path, test_data)
    test_ranks_hybrid = [rb if p == 0 else rc for p, rb, rc in zip(preds, test_ranks_b, test_ranks_c)]
    test_ranks_oracle = [min(rb, rc) for rb, rc in zip(test_ranks_b, test_ranks_c)]

    mets = {
        "Additive (B)": compute_cat_metrics(test_ranks_b, test_data),
        "GRAIL (C)": compute_cat_metrics(test_ranks_c, test_data),
        "Hybrid": compute_cat_metrics(test_ranks_hybrid, test_data),
        "Oracle": compute_cat_metrics(test_ranks_oracle, test_data),
    }

    cats = ["Uni-modal", "Intersect", "Compare", "Compose", "Micro Avg", "Macro Avg"]
    print("\n=== Hybrid routing on TEST split ===")
    for metric_key in ["R@1", "R@10", "MRR"]:
        print(f"\n--- {metric_key} ---")
        print(f"{'Category':<12} | " + " | ".join(f"{k:<14}" for k in mets))
        for c in cats:
            vals = " | ".join(f"{mets[k][c][metric_key]:<14.4f}" for k in mets)
            print(f"{c:<12} | {vals}")

    out_file = os.path.join(BASE_DIR, "hybrid_selector_report.json")
    with open(out_file, "w") as f:
        json.dump(mets, f, indent=4)
    print(f"\nSaved report: {out_file}")


if __name__ == "__main__":
    main()
