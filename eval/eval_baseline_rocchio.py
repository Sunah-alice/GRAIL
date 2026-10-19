import json
import os
import torch
import numpy as np
import csv
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multimodal_encoder import MultimodalEvidenceEncoder
from biencoder import BASE_DIR, BiEncoder

DEVICE_DIAG = "cuda" if torch.cuda.is_available() else "cpu"

def build_global_pool(encoder):
    encoder.eval()
    id_maps, offsets = {}, {}
    current_offset = 0
    all_embs = []
    for mod in ["text", "table", "image"]:
        store = getattr(encoder, f"{mod}_store")
        ids = store.ids
        id_maps[mod] = {str(_id): i for i, _id in enumerate(ids)}
        offsets[mod] = current_offset
        with torch.no_grad():
            num_items = store.num_items
            indices = torch.arange(num_items, device=DEVICE_DIAG)
            mod_embs = encoder._encode_indices(mod, indices)
            all_embs.append(mod_embs.cpu())
        current_offset += len(ids)
    global_pool = torch.cat(all_embs, dim=0).to(DEVICE_DIAG)
    return global_pool, id_maps, offsets

def run_rocchio(q_emb_raw, s3a, global_pool, base_size, total_k, alpha=1.0, beta=0.75):
    topk_base = torch.topk(s3a, base_size).indices.tolist()
    retrieved_embs = global_pool[topk_base]
    feedback_vec = torch.mean(retrieved_embs, dim=0)
    q_new = alpha * q_emb_raw + beta * feedback_vec
    q_new = torch.nn.functional.normalize(q_new, p=2, dim=-1)
    cur_scores = (q_new.unsqueeze(0) @ global_pool.T).squeeze(0)
    cur_scores[topk_base] = -1e9
    remaining_size = total_k - base_size
    topk_next = torch.topk(cur_scores, remaining_size).indices.tolist()
    all_ret_indices = topk_base + topk_next
    return all_ret_indices

def main():
    print("Loading data...", flush=True)
    val_file = "mmqa_masked_renewal_val.jsonl"
    qid_groups = {}
    with open(os.path.join(BASE_DIR, val_file), "r") as f:
        for line in f:
            item = json.loads(line)
            qid = item["qid"]
            if qid not in qid_groups: qid_groups[qid] = {"question": item["question"], "gold_keys": set()}
            for ev in item.get("evidences", []): qid_groups[qid]["gold_keys"].add((ev["modality"], str(ev["id"])))
            qid_groups[qid]["gold_keys"].add((item["target"]["modality"], str(item["target"]["id"])))
            
    print(f"Loading BiEncoder(3A) to {DEVICE_DIAG}...", flush=True)
    model_3a = BiEncoder(BASE_DIR)
    model_3a.load_state_dict(torch.load(os.path.join(BASE_DIR, "biencoder_3A_freeze.pt"), map_location=DEVICE_DIAG), strict=False)
    model_3a.to(DEVICE_DIAG).eval()
    
    print("Building global pool...", flush=True)
    global_pool, id_maps, offsets = build_global_pool(model_3a.evidence_encoder)
    
    print("Pre-calculating initial 3A scores & query embeddings...", flush=True)
    qid_3a_info = {}
    for qid, info in tqdm(qid_groups.items(), desc="Pre-calc 3A"):
        with torch.no_grad():
            q_emb_raw = model_3a.question_encoder([info["question"]])
            h0 = torch.nn.functional.normalize(model_3a.layer_norm_set(q_emb_raw), p=2, dim=-1).squeeze(0)
            scores_3a = (h0.unsqueeze(0) @ global_pool.T).squeeze(0)
            ranks_3a = torch.argsort(torch.argsort(scores_3a, descending=True)) + 1
            gold_idx = []
            for mod, _id in info["gold_keys"]:
                if mod in id_maps and str(_id) in id_maps[mod]: gold_idx.append(offsets[mod] + id_maps[mod][str(_id)])
            qid_3a_info[qid] = {"q_emb": h0, "scores": scores_3a, "ranks": ranks_3a, "gold_idx": gold_idx}

    # Rocchio experiment configurations.
    experiments = [
        (10, 10, 0.0, "K10-Baseline(3A)"),
        (10, 3, 0.25, "K10-Rocchio[3+7](b=0.25)"),
        (10, 3, 0.50, "K10-Rocchio[3+7](b=0.50)"),
        (10, 3, 0.75, "K10-Rocchio[3+7](b=0.75)"),
        (10, 3, 1.00, "K10-Rocchio[3+7](b=1.00)"),
        (10, 5, 0.50, "K10-Rocchio[5+5](b=0.50)"),
        (10, 5, 0.75, "K10-Rocchio[5+5](b=0.75)"),
        
        (20, 20, 0.0, "K20-Baseline(3A)"),
        (20, 5, 0.25, "K20-Rocchio[5+15](b=0.25)"),
        (20, 5, 0.50, "K20-Rocchio[5+15](b=0.50)"),
        (20, 5, 0.75, "K20-Rocchio[5+15](b=0.75)"),
        (20, 5, 1.00, "K20-Rocchio[5+15](b=1.00)"),
        (20, 10, 0.25, "K20-Rocchio[10+10](b=0.25)"),
        (20, 10, 0.50, "K20-Rocchio[10+10](b=0.50)"),
        (20, 10, 0.75, "K20-Rocchio[10+10](b=0.75)"),
        (20, 10, 1.00, "K20-Rocchio[10+10](b=1.00)"),
    ]
    
    print("\nStarting Rocchio Experiments...", flush=True)
    results = []
    
    for total_k, base_size, beta, name in experiments:
        all_found_gold_ranks = []
        rescue_3a_sum = 0
        rescue_rocchio_sum = 0
        noise_samples_count = 0
        
        macro_recall_sum = 0.0
        micro_correct_count = 0
        total_gold_count = 0
        
        for qid, info in qid_groups.items():
            s3a = qid_3a_info[qid]["scores"]
            r3a = qid_3a_info[qid]["ranks"]
            q_emb = qid_3a_info[qid]["q_emb"]
            gold_idx = set(qid_3a_info[qid]["gold_idx"])
            if not gold_idx: continue
            
            # 3A Baseline for total_k
            topk_a = set(torch.topk(s3a, total_k).indices.tolist())
            
            # Rocchio Feedback Retrieval
            if beta == 0.0:
                topk_r = topk_a
            else:
                indices = run_rocchio(q_emb, s3a, global_pool, base_size, total_k, alpha=1.0, beta=beta)
                topk_r = set(indices)
            
            # Micro Recall Metrics
            num_correct = len(topk_r.intersection(gold_idx))
            micro_correct_count += num_correct
            total_gold_count += len(gold_idx)
            
            # Macro Recall Metrics (Per-query Recall averaged)
            macro_recall_sum += num_correct / len(gold_idx)
            
            # Rank Jump
            rescued = [idx for idx in gold_idx if idx in topk_r and idx not in topk_a]
            for rg in rescued: all_found_gold_ranks.append(r3a[rg].item())
                
            # Noise Resilience Margin (NRM)
            top_base = set(torch.topk(s3a, base_size).indices.tolist())
            if len(top_base.intersection(gold_idx)) == 0:
                noise_samples_count += 1
                rescue_3a_sum += len(topk_a.intersection(gold_idx)) / len(gold_idx)
                rescue_rocchio_sum += len(topk_r.intersection(gold_idx)) / len(gold_idx)
                
        avg_macro_recall = macro_recall_sum / len(qid_groups)
        avg_micro_recall = micro_correct_count / total_gold_count
        avg_jump = np.mean(all_found_gold_ranks) if all_found_gold_ranks else 0
        avg_nrm = ((rescue_rocchio_sum / noise_samples_count) - (rescue_3a_sum / noise_samples_count)) * 100.0 if noise_samples_count > 0 else 0
        results.append([name, f"{avg_macro_recall:.4%}", f"{avg_micro_recall:.4%}", f"{avg_jump:.1f}", f"{avg_nrm:+.2f}pp"])

    print("\n" + "="*95, flush=True)
    header = ["Experiment", "Macro Recall", "Micro Recall", "Rank Jump", "NRM (Pure Rescue)"]
    print(f"{header[0]:<28} | {header[1]:<12} | {header[2]:<12} | {header[3]:<10} | {header[4]}", flush=True)
    print("-" * 100, flush=True)
    for r in results:
        print(f"{r[0]:<28} | {r[1]:<12} | {r[2]:<12} | {r[3]:<10} | {r[4]}", flush=True)
 
    with open("rocchio_extended_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)
    print("✅ Saved updated Rocchio Diagnostics to rocchio_extended_results.csv")

if __name__ == "__main__":
    main()
