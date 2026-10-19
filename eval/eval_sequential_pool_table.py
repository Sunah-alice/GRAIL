import json
import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multimodal_encoder import MultimodalEvidenceEncoder
from biencoder import BASE_DIR, BiEncoder
from grail_model import UnifiedBiEncoder

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

def get_item_meta(idx, id_maps, offsets):
    if not hasattr(get_item_meta, "rev_offsets"):
        get_item_meta.rev_offsets = {v: k for k, v in offsets.items()}
        get_item_meta.offset_list = sorted(list(offsets.values()))
    current_mod = "text"
    for o in get_item_meta.offset_list:
        if idx >= o: current_mod = get_item_meta.rev_offsets[o]
    if not hasattr(get_item_meta, "rev_id_maps"):
        get_item_meta.rev_id_maps = {mod: {v: k for k, v in m.items()} for mod, m in id_maps.items()}
    return {"modality": current_mod, "id": get_item_meta.rev_id_maps[current_mod][idx - offsets[current_mod]]}

def encode_step2(model, question, context_items):
    model.eval()
    with torch.no_grad():
        q_vec_base = model.question_encoder([question])[0]
        ev_reps = [model.evidence_encoder.encode_step(e["modality"], e["id"]) for e in context_items]
        
        # Steering (Option B/C) uses the model's step2_option, mirroring grail_model.py
        # But we can just use the model's forward or its specific methods
        # To be safe and identical to before:
        h_sat = model.get_h_sat(q_vec_base, ev_reps)
        
        if model.step2_option == "C":
            if h_sat.norm() > 1e-6:
                h_sat_norm = torch.nn.functional.normalize(h_sat, p=2, dim=-1)
                proj_i = torch.dot(q_vec_base, h_sat_norm) * h_sat_norm
                g_i = model.W_gap(torch.cat([q_vec_base, h_sat], dim=-1))
                q_perp_i = q_vec_base - g_i * proj_i
                h_req_i = model.layer_norm_req(q_perp_i)
                final_q = torch.nn.functional.normalize(h_req_i, p=2, dim=-1)
            else:
                final_q = torch.nn.functional.normalize(q_vec_base, p=2, dim=-1)
        elif model.step2_option == "B":
            if h_sat.norm() > 1e-6:
                h_sat_norm = torch.nn.functional.normalize(h_sat, p=2, dim=-1)
                final_q = torch.nn.functional.normalize(q_vec_base + h_sat_norm, p=2, dim=-1)
            else:
                final_q = torch.nn.functional.normalize(q_vec_base, p=2, dim=-1)
        
        return final_q

def run_iterative(question, scores_3a, model, hop_config, global_pool, id_maps, offsets):
    all_ret_indices = []
    ctx_items = []
    base_size = hop_config[0]
    if base_size > 0:
        topk_base = torch.topk(scores_3a, base_size).indices.tolist()
        all_ret_indices.extend(topk_base)
        ctx_items.extend([get_item_meta(idx, id_maps, offsets) for idx in topk_base])
    
    q_final = None
    for h_size in hop_config[1:]:
        if h_size == 0: continue
        q_final = encode_step2(model, question, ctx_items)
        cur_scores = (q_final.unsqueeze(0) @ global_pool.T).squeeze(0)
        cur_scores[all_ret_indices] = -1e9
        topk_next = torch.topk(cur_scores, h_size).indices.tolist()
        all_ret_indices.extend(topk_next)
        ctx_items.extend([get_item_meta(idx, id_maps, offsets) for idx in topk_next])
    
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
            
    print(f"Loading models to {DEVICE_DIAG}...", flush=True)
    model_3a = BiEncoder(BASE_DIR)
    model_3a.load_state_dict(torch.load(os.path.join(BASE_DIR, "biencoder_3A_freeze.pt"), map_location=DEVICE_DIAG), strict=False)
    model_3a.to(DEVICE_DIAG).eval()
    
    model_3b = UnifiedBiEncoder(BASE_DIR, step2_option="B")
    model_3b.load_state_dict(torch.load(os.path.join(BASE_DIR, "biencoder_3B_freeze.pt"), map_location=DEVICE_DIAG), strict=False)
    model_3b.to(DEVICE_DIAG).eval()
    
    model_3c = UnifiedBiEncoder(BASE_DIR, step2_option="C")
    model_3c.load_state_dict(torch.load(os.path.join(BASE_DIR, "biencoder_3C_freeze_dy.pt"), map_location=DEVICE_DIAG), strict=False)
    model_3c.to(DEVICE_DIAG).eval()
    
    print("Building global pool...", flush=True)
    global_pool, id_maps, offsets = build_global_pool(model_3a.evidence_encoder)
    
    print("Pre-calculating 3A scores...", flush=True)
    qid_3a_info = {}
    base_correct_count = 0
    total_gold_count = 0
    
    for qid, info in tqdm(qid_groups.items()):
        with torch.no_grad():
            q_emb_raw = model_3a.question_encoder([info["question"]])
            h0 = torch.nn.functional.normalize(model_3a.layer_norm_set(q_emb_raw), p=2, dim=-1).squeeze(0)
            scores_3a = (h0.unsqueeze(0) @ global_pool.T).squeeze(0)
            ranks_3a = torch.argsort(torch.argsort(scores_3a, descending=True)) + 1
            gold_idx = []
            for mod, _id in info["gold_keys"]:
                if mod in id_maps and str(_id) in id_maps[mod]: gold_idx.append(offsets[mod] + id_maps[mod][str(_id)])
            qid_3a_info[qid] = {"scores": scores_3a, "ranks": ranks_3a, "gold_idx": gold_idx}
            
            top10_a = set(torch.topk(scores_3a, 10).indices.tolist())
            gold_idx_set = set(gold_idx)
            if gold_idx_set:
                base_correct_count += len(top10_a.intersection(gold_idx_set)) / len(gold_idx_set)

    base_recall = base_correct_count / len(qid_groups)

    experiments = [
        ("3B", [8, 1, 1], "8+1+1"),
        ("3B", [8, 2], "8+2"),
        ("3B", [5, 5], "5+5"),
        ("3B", [3, 7], "3+7"),
        ("3B", [3, 3, 4], "3+3+4"),
        ("3B", [3, 2, 3, 2], "3+2+3+2"),
        ("3B", [2, 2, 2, 2, 2], "2*5"),
        ("3B", [1]*10, "1*10"),
        ("3C", [8, 1, 1], "8+1+1"),
        ("3C", [8, 2], "8+2"),
        ("3C", [5, 5], "5+5"),
        ("3C", [3, 7], "3+7"),
        ("3C", [3, 3, 4], "3+3+4"),
        ("3C", [3, 2, 3, 2], "3+2+3+2"),
        ("3C", [2, 2, 2, 2, 2], "2*5"),
        ("3C", [1]*10, "1*10"),
    ]
    
    results = []

    print("\nStarting evaluations...", flush=True)
    for model_type, hop, name in experiments:
        print(f"Running {model_type} {name}...", flush=True)
        model_to_use = model_3b if model_type == "3B" else model_3c
        
        all_found_gold_ranks = []
        rescue_3a_sum = 0
        rescue_model_sum = 0
        noise_samples_count = 0
        correct_count = 0
        
        base_size = hop[0]
        
        for qid, info in tqdm(qid_groups.items(), leave=False):
            s3a = qid_3a_info[qid]["scores"]
            r3a = qid_3a_info[qid]["ranks"]
            gold_idx = set(qid_3a_info[qid]["gold_idx"])
            if not gold_idx: continue
            
            indices = run_iterative(info["question"], s3a, model_to_use, hop, global_pool, id_maps, offsets)
            
            top10_m = set(indices)
            top10_a = set(torch.topk(s3a, 10).indices.tolist())
            
            # Macro Recall
            correct_count += len(top10_m.intersection(gold_idx)) / len(gold_idx)
            
            rescued = [idx for idx in gold_idx if idx in top10_m and idx not in top10_a]
            for rg in rescued: all_found_gold_ranks.append(r3a[rg].item())

            top_base = set(torch.topk(s3a, base_size).indices.tolist())
            if len(top_base.intersection(gold_idx)) == 0:
                noise_samples_count += 1
                rescue_3a_sum += len(top10_a.intersection(gold_idx)) / len(gold_idx)
                rescue_model_sum += len(top10_m.intersection(gold_idx)) / len(gold_idx)
                
        avg_recall = correct_count / len(qid_groups)
        recall_diff = (avg_recall - base_recall) * 100.0
        
        avg_jump = np.mean(all_found_gold_ranks) if all_found_gold_ranks else 0
        std_jump = np.std(all_found_gold_ranks) if all_found_gold_ranks else 0
        
        avg_nrm = ((rescue_model_sum / noise_samples_count) - (rescue_3a_sum / noise_samples_count)) * 100.0 if noise_samples_count > 0 else 0
        
        results.append({
            "Type": "Traditional PRF" if model_type == "3B" else "Gap-Aware Iterative Retrieval (Ours)",
            "Steps": name,
            "Set-Rec@10": f"{recall_diff:+.2f}pp",
            "Jump": f"{avg_jump:.1f} (±{std_jump:.1f})",
            "NRM": f"{avg_nrm:+.2f}pp"
        })

    print("\n" + "="*80)
    print(f"Base 3A Recall: {base_recall:.4%}")
    print("-" * 80)
    print(f"{'Type':<40} | {'Steps':<10} | {'Set-Rec@10':<10} | {'Jump':<10} | {'NRM'}")
    print("-" * 80)
    current_type = ""
    for r in results:
        if r['Type'] != current_type:
            print(f"--- {r['Type']} ---")
            current_type = r['Type']
        print(f"{'':<40} | {r['Steps']:<10} | {r['Set-Rec@10']:<10} | {r['Jump']:<10} | {r['NRM']}")

if __name__ == "__main__":
    main()
