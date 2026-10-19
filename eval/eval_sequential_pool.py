import json
import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import argparse
import csv

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

def encode_step2(model, question, context_items, option="C"):
    model.eval()
    with torch.no_grad():
        q_vec_base = model.question_encoder([question])[0]
        ev_reps = [model.evidence_encoder.encode_step(e["modality"], e["id"]) for e in context_items]
        h_sat = model.get_h_sat(q_vec_base, ev_reps)
        if option == "B":
            h_sat_norm = torch.nn.functional.normalize(h_sat, p=2, dim=-1) if h_sat.norm() > 1e-6 else h_sat
            final_q = torch.nn.functional.normalize(q_vec_base + h_sat_norm, p=2, dim=-1)
        else: # C
            if h_sat.norm() > 1e-6:
                h_sat_norm = torch.nn.functional.normalize(h_sat, p=2, dim=-1)
                proj_i = torch.dot(q_vec_base, h_sat_norm) * h_sat_norm
                g_i = model.W_gap(torch.cat([q_vec_base, h_sat], dim=-1))
                q_perp_i = q_vec_base - g_i * proj_i
                h_req_i = model.layer_norm_req(q_perp_i)
                final_q = torch.nn.functional.normalize(h_req_i, p=2, dim=-1)
            else:
                final_q = torch.nn.functional.normalize(q_vec_base, p=2, dim=-1)
        return final_q

def run_iterative(question, scores_3a, model, hop_config, global_pool, id_maps, offsets, option="C"):
    all_ret_indices = []
    ctx_items = []
    base_size = hop_config[0]
    if base_size > 0:
        topk_base = torch.topk(scores_3a, base_size).indices.tolist()
        all_ret_indices.extend(topk_base)
        ctx_items.extend([get_item_meta(idx, id_maps, offsets) for idx in topk_base])
    for h_size in hop_config[1:]:
        if h_size == 0: continue
        q_refined = encode_step2(model, question, ctx_items, option=option)
        cur_scores = (q_refined.unsqueeze(0) @ global_pool.T).squeeze(0)
        cur_scores[all_ret_indices] = -1e9
        topk_next = torch.topk(cur_scores, h_size).indices.tolist()
        all_ret_indices.extend(topk_next)
        ctx_items.extend([get_item_meta(idx, id_maps, offsets) for idx in topk_next])
    return set((e["modality"], str(e["id"])) for e in ctx_items), all_ret_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_file", type=str, default="mmqa_masked_renewal_val.jsonl")
    parser.add_argument("--ckpt_3a", type=str, default="biencoder_3A_freeze.pt")
    parser.add_argument("--ckpt_3c", type=str, default="biencoder_3C_freeze_dy.pt")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    qid_groups = {}
    with open(os.path.join(BASE_DIR, args.val_file), "r") as f:
        lines = f.readlines()
        if args.sample:
            import random
            lines = random.sample(lines, min(len(lines), args.sample))
        for line in lines:
            item = json.loads(line)
            qid = item["qid"]
            if qid not in qid_groups: qid_groups[qid] = {"question": item["question"], "gold_keys": set()}
            for ev in item.get("evidences", []): qid_groups[qid]["gold_keys"].add((ev["modality"], str(ev["id"])))
            qid_groups[qid]["gold_keys"].add((item["target"]["modality"], str(item["target"]["id"])))
            
    model_3c = UnifiedBiEncoder(BASE_DIR, step2_option="C")
    model_3c.load_state_dict(torch.load(os.path.join(BASE_DIR, args.ckpt_3c), map_location=DEVICE_DIAG), strict=False)
    model_3c.to(DEVICE_DIAG).eval()
    model_3a = BiEncoder(BASE_DIR)
    model_3a.load_state_dict(torch.load(os.path.join(BASE_DIR, args.ckpt_3a), map_location=DEVICE_DIAG), strict=False)
    model_3a.to(DEVICE_DIAG).eval()
    
    global_pool, id_maps, offsets = build_global_pool(model_3c.evidence_encoder)
    
    # 3rd Comprehensive Sweep
    experiments = [
        # Baseline
        (10, [10], "3A-Baseline-10"),
        (20, [20], "3A-Baseline-20"),
        # K=10: 2-Step Full Sweep
        (10, [1, 9], "K10-1+9"), (10, [2, 8], "K10-2+8"), (10, [3, 7], "K10-3+7"),
        (10, [4, 6], "K10-4+6"), (10, [5, 5], "K10-5+5"), (10, [6, 4], "K10-6+4"),
        (10, [7, 3], "K10-7+3"), (10, [8, 2], "K10-8+2"), (10, [9, 1], "K10-9+1"),
        # K=10: Multi-Step
        (10, [1]*10, "K10-[1x10]"),
        (10, [1, 3, 3, 3], "K10-[1+3x3]"),
        (10, [2]*5, "K10-[2x5]"),
        (10, [2, 4, 4], "K10-[2+4x2]"),
        (10, [3, 2, 3, 2], "K10-[3+2+3+2]"),
        (10, [4, 2, 2, 2], "K10-[4+2x3]"),
        (10, [4, 3, 3], "K10-[4+3+3]"),
        (10, [6, 2, 2], "K10-[6+2+2]"),
        (10, [8, 1, 1], "K10-[8+1+1]"),
        # K=20: Selection
        (20, [2, 18], "K20-2+18"), (20, [4, 16], "K20-4+16"), (20, [10, 10], "K20-10+10"), (20, [15, 5], "K20-15+5"),
        (20, [2]*10, "K20-[2x10]"),
        (20, [4]*5, "K20-[4x5]"),
        (20, [5]*4, "K20-[5x4]"),
        (20, [2, 6, 6, 6], "K20-[2+6x3]"),
        (20, [4, 8, 8], "K20-[4+8x2]"),
        (20, [10, 5, 5], "K20-[10+5+5]"),
    ]
    
    results = defaultdict(lambda: defaultdict(lambda: {
        "recall_sum": 0.0, 
        "rescue_num_user": 0, "rescue_den_user": 0, 
        "rescue_num_base": 0, "rescue_den_base": 0, 
        "drift_loss": 0, "count": 0
    }))

    for qid, info in tqdm(qid_groups.items(), desc="Iterative V3 Sweep"):
        question = info["question"]
        gold_keys = info["gold_keys"]
        num_gold = len(gold_keys)
        with torch.no_grad():
            q_emb = model_3a.question_encoder([question])
            scores_3a = (torch.nn.functional.normalize(model_3a.layer_norm_set(q_emb), p=2, dim=-1) @ global_pool.T).squeeze(0)
            
        top10_3a_indices = torch.topk(scores_3a, 10).indices.tolist()
        top20_3a_indices = torch.topk(scores_3a, 20).indices.tolist()
        top10_3a_keys = set((get_item_meta(idx, id_maps, offsets)["modality"], str(get_item_meta(idx, id_maps, offsets)["id"])) for idx in top10_3a_indices)
        top20_3a_keys = set((get_item_meta(idx, id_maps, offsets)["modality"], str(get_item_meta(idx, id_maps, offsets)["id"])) for idx in top20_3a_indices)
        
        missed_by_a = {10: gold_keys - top10_3a_keys, 20: gold_keys - top20_3a_keys}

        for exp_idx, (total_k, hop_config, exp_name) in enumerate(experiments):
            if len(hop_config) == 1:
                keys_baseline = top10_3a_keys if total_k == 10 else top20_3a_keys
                recall = len(keys_baseline.intersection(gold_keys)) / num_gold
                results[exp_idx]["3A"]["recall_sum"] += recall
                results[exp_idx]["3A"]["count"] += 1
                continue

            # Base info
            base_k = hop_config[0]
            indices_base = torch.topk(scores_3a, base_k).indices.tolist()
            keys_base = set((get_item_meta(idx, id_maps, offsets)["modality"], str(get_item_meta(idx, id_maps, offsets)["id"])) for idx in indices_base)
            missed_by_base = gold_keys - keys_base

            # 3C only
            keys_final, _ = run_iterative(question, scores_3a, model_3c, hop_config, global_pool, id_maps, offsets, option="C")
            
            recall = len(keys_final.intersection(gold_keys)) / num_gold
            gold_in_iter = keys_final.intersection(gold_keys)
            
            baseline_keys = top10_3a_keys if total_k == 10 else top20_3a_keys
            rescued_vs_baseline = gold_in_iter - baseline_keys
            rescued_vs_base = gold_in_iter - keys_base
            drift_lost_keys = baseline_keys.intersection(gold_keys) - gold_in_iter

            results[exp_idx]["3C"]["recall_sum"] += recall
            results[exp_idx]["3C"]["rescue_num_user"] += len(rescued_vs_baseline)
            results[exp_idx]["3C"]["rescue_den_user"] += len(missed_by_a[total_k])
            results[exp_idx]["3C"]["rescue_num_base"] += len(rescued_vs_base)
            results[exp_idx]["3C"]["rescue_den_base"] += len(missed_by_base)
            results[exp_idx]["3C"]["drift_loss"] += len(drift_lost_keys)
            results[exp_idx]["3C"]["count"] += 1

    csv_rows = []
    header = ["Experiment", "Recall", "Recall Recovery (RR)", "Stepwise Gain (SG)", "Gain vs 3A"]
    print("\n" + "="*120)
    print(f"{header[0]:<25} | {header[1]:<10} | {header[2]:<20} | {header[3]:<20} | {header[4]}")
    print("-" * 120)
    
    baseline_3a = {10: results[0]["3A"]["recall_sum"]/results[0]["3A"]["count"], 
                   20: results[1]["3A"]["recall_sum"]/results[1]["3A"]["count"]}

    for exp_idx, (total_k, hop_config, exp_name) in enumerate(experiments):
        if len(hop_config) == 1:
            avg_recall = results[exp_idx]["3A"]["recall_sum"] / results[exp_idx]["3A"]["count"]
            print(f"{exp_name:<25} | {avg_recall:<10.2%} | {'N/A':<20} | {'N/A':<20} | -")
            csv_rows.append([exp_name, f"{avg_recall:.4f}", "N/A", "N/A", "0.0"])
            continue
        
        res = results[exp_idx]["3C"]
        avg_recall = res["recall_sum"] / res["count"]
        rr = res["rescue_num_user"] / res["rescue_den_user"] if res["rescue_den_user"] > 0 else 0
        sg = res["rescue_num_base"] / res["rescue_den_base"] if res["rescue_den_base"] > 0 else 0
        gain = avg_recall - baseline_3a[total_k]
        print(f"{exp_name:<25} | {avg_recall:<10.2%} | {rr:<20.2%} | {sg:<20.2%} | {gain:+.2%}p")
        csv_rows.append([exp_name, f"{avg_recall:.4f}", f"{rr:.4f}", f"{sg:.4f}", f"{gain:.4f}"])

    with open("iterative_study_results_v3.csv", "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(header); writer.writerows(csv_rows)
    print(f"\n📊 Sweep v3 results saved to iterative_study_results_v3.csv")

if __name__ == "__main__":
    main()
