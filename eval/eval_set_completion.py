import json
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from typing import List, Dict, Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multimodal_encoder import MultimodalEvidenceEncoder
from biencoder import DEVICE, BASE_DIR
from grail_model import UnifiedBiEncoder

def build_global_pool(encoder: MultimodalEvidenceEncoder):
    encoder.eval()
    id_maps = {}
    offsets = {}
    current_offset = 0
    all_embs = []
    
    for mod in ["text", "table", "image"]:
        store = getattr(encoder, f"{mod}_store")
        ids = store.ids
        id_maps[mod] = {str(_id): i for i, _id in enumerate(ids)}
        offsets[mod] = current_offset
        
        mod_embs = []
        batch_size = 512
        with torch.no_grad():
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                reps = torch.stack([encoder.encode_step(mod, str(bid)) for bid in batch_ids])
                mod_embs.append(reps.cpu())
        
        mod_embs = torch.cat(mod_embs, dim=0)
        mod_embs = torch.nn.functional.normalize(mod_embs, p=2, dim=-1)
        all_embs.append(mod_embs)
        current_offset += len(ids)
        
    global_pool = torch.cat(all_embs, dim=0).to(DEVICE)
    return global_pool, id_maps, offsets

def run_evaluation(model, global_pool, id_maps, offsets, val_data, exclude_ctx=False, save_per_sample=False):
    model.eval()
    ks = [1, 3, 5, 10, 20, 50, 100]
    results = {
        "target_ranks": [],
        "interference_count": 0,
        "total_higher_ctx_count": 0,
        "total_ctx_count": 0,
        "total_interference_ev_count": 0,
        "total_score_diff": 0,
        "sample_occupancies": [],
        "per_modality_transition": {}, # {(ctx_m, tgt_m): [ranks]}
        "per_overlap_bucket": {}, # {bucket: [ranks]}
        "per_ev_count": {}, # {n_ev: [ranks]}
        "per_qtype": {},
        "S_tgt_list": [],
        "S_ctx_max_list": [],
        "DC_Gap_list": [],
        "per_sample_details": [] # List of dicts if save_per_sample is True
    }
    
    # Reverse id_maps for debugging
    rev_id_maps = {mod: {idx: id_str for id_str, idx in m.items()} for mod, m in id_maps.items()}
    rev_offsets = {v: k for k, v in offsets.items()}
    offset_list = sorted(list(offsets.values()))
    
    with torch.no_grad():
        for item in tqdm(val_data, desc="Evaluating", leave=False):
            question = item["question"]
            evidences = item.get("evidences", [])
            target = item["target"]
            
            # [NEW] Include distractors in the context during evaluation to test robustness
            # Filter distractors to ensure they exist in the embedding store
            distractors = item.get("distractors", [])
            valid_distractors = [
                d for d in distractors 
                if d["modality"] in id_maps and str(d["id"]) in id_maps[d["modality"]]
            ]
            all_context_evidences = evidences + valid_distractors
            
            # Use forward-like logic but for inference (one sample)
            # UnifiedBiEncoder forward normally handles batches, but for eval we can call its encoding logic.
            # However, UnifiedBiEncoder has different forward pass per option.
            # To be safe, we'll manually compute the representation based on the model's option.
            
            # Encode Question
            q_vec = model.question_encoder([question])[0] # (D,)
            
            # Determine Final Query Representation based on option
            if model.step2_option == "A":
                final_q = q_vec
            else:
                ev_reps = [model.evidence_encoder.encode_step(e["modality"], e["id"]) for e in all_context_evidences]
                h_sat_i = model.get_h_sat(q_vec, ev_reps)
                
                # Normalized-vector path, matching the training logic.
                h_sat_norm = torch.nn.functional.normalize(h_sat_i, p=2, dim=-1) if h_sat_i.norm() > 1e-6 else h_sat_i
                
                if model.step2_option == "B":
                    if not ev_reps:
                        final_q = q_vec
                    else:
                        H_seq = torch.stack(ev_reps).unsqueeze(0) # (1, N, D)
                        Q_seq = q_vec.unsqueeze(0).unsqueeze(1)    # (1, 1, D)
                        attn_output, _ = model.cross_attention(query=Q_seq, key=H_seq, value=H_seq)
                        final_q = torch.nn.functional.normalize(model.layer_norm(q_vec + attn_output.squeeze(0).squeeze(0)), p=2, dim=-1)
                elif model.step2_option == "C":
                    if h_sat_i.norm() > 1e-6:
                        # [MATCH TRAIN] Projection uses normalized h_sat
                        h_sat_norm = torch.nn.functional.normalize(h_sat_i, p=2, dim=-1)
                        proj_i = torch.dot(q_vec, h_sat_norm) * h_sat_norm
                        # [MATCH TRAIN] Gate input uses raw h_sat_i
                        g_i = model.W_gap(torch.cat([q_vec, h_sat_i], dim=-1))
                        q_perp_i = q_vec - g_i * proj_i
                    else:
                        q_perp_i = q_vec
                    
                    h_req_i = model.layer_norm_req(q_perp_i)
                    final_q = torch.nn.functional.normalize(h_req_i, p=2, dim=-1)
                else:
                    final_q = q_vec
            
            # Search
            raw_scores = (final_q.unsqueeze(0) @ global_pool.T).squeeze(0)
            scores = raw_scores.clone()
            
            # [ADD] Exclude Context Evidence (including distractors)
            ctx_indices = []
            for ev in all_context_evidences:
                ev_mod, ev_id = ev["modality"], str(ev["id"])
                if ev_mod in id_maps and ev_id in id_maps[ev_mod]:
                    idx = offsets[ev_mod] + id_maps[ev_mod][ev_id]
                    ctx_indices.append(idx)
            
            s_ctx_max = -1.0
            if ctx_indices:
                s_ctx_max = torch.max(raw_scores[ctx_indices]).item()

            if exclude_ctx and ctx_indices:
                scores[ctx_indices] = -1e9 # Mask out

            sorted_indices = torch.argsort(scores, descending=True)
            
            # Find target rank
            tgt_mod, tgt_id = target["modality"], str(target["id"])
            tgt_idx = offsets[tgt_mod] + id_maps[tgt_mod][tgt_id]
            target_rank = (sorted_indices == tgt_idx).nonzero(as_tuple=False)[0, 0].item() + 1
            target_score = raw_scores[tgt_idx].item()
            results["target_ranks"].append(target_rank)
            
            # DC-Gap metrics collection
            if s_ctx_max > -1.0:
                results["S_tgt_list"].append(target_score)
                results["S_ctx_max_list"].append(s_ctx_max)
                results["DC_Gap_list"].append(target_score - s_ctx_max)

            # Modality analysis
            ctx_m_str = "+".join(sorted(list(set(e["modality"] for e in evidences)))) if evidences else "None"
            m_key = f"{ctx_m_str}->{tgt_mod}"
            if m_key not in results["per_modality_transition"]:
                results["per_modality_transition"][m_key] = []
            results["per_modality_transition"][m_key].append({
                "rank": target_rank,
                "dc_gap": target_score - s_ctx_max if s_ctx_max > -1.0 else None
            })

            # Overlap analysis
            overlap = item.get("overlap_count", -1)
            if overlap != -1:
                bucket = "0" if overlap == 0 else "1-5" if overlap <= 5 else "6-15" if overlap <= 15 else "16+"
                if bucket not in results["per_overlap_bucket"]:
                    results["per_overlap_bucket"][bucket] = []
                results["per_overlap_bucket"][bucket].append(target_rank)
            
            # Evidence count (Hop) analysis
            n_ev = len(evidences)
            ev_bucket = str(n_ev) if n_ev < 2 else "2+"
            if ev_bucket not in results["per_ev_count"]:
                results["per_ev_count"][ev_bucket] = []
            results["per_ev_count"][ev_bucket].append(target_rank)
            
            # Question-type analysis
            qtype = item.get("question_type", "unknown")
            if qtype not in results["per_qtype"]:
                results["per_qtype"][qtype] = []
            dc_gap_val = target_score - s_ctx_max if s_ctx_max > -1.0 else None
            results["per_qtype"][qtype].append((target_rank, dc_gap_val))
            
            # Interference & Occupancy
            sample_higher_count = 0
            results["total_ctx_count"] += len(all_context_evidences)
            
            for idx in ctx_indices:
                ev_score = scores[idx].item()
                if not exclude_ctx and ev_score > (target_score + 1e-6):
                    sample_higher_count += 1
                    results["total_higher_ctx_count"] += 1
                    results["total_interference_ev_count"] += 1
                    results["total_score_diff"] += (ev_score - target_score)
            
            if len(all_context_evidences) > 0:
                results["sample_occupancies"].append(sample_higher_count / len(all_context_evidences))
                if sample_higher_count > 0:
                    results["interference_count"] += 1
            
            if save_per_sample:
                top1_idx = sorted_indices[0].item()
                # Find modality
                current_mod = "text"
                for o in offset_list:
                    if top1_idx >= o:
                        current_mod = rev_offsets[o]
                
                top1_id = rev_id_maps[current_mod][top1_idx - offsets[current_mod]]
                
                results["per_sample_details"].append({
                    "qid": item["qid"],
                    "target_rank": target_rank,
                    "target_id": tgt_id,
                    "target_mod": tgt_mod,
                    "top1_id": top1_id,
                    "top1_mod": current_mod,
                    "top1_score": raw_scores[top1_idx].item(),
                    "target_score": target_score,
                    "S_ctx_max": s_ctx_max,
                    "DC_Gap": target_score - s_ctx_max if s_ctx_max > -1.0 else None,
                    "overlap_count": item.get("overlap_count", 0),
                    "sample_higher_count": sample_higher_count,
                    "num_evidences": len(evidences),
                    "num_context": len(all_context_evidences)
                })

                    
    # Metrics
    target_ranks = np.array(results["target_ranks"])
    metrics = {f"Recall@{k}": float(np.mean(target_ranks <= k)) for k in ks}
    metrics["MRR"] = float(np.mean(1.0 / target_ranks))
    metrics["Inter%"] = float((results["interference_count"] / len(val_data)) * 100)
    metrics["Occ_Micro%"] = float((results["total_higher_ctx_count"] / results["total_ctx_count"]) * 100 if results["total_ctx_count"] > 0 else 0)
    metrics["Occ_Macro%"] = float(np.mean(results["sample_occupancies"]) * 100 if results["sample_occupancies"] else 0)
    metrics["Gap"] = float(results["total_score_diff"] / results["total_interference_ev_count"] if results["total_interference_ev_count"] > 0 else 0)
    metrics["S_tgt"] = float(np.mean(results["S_tgt_list"]) if results["S_tgt_list"] else 0)
    metrics["S_ctx_max"] = float(np.mean(results["S_ctx_max_list"]) if results["S_ctx_max_list"] else 0)
    metrics["DC_Gap"] = float(np.mean(results["DC_Gap_list"]) if results["DC_Gap_list"] else 0)
    metrics["DC_Gap_std"] = float(np.std(results["DC_Gap_list"]) if results["DC_Gap_list"] else 0)
    
    # Breakdown analysis
    m_breakdown = {}
    for key, items in results["per_modality_transition"].items():
        ranks = np.array([item["rank"] for item in items])
        dc_gaps = [item["dc_gap"] for item in items if item["dc_gap"] is not None]
        m_breakdown[key] = {
            "R@1": float(np.mean(ranks <= 1)),
            "R@5": float(np.mean(ranks <= 5)),
            "R@20": float(np.mean(ranks <= 20)),
            "R@50": float(np.mean(ranks <= 50)),
            "R@100": float(np.mean(ranks <= 100)),
            "DC_Gap": float(np.mean(dc_gaps)) if dc_gaps else 0.0,
            "count": len(ranks)
        }
    metrics["modality_breakdown"] = m_breakdown
    
    # Overlap bucket analysis
    o_breakdown = {}
    for bucket, ranks in results["per_overlap_bucket"].items():
        ranks = np.array(ranks)
        o_breakdown[bucket] = {
            "R@1": float(np.mean(ranks <= 1)),
            "count": len(ranks)
        }
    metrics["overlap_breakdown"] = o_breakdown
    
    # Evidence count (Hop) breakdown
    e_breakdown = {}
    for bucket, ranks in results["per_ev_count"].items():
        ranks = np.array(ranks)
        e_breakdown[bucket] = {
            "R@1": float(np.mean(ranks <= 1)),
            "count": len(ranks)
        }
    metrics["ev_count_breakdown"] = e_breakdown
    
    # Question-type breakdown
    q_breakdown = {}
    grouped_items = {"uni-": [], "intersect": [], "compare": [], "compose": [], "other": []}
    
    for qt, items in results["per_qtype"].items():
        ranks_arr = np.array([x[0] for x in items])
        dc_gaps = [x[1] for x in items if x[1] is not None]
        if len(ranks_arr) > 0:
            q_breakdown[qt] = {
                "R@1": float((ranks_arr <= 1).mean()),
                "R@3": float((ranks_arr <= 3).mean()),
                "R@5": float((ranks_arr <= 5).mean()),
                "R@10": float((ranks_arr <= 10).mean()),
                "R@20": float((ranks_arr <= 20).mean()),
                "R@50": float((ranks_arr <= 50).mean()),
                "R@100": float((ranks_arr <= 100).mean()),
                "DC_Gap": float(np.mean(dc_gaps)) if dc_gaps else 0.0,
                "DC_Gap_std": float(np.std(dc_gaps)) if dc_gaps else 0.0,
                "count": len(ranks_arr)
            }
        else:
            q_breakdown[qt] = {"count": 0}
            
        # Grouping logic
        if qt in ['TextQ', 'TableQ', 'ImageListQ', 'ImageQ']:
            group_key = 'uni-'
        elif qt.startswith('Compose'):
            group_key = 'compose'
        elif qt.startswith('Compare'):
            group_key = 'compare'
        elif qt.startswith('Intersect'):
            group_key = 'intersect'
        else:
            group_key = 'other'
        grouped_items[group_key].extend(items)
        
    metrics["qtype_breakdown"] = q_breakdown
    
    grouped_breakdown = {}
    for g_key, g_items in grouped_items.items():
        if len(g_items) > 0:
            g_ranks_arr = np.array([x[0] for x in g_items])
            g_dc_gaps = [x[1] for x in g_items if x[1] is not None]
            grouped_breakdown[g_key] = {
                "R@1": float((g_ranks_arr <= 1).mean()),
                "R@3": float((g_ranks_arr <= 3).mean()),
                "R@5": float((g_ranks_arr <= 5).mean()),
                "R@10": float((g_ranks_arr <= 10).mean()),
                "R@20": float((g_ranks_arr <= 20).mean()),
                "R@50": float((g_ranks_arr <= 50).mean()),
                "R@100": float((g_ranks_arr <= 100).mean()),
                "DC_Gap": float(np.mean(g_dc_gaps)) if g_dc_gaps else 0.0,
                "DC_Gap_std": float(np.std(g_dc_gaps)) if g_dc_gaps else 0.0,
                "count": len(g_ranks_arr)
            }
        else:
            grouped_breakdown[g_key] = {"count": 0}
    metrics["grouped_qtype_breakdown"] = grouped_breakdown
    
    if save_per_sample:
        metrics["per_sample_details"] = results["per_sample_details"]
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tag", type=str, required=True)
    parser.add_argument("--job_id", type=str, default="default")
    parser.add_argument("--ckpt_dir", type=str, default=BASE_DIR)
    parser.add_argument("--exclude_ctx", action="store_true")
    parser.add_argument("--save_details", action="store_true")
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--s2", type=str, default=None, choices=["A", "B", "C"], help="Query steering option")
    args = parser.parse_args()

    # Use specified validation file if provided; default to the renewal val split.
    if args.val_file and os.path.exists(args.val_file):
        val_file = args.val_file
    else:
        val_file = os.path.join(BASE_DIR, "mmqa_masked_renewal_val.jsonl")
    
    print(f"Using validation file: {val_file}")
    val_data = []
    with open(val_file, "r") as f:
        for line in f:
            val_data.append(json.loads(line))
            
    # Initialize model
    if args.s2:
        option_s2 = args.s2
    else:
        # Fallback to old heuristic: find B or C in run_tag
        if "_3B" in args.run_tag or "_0B" in args.run_tag: option_s2 = "B"
        elif "_3C" in args.run_tag or "_0C" in args.run_tag: option_s2 = "C"
        else: option_s2 = args.run_tag[1] # Original logic
        
    print(f"Using Step 2 Option: {option_s2}")
    model = UnifiedBiEncoder(base_dir=BASE_DIR, step2_option=option_s2)
    
    # Checkpoint path: biencoder_{run_tag}.pt  (e.g. run_tag=3C_freeze_dy)
    ckpt_path = os.path.join(args.ckpt_dir, f"biencoder_{args.run_tag}.pt")
    if not os.path.exists(ckpt_path) and args.run_tag.startswith("biencoder_"):
        ckpt_path = os.path.join(args.ckpt_dir, f"{args.run_tag}.pt")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return
        
    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=False)
    model.to(DEVICE)
    
    # Build global pool
    global_pool, id_maps, offsets = build_global_pool(model.evidence_encoder)
    
    # Run Evaluation
    metrics = run_evaluation(model, global_pool, id_maps, offsets, val_data, 
                            exclude_ctx=args.exclude_ctx, 
                            save_per_sample=args.save_details)

    # Save results
    save_dir = os.path.join(BASE_DIR, "evaluation", args.job_id)
    os.makedirs(save_dir, exist_ok=True)
    suffix = "_exclude" if args.exclude_ctx else ""
    output_path = os.path.join(save_dir, f"{args.run_tag}{suffix}.json")
    
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Results saved to {output_path}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
