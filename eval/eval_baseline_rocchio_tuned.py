import torch
import numpy as np
import json
import os
import sys
import csv
from tqdm import tqdm
from typing import List, Dict, Any
import argparse

# Repo root (parent of this eval/ folder) holds the shared library modules.
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT_DIR)
# Data/checkpoint directory. Defaults to repo root; override with MMQA_DATA_DIR.
BASE_DIR = os.environ.get("MMQA_DATA_DIR", _ROOT_DIR)
from grail_model import UnifiedBiEncoder

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VAL_FILE = os.path.join(BASE_DIR, 'mmqa_masked_renewal_val.jsonl')
TEST_FILE = os.path.join(BASE_DIR, 'mmqa_masked_renewal_test.jsonl')

def load_data(file_path):
    data = []
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def build_pool(model):
    model.eval()
    encoder = model.evidence_encoder
    
    print("  - Loading base embeddings...")
    # Text
    emb_text = torch.from_numpy(np.load(os.path.join(BASE_DIR, 'embeddings_text.npy')))
    # Table
    emb_table = torch.from_numpy(np.load(os.path.join(BASE_DIR, 'embeddings_table.npy')))
    # Image
    img_embs_raw = np.load(os.path.join(BASE_DIR, 'mmqa_image_embeddings.npy'), allow_pickle=True).item()
    img_ids = sorted(img_embs_raw.keys())
    emb_image = torch.from_numpy(np.stack([img_embs_raw[iid] for iid in img_ids]))
    
    print("  - Running projections...")
    with torch.no_grad():
        p_t = torch.nn.functional.normalize(encoder.layer_norm_text(encoder.text_proj(emb_text.to(DEVICE))), p=2, dim=-1)
        p_tbl = torch.nn.functional.normalize(encoder.layer_norm_table(encoder.table_proj(emb_table.to(DEVICE))), p=2, dim=-1)
        p_i = torch.nn.functional.normalize(encoder.layer_norm_image(encoder.image_proj(emb_image.to(DEVICE))), p=2, dim=-1)
        
    pool = torch.cat([p_t, p_tbl, p_i], dim=0)
    
    # Id Maps
    with open(os.path.join(BASE_DIR, 'map_text_ids.json'), 'r') as f: map_text = json.load(f)
    with open(os.path.join(BASE_DIR, 'map_table_ids.json'), 'r') as f: map_table = json.load(f)
    
    offsets = {'text': 0, 'table': len(map_text), 'image': len(map_text) + len(map_table)}
    
    id_to_global_idx = {}
    for tid, idx in map_text.items(): id_to_global_idx[('text', str(tid))] = offsets['text'] + idx
    for tid, idx in map_table.items(): id_to_global_idx[('table', str(tid))] = offsets['table'] + idx
    for iid_idx, iid in enumerate(img_ids): id_to_global_idx[('image', str(iid))] = offsets['image'] + iid_idx
    
    return pool, id_to_global_idx

def precompute_data(model, pool, id_to_global_idx, data):
    model.eval()
    print("Precomputing query and context embeddings...")
    queries = []
    contexts = []
    targets = []
    
    with torch.no_grad():
        # Encode queries in batches for speed
        questions = [item['question'] for item in data]
        q_embs = []
        batch_size = 256
        for i in range(0, len(questions), batch_size):
            batch_qs = questions[i:i+batch_size]
            batch_embs = model.question_encoder(batch_qs)
            q_embs.append(batch_embs)
        q_embs = torch.cat(q_embs, dim=0)
        
        for i, item in enumerate(data):
            q_vec = q_embs[i]
            
            ev_indices = []
            for e in item['evidences']:
                key = (e['modality'], str(e['id']))
                if key in id_to_global_idx:
                    ev_indices.append(id_to_global_idx[key])
            
            target_key = (item['target']['modality'], str(item['target']['id']))
            if target_key not in id_to_global_idx:
                continue
            target_idx = id_to_global_idx[target_key]
            
            queries.append(q_vec)
            contexts.append(ev_indices)
            targets.append(target_idx)
            
    return torch.stack(queries), contexts, targets

def get_feedback_vectors(queries_tensor, contexts, pool, num_negs_list):
    """
    Precompute R_vec and N_vec for all samples to speed up grid search.
    """
    N, D = queries_tensor.shape
    print("Precomputing positive feedback vectors (R_vec)...")
    R_vecs = torch.zeros((N, D), device=DEVICE)
    for i, ctx in enumerate(contexts):
        if ctx:
            pos_embs = pool[ctx]
            r = torch.mean(pos_embs, dim=0)
            if r.norm() > 1e-6:
                R_vecs[i] = torch.nn.functional.normalize(r, p=2, dim=-1)
                
    # Initial scores to determine negative feedback
    print("Precomputing initial scores for negative feedback selection...")
    initial_scores = queries_tensor @ pool.T  # (N, Pool_Size)
    
    N_vecs_dict = {}
    for num_neg in num_negs_list:
        print(f"Precomputing negative feedback vectors (N_vec) for U={num_neg}...")
        N_vecs = torch.zeros((N, D), device=DEVICE)
        
        # Sort initial scores to find top non-context items
        # To avoid sorting the whole pool for each sample, we do a argsort on top elements or loop
        # Since this is a one-time precomputation, a simple loop over samples is fine
        for i in range(N):
            ctx_set = set(contexts[i])
            scores = initial_scores[i]
            # Get top K + len(ctx) elements
            topk_vals, topk_indices = torch.topk(scores, num_neg + len(ctx_set) + 5)
            neg_indices = []
            for idx in topk_indices.tolist():
                if idx not in ctx_set:
                    neg_indices.append(idx)
                    if len(neg_indices) == num_neg:
                        break
            if neg_indices:
                neg_embs = pool[neg_indices]
                n_vec = torch.mean(neg_embs, dim=0)
                if n_vec.norm() > 1e-6:
                    N_vecs[i] = torch.nn.functional.normalize(n_vec, p=2, dim=-1)
        N_vecs_dict[num_neg] = N_vecs
        
    return R_vecs, N_vecs_dict

def evaluate_rocchio_config(queries_tensor, R_vecs, N_vecs, targets, contexts, pool, alpha, beta, gamma):
    """
    Evaluate a single Rocchio configuration in a highly vectorized manner on GPU.
    """
    N = queries_tensor.shape[0]
    targets_tensor = torch.tensor(targets, device=DEVICE)
    
    # 1. Update queries
    q_news = alpha * queries_tensor + beta * R_vecs - gamma * N_vecs
    q_news = torch.nn.functional.normalize(q_news, p=2, dim=-1)
    
    # 2. Compute similarity scores
    scores = q_news @ pool.T  # (N, Pool_Size)
    
    # 3. Include setting ranks
    target_scores = scores[torch.arange(N), targets_tensor].unsqueeze(1)
    ranks_inc = (scores > target_scores).sum(dim=1) + 1
    
    # 4. Exclude setting ranks
    scores_exc = scores.clone()
    for i, ctx in enumerate(contexts):
        if ctx:
            scores_exc[i, ctx] = -1e9
    target_scores_exc = scores_exc[torch.arange(N), targets_tensor].unsqueeze(1)
    ranks_exc = (scores_exc > target_scores_exc).sum(dim=1) + 1
    
    # Move to CPU for metrics
    ranks_inc = ranks_inc.cpu().numpy()
    ranks_exc = ranks_exc.cpu().numpy()
    
    # Calculate main metrics
    metrics = {
        'Include': {
            'MRR': np.mean(1.0 / ranks_inc) * 100,
            'R@1': np.mean(ranks_inc <= 1) * 100,
            'R@5': np.mean(ranks_inc <= 5) * 100,
            'R@20': np.mean(ranks_inc <= 20) * 100,
            'R@100': np.mean(ranks_inc <= 100) * 100,
        },
        'Exclude': {
            'MRR': np.mean(1.0 / ranks_exc) * 100,
            'R@1': np.mean(ranks_exc <= 1) * 100,
            'R@5': np.mean(ranks_exc <= 5) * 100,
            'R@20': np.mean(ranks_exc <= 20) * 100,
            'R@100': np.mean(ranks_exc <= 100) * 100,
        }
    }
    
    # Interference metrics (Include setting only)
    total_gap = 0.0
    total_occ = 0.0
    inter_count = 0
    valid_samples = 0
    for i, ctx in enumerate(contexts):
        if ctx:
            valid_samples += 1
            ctx_scores = scores[i, ctx]
            t_score = scores[i, targets[i]].item()
            max_ctx_score = torch.max(ctx_scores).item()
            gap = max_ctx_score - t_score
            total_gap += gap
            
            higher_count = (ctx_scores > t_score).sum().item()
            total_occ += (higher_count / len(ctx))
            if higher_count > 0:
                inter_count += 1
                
    if valid_samples > 0:
        metrics['Include']['Inter%'] = (inter_count / valid_samples) * 100
        metrics['Include']['Occ%'] = (total_occ / valid_samples) * 100
        metrics['Include']['Gap'] = (total_gap / valid_samples)
    else:
        metrics['Include']['Inter%'] = 0.0
        metrics['Include']['Occ%'] = 0.0
        metrics['Include']['Gap'] = 0.0
        
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['grid_search', 'evaluate'], default='grid_search',
                        help="Mode: 'grid_search' to find best params on val, or 'evaluate' to run a single configuration.")
    parser.add_argument('--beta', type=float, default=0.5, help="Beta parameter (for positive feedback)")
    parser.add_argument('--gamma', type=float, default=0.25, help="Gamma parameter (for negative feedback)")
    parser.add_argument('--num_neg', type=int, default=5, help="Number of negative feedback documents")
    parser.add_argument('--test', action='store_true', help="Use test file instead of val file")
    args = parser.parse_args()

    # Load 3A base model
    ckpt_path = os.path.join(BASE_DIR, 'biencoder_3A_freeze.pt')
    if not os.path.exists(ckpt_path):
        print(f"Error: 3A checkpoint {ckpt_path} not found.")
        return
        
    print(f"Loading 3A Model from {ckpt_path}...")
    model = UnifiedBiEncoder(base_dir=BASE_DIR, step2_option='A')
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=False)
    model.to(DEVICE)
    
    # Load Pool
    pool, id_to_global_idx = build_pool(model)
    
    # Load Data
    data_file = TEST_FILE if args.test else VAL_FILE
    data = load_data(data_file)
    
    # Precompute data
    queries_tensor, contexts, targets = precompute_data(model, pool, id_to_global_idx, data)
    
    if args.mode == 'grid_search':
        print("\n=== STARTING GRID SEARCH ON VALIDATION SET ===")
        # Define search spaces
        betas = [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
        gammas = [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
        num_negs = [1, 3, 5, 10]
        
        # Precompute R_vec and N_vec for all U
        R_vecs, N_vecs_dict = get_feedback_vectors(queries_tensor, contexts, pool, num_negs)
        
        results = []
        best_inc_mrr = -1
        best_exc_mrr = -1
        best_inc_config = None
        best_exc_config = None
        
        total_runs = len(betas) * len(gammas) * len(num_negs)
        print(f"Total configurations to evaluate: {total_runs}")
        
        pbar = tqdm(total=total_runs, desc="Grid Search")
        for num_neg in num_negs:
            N_vecs = N_vecs_dict[num_neg]
            for beta in betas:
                for gamma in gammas:
                    # Run evaluation
                    metrics = evaluate_rocchio_config(
                        queries_tensor, R_vecs, N_vecs, targets, contexts, pool,
                        alpha=1.0, beta=beta, gamma=gamma
                    )
                    
                    inc_mrr = metrics['Include']['MRR']
                    exc_mrr = metrics['Exclude']['MRR']
                    
                    config_res = {
                        'beta': beta,
                        'gamma': gamma,
                        'num_neg': num_neg,
                        'inc_MRR': inc_mrr,
                        'inc_R@1': metrics['Include']['R@1'],
                        'inc_R@20': metrics['Include']['R@20'],
                        'inc_R@100': metrics['Include']['R@100'],
                        'exc_MRR': exc_mrr,
                        'exc_R@1': metrics['Exclude']['R@1'],
                        'exc_R@20': metrics['Exclude']['R@20'],
                        'exc_R@100': metrics['Exclude']['R@100'],
                        'inc_Inter%': metrics['Include']['Inter%'],
                        'inc_Occ%': metrics['Include']['Occ%'],
                        'inc_Gap': metrics['Include']['Gap']
                    }
                    results.append(config_res)
                    
                    if inc_mrr > best_inc_mrr:
                        best_inc_mrr = inc_mrr
                        best_inc_config = config_res
                    if exc_mrr > best_exc_mrr:
                        best_exc_mrr = exc_mrr
                        best_exc_config = config_res
                        
                    pbar.update(1)
        pbar.close()
        
        # Save search results to CSV
        csv_file = os.path.join(BASE_DIR, 'rocchio_grid_search_results.csv')
        print(f"Saving grid search results to {csv_file}...")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
        print("\n" + "="*50)
        print("GRID SEARCH COMPLETED!")
        print("="*50)
        print(f"Best Configuration sorted by Include MRR:")
        print(f"  beta: {best_inc_config['beta']}")
        print(f"  gamma: {best_inc_config['gamma']}")
        print(f"  num_neg: {best_inc_config['num_neg']}")
        print(f"  Include MRR: {best_inc_config['inc_MRR']:.2f}% | R@1: {best_inc_config['inc_R@1']:.2f}% | R@20: {best_inc_config['inc_R@20']:.2f}%")
        print(f"  Include Inter%: {best_inc_config['inc_Inter%']:.1f}% | Occ%: {best_inc_config['inc_Occ%']:.1f}% | Gap: {best_inc_config['inc_Gap']:.4f}")
        print("-"*50)
        print(f"Best Configuration sorted by Exclude MRR:")
        print(f"  beta: {best_exc_config['beta']}")
        print(f"  gamma: {best_exc_config['gamma']}")
        print(f"  num_neg: {best_exc_config['num_neg']}")
        print(f"  Exclude MRR: {best_exc_config['exc_MRR']:.2f}% | R@1: {best_exc_config['exc_R@1']:.2f}% | R@20: {best_exc_config['exc_R@20']:.2f}%")
        print("="*50)
        
    else:
        # Single configuration evaluation (e.g. for evaluating on test set using best params)
        print(f"\nEvaluating Single Configuration on {'Test' if args.test else 'Val'} set...")
        print(f"Params: beta={args.beta}, gamma={args.gamma}, num_neg={args.num_neg}")
        
        # Precompute R_vec and N_vec for the single U
        R_vecs, N_vecs_dict = get_feedback_vectors(queries_tensor, contexts, pool, [args.num_neg])
        N_vecs = N_vecs_dict[args.num_neg]
        
        metrics = evaluate_rocchio_config(
            queries_tensor, R_vecs, N_vecs, targets, contexts, pool,
            alpha=1.0, beta=args.beta, gamma=args.gamma
        )
        
        # Format output similar to eval_3ABC_final_check.py
        rows = []
        for setting in ['Include', 'Exclude']:
            res = metrics[setting]
            row = {
                'Model': f"Rocchio(b={args.beta},g={args.gamma},u={args.num_neg})",
                'Setting': setting,
                'Count': len(data),
                'MRR': res['MRR'],
                'R@1': res['R@1'],
                'R@5': res['R@5'],
                'R@20': res['R@20'],
                'R@100': res['R@100'],
                'Inter%': metrics['Include']['Inter%'] if setting == 'Include' else metrics['Include']['Inter%'],
                'Occ%': metrics['Include']['Occ%'] if setting == 'Include' else metrics['Include']['Occ%'],
                'Gap': metrics['Include']['Gap'] if setting == 'Include' else metrics['Include']['Gap']
            }
            rows.append(row)
            
        print("\n" + "="*120)
        header = f"{'Model':<30} | {'Set':<8} | {'Count':<6} | {'R@1':<7} | {'R@20':<7} | {'R@100':<7} | {'MRR':<7} | {'Inter%':<7} | {'Occ%':<7} | {'Gap':<7}"
        print(header)
        print("-" * 120)
        for r in rows:
            row_str = (f"{r['Model']:<30} | {r['Setting']:<8} | {r['Count']:<6} | "
                       f"{r['R@1']:>6.2f}% | {r['R@20']:>6.2f}% | {r['R@100']:>6.2f}% | "
                       f"{r['MRR']:>6.2f}% | {r['Inter%']:>6.1f}% | {r['Occ%']:>6.1f}% | {r['Gap']:>7.4f}")
            print(row_str)
        print("="*120)
        
        # Save to json file
        out_name = f"rocchio_tuned_{'test' if args.test else 'val'}_results.json"
        out_path = os.path.join(BASE_DIR, out_name)
        with open(out_path, 'w') as f:
            json.dump(rows, f, indent=4)
        print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
