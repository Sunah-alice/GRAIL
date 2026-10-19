import json
import os
import argparse
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multimodal_encoder import MultimodalEvidenceEncoder
from biencoder import BASE_DIR, DEVICE, BATCH_SIZE, QuestionEncoder, MODEL_NAME

# --- Configuration (Defaults) ---
NUM_EPOCHS = 100
PRETRAIN_LR = 1e-4
TEMPERATURE = 0.05

class LocalQueryDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.samples = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                q = item.get("question", "")
                evs = item.get("evidences", [])
                target = item.get("target")
                if not q or not target: continue
                self.samples.append({"question": q, "evidences": evs, "target": target})
    
    def __len__(self) -> int: return len(self.samples)
    def __getitem__(self, idx: int) -> Dict[str, Any]: return self.samples[idx]

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "questions": [b["question"] for b in batch],
        "evidences": [b["evidences"] for b in batch],
        "targets": [b["target"] for b in batch]
    }

class LocalQueryAlignment(nn.Module):
    def __init__(self, base_dir: str, shared_dim: Optional[int] = None, suffix: str = "") -> None:
        super().__init__()
        # With a suffix, use custom embedding files (e.g. embeddings_text_sfr.npy).
        if suffix:
            text_mat  = f"embeddings_text_{suffix}.npy"
            text_map  = f"map_text_ids_{suffix}.json"
            table_mat = f"embeddings_table_{suffix}.npy"
            table_map = f"map_table_ids_{suffix}.json"
        else:
            text_mat  = "embeddings_text.npy"
            text_map  = "map_text_ids.json"
            table_mat = "embeddings_table.npy"
            table_map = "map_table_ids.json"
        self.evidence_encoder = MultimodalEvidenceEncoder(
            base_dir=base_dir,
            shared_dim=shared_dim,
            text_mat_name=text_mat,
            text_map_name=text_map,
            table_mat_name=table_mat,
            table_map_name=table_map,
        )
        self.shared_dim = self.evidence_encoder.shared_dim
        self.question_encoder = QuestionEncoder(MODEL_NAME, self.shared_dim)
        
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        questions = batch["questions"]
        evidences_list = batch["evidences"] # List[List[Dict]]
        targets = batch["targets"] # List[Dict]
        
        q_reps = self.question_encoder(questions) 
        
        flat_q_reps = []
        flat_ev_reps = []
        chain_indices = [] 
        
        for i, evs in enumerate(evidences_list):
            target = targets[i]
            all_evs = evs + [target]
            ev_embs, _ = self.evidence_encoder.encode_steps(all_evs)
            if ev_embs.numel() == 0:
                continue
            
            flat_ev_reps.append(ev_embs)
            flat_q_reps.append(q_reps[i:i+1].expand(len(ev_embs), -1))
            chain_indices.extend([i] * len(ev_embs))
            
        if not flat_ev_reps:
            return {}
            
        H_q = torch.cat(flat_q_reps, dim=0) 
        H_e = torch.cat(flat_ev_reps, dim=0) 
        
        logits = (H_q @ H_e.T) / TEMPERATURE 
        
        mask = torch.eye(len(H_q), device=H_q.device)
        chain_indices_tensor = torch.tensor(chain_indices, device=H_q.device)
        same_chain_mask = (chain_indices_tensor.unsqueeze(0) == chain_indices_tensor.unsqueeze(1)).float()
        
        logits = logits - 1e9 * (same_chain_mask - mask)
        
        return {"logits": logits}

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="mmqa_masked_renewal_train.jsonl")
    parser.add_argument("--save_name", type=str, default="alignment_query_evidence.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--suffix", type=str, default="",
                        help="Embedding file suffix (e.g. sfr, nv); default uses the BGE-based files.")
    args = parser.parse_args()

    print(f"Starting Local Query Alignment Training | Data: {args.data_file}")
    dataset = LocalQueryDataset(os.path.join(BASE_DIR, args.data_file))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    model = LocalQueryAlignment(base_dir=BASE_DIR, suffix=args.suffix)
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            optimizer.zero_grad()
            out = model(batch)
            if not out: continue
            
            logits = out["logits"]
            labels = torch.arange(len(logits), device=DEVICE)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
        
        if epoch == args.epochs:
            ckpt_path = os.path.join(BASE_DIR, args.save_name)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    train()