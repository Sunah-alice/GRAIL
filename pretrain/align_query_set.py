import json
import os
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multimodal_encoder import MultimodalEvidenceEncoder
from biencoder import BASE_DIR, DEVICE, BATCH_SIZE, QuestionEncoder, MODEL_NAME

# --- Configuration ---
MASKED_FILE = os.path.join(BASE_DIR, "mmqa_masked_renewal_train.jsonl")
NUM_EPOCHS = 100
PRETRAIN_LR = 1e-4
TEMPERATURE = 0.05

class ContextQueryDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.samples = []
        with open(path, encoding="utf-8") as f:
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

class ContextQueryAlignment(nn.Module):
    def __init__(self, base_dir: str, shared_dim: Optional[int] = None) -> None:
        super().__init__()
        self.evidence_encoder = MultimodalEvidenceEncoder(base_dir=base_dir, shared_dim=shared_dim)
        self.shared_dim = self.evidence_encoder.shared_dim
        self.question_encoder = QuestionEncoder(MODEL_NAME, self.shared_dim)
        
        self.sep_token = nn.Parameter(torch.randn(1, 1, self.shared_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.shared_dim, 
            nhead=8, 
            dim_feedforward=self.shared_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        questions = batch["questions"]
        evidences_list = batch["evidences"]
        targets = batch["targets"]
        
        q_reps = self.question_encoder(questions) 
        
        flat_query_reps = []
        flat_ev_reps = []
        chain_indices = []
        
        # Pre-encode Targets (to align with q_out as positive)
        target_reps_list, _ = self.evidence_encoder.encode_steps(targets) # (B, D)
        
        for i, evs in enumerate(evidences_list):
            # 1. Encode ONLY context evidences for Transformer input
            if evs:
                ev_embs, _ = self.evidence_encoder.encode_steps(evs) # (N, D)
            else:
                ev_embs = torch.empty(0, self.shared_dim, device=DEVICE)
                
            # 2. Prepare Transformer Input: [query] [SEP] e1 [SEP] e2 ... (No Target here)
            tokens = [q_reps[i:i+1].unsqueeze(0)]
            for j in range(len(ev_embs)):
                tokens.append(self.sep_token)
                tokens.append(ev_embs[j:j+1].unsqueeze(0))
            
            x = torch.cat(tokens, dim=1)
            out = self.transformer(x)
            q_out = out[:, 0, :] # Output at query position (1, D)
            
            # 3. Align q_out with ALL elements of its chain: (E1, E2, ... Target)
            # This makes the "Alignment Targets" identical to v1 (Option 4)
            target_rep = target_reps_list[i:i+1]
            all_positives = torch.cat([ev_embs, target_rep], dim=0) # (N+1, D)
            
            for j in range(len(all_positives)):
                flat_query_reps.append(q_out)
                flat_ev_reps.append(all_positives[j:j+1])
                chain_indices.append(i)
            
        H_q = torch.cat(flat_query_reps, dim=0) 
        H_ev = torch.cat(flat_ev_reps, dim=0)   
        
        H_q = torch.nn.functional.normalize(H_q, p=2, dim=-1)
        H_ev = torch.nn.functional.normalize(H_ev, p=2, dim=-1)
        
        # 4. InfoNCE with same-chain mask (Identical to v1)
        logits = (H_q @ H_ev.T) / TEMPERATURE
        
        mask = torch.eye(len(H_q), device=H_q.device)
        indices_tensor = torch.tensor(chain_indices, device=H_q.device)
        same_chain_mask = (indices_tensor.unsqueeze(0) == indices_tensor.unsqueeze(1)).float()
        
        # Mask same-chain items to only push against other chains (questions)
        logits = logits - 1e9 * (same_chain_mask - mask)
        
        return {"logits": logits}

def train():
    print("Starting Context Query Alignment Training (v2: Target Excluded From Input, Included In Alignment)...")
    dataset = ContextQueryDataset(MASKED_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    model = ContextQueryAlignment(base_dir=BASE_DIR)
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"v2 Epoch {epoch}")
        
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
        
        if epoch == NUM_EPOCHS:
            ckpt_path = os.path.join(BASE_DIR, "alignment_query_set.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved Query-set alignment checkpoint: {ckpt_path}")

if __name__ == "__main__":
    train()
