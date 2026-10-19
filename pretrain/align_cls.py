import json
import os
import random
from typing import Any, Dict, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multimodal_encoder import MultimodalEvidenceEncoder
from biencoder import BASE_DIR, DEVICE, BATCH_SIZE

# --- Configuration ---
MASKED_FILE = os.path.join(BASE_DIR, "mmqa_masked_renewal_train.jsonl")
NUM_EPOCHS = 100
PRETRAIN_LR = 1e-4
TEMPERATURE = 0.05
QUEUE_SIZE = 4096

class ContextNeutralDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                evs = item.get("evidences", [])
                target = item.get("target")
                
                # Require a target and at least one evidence item.
                if not target: continue
                if len(evs) < 1: continue 
                
                self.samples.append({"evidences": evs, "target": target})
    
    def __len__(self) -> int: return len(self.samples)
    def __getitem__(self, idx: int) -> Dict[str, Any]: return self.samples[idx]

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "evidences": [b["evidences"] for b in batch],
        "targets": [b["target"] for b in batch]
    }

class ContextNeutralAlignment(nn.Module):
    def __init__(self, base_dir: str, shared_dim: int | None = None) -> None:
        super().__init__()
        self.evidence_encoder = MultimodalEvidenceEncoder(base_dir=base_dir, shared_dim=shared_dim)
        self.shared_dim = self.evidence_encoder.shared_dim
        
        # Transformer Special Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.shared_dim))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, self.shared_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.sep_token, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.shared_dim, 
            nhead=8, 
            dim_feedforward=self.shared_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        evidences_list = batch["evidences"]
        targets = batch["targets"]
        
        device = next(self.parameters()).device
        batch_size = len(evidences_list)
        
        # 1. Encode all evidences + targets
        target_reps, _ = self.evidence_encoder.encode_steps(targets)
        
        all_reps_per_batch = [] # List of (N_seq, D) tensors
        flat_reps = [] # For loss calculation (Evidences + Target)
        batch_indices = []
        
        # Build sequences: [CLS] E1 E2 ... [SEP] T
        for i, evs in enumerate(evidences_list):
            curr_seq = []
            
            # Evidences
            if evs:
                ev_embs, _ = self.evidence_encoder.encode_steps(evs)
                curr_seq.append(ev_embs)
                flat_reps.append(ev_embs)
                batch_indices.extend([i] * ev_embs.size(0))
            
            # [SEP] (This is handled in batch construction below, but conceptually here)
            
            # Target
            tgt_unsqueezed = target_reps[i].unsqueeze(0)
            curr_seq.append(tgt_unsqueezed)
            flat_reps.append(tgt_unsqueezed)
            batch_indices.extend([i]) # Target index
            
            all_reps_per_batch.append(curr_seq) # List of tensors for this batch item
            
        if not flat_reps: return {}
        
        # 2. Construct Batched Transformer Input with [SEP]
        # Max sequence length calculation: 1 (CLS) + Max_Ev + 1 (SEP) + 1 (Target)
        # Note: curr_seq contains [Evs..., Target] tensors.
        max_ev_count = 0
        for seq_list in all_reps_per_batch:
            # seq_list[0] might be Evidences (multi-row), seq_list[-1] is Target (1 row)
            # If Evidences exist, len is seq_list[0].size(0)
            ev_cnt = seq_list[0].size(0) if len(seq_list) > 1 else 0
            if ev_cnt > max_ev_count:
                max_ev_count = ev_cnt
                
        seq_len = 1 + max_ev_count + 1 + 1 # [CLS] ...Evs... [SEP] [Target]
        
        x = torch.zeros(batch_size, seq_len, self.shared_dim, device=device)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device) # True = Ignored/Padding
        
        x[:, 0, :] = self.cls_token
        mask[:, 0] = False
        
        # Fill Batch
        for i, seq_list in enumerate(all_reps_per_batch):
            # seq_list: [Ev_tensor, Tgt_tensor] or just [Tgt_tensor] if no evs
            current_pos = 1
            
            # Evidences
            if len(seq_list) > 1: # Has evidences
                ev_tensor = seq_list[0]
                n_ev = ev_tensor.size(0)
                x[i, current_pos : current_pos+n_ev, :] = ev_tensor
                mask[i, current_pos : current_pos+n_ev] = False
                current_pos += n_ev
            
            # SEP Token
            x[i, current_pos, :] = self.sep_token
            mask[i, current_pos] = False
            current_pos += 1
            
            # Target
            tgt_tensor = seq_list[-1]
            x[i, current_pos, :] = tgt_tensor
            mask[i, current_pos] = False
            # current_pos += 1
            
        # Transformer forward
        out = self.transformer(x, src_key_padding_mask=mask)
        H_cls = out[:, 0, :] # (B, D) - Centroid
        
        flat_tensor = torch.cat(flat_reps, dim=0) # (Total_N_Items, D)
        
        return {
            "H_cls": H_cls, 
            "flat_tensor": flat_tensor, 
            "batch_indices": batch_indices
        }

def train():
    print("Starting Option 2: Context Neutral Alignment Training (Transformer-Centering, Optimized)...")
    print(" >>> Enhanced: Input Format [CLS] Evidences ... [SEP] Target")
    
    dataset = ContextNeutralDataset(MASKED_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    model = ContextNeutralAlignment(base_dir=BASE_DIR)
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR)
    
    # Memory Bank
    centroid_queue = torch.empty(0, model.shared_dim, device=DEVICE)
    
    # Gradient Accumulation
    ACCUM_STEPS = 4
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(pbar):
            out = model(batch)
            if not out: continue
            
            H_cls = out["H_cls"]
            flat_tensor = out["flat_tensor"]
            batch_indices = torch.tensor(out["batch_indices"], device=DEVICE)
            
            # 1. MSE Loss: Pull items to centroid
            target_centroids = H_cls[batch_indices]
            loss_mse = nn.functional.mse_loss(flat_tensor, target_centroids)
            
            # 2. Contrastive Loss
            flat_norm = torch.nn.functional.normalize(flat_tensor, p=2, dim=-1)
            H_cls_norm = torch.nn.functional.normalize(H_cls, p=2, dim=-1)
            
            logits = (flat_norm @ H_cls_norm.T) / TEMPERATURE
            
            if centroid_queue.numel() > 0:
                queue_logits = (flat_norm @ centroid_queue.T) / TEMPERATURE
                logits = torch.cat([logits, queue_logits], dim=1)
            
            loss_cont = nn.functional.cross_entropy(logits, batch_indices)
            
            loss = 0.1 * loss_mse + 0.9 * loss_cont
            loss = loss / ACCUM_STEPS
            loss.backward()
            
            if (step + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                with torch.no_grad():
                    centroid_queue = torch.cat([centroid_queue, H_cls_norm.detach()], dim=0)[-QUEUE_SIZE:]
            
            total_loss += loss.item() * ACCUM_STEPS  # undo gradient-accum scaling
            pbar.set_postfix({"loss": f"{loss.item() * ACCUM_STEPS:.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
        
    ckpt_path = os.path.join(BASE_DIR, "alignment_cls.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved CLS alignment checkpoint: {ckpt_path}")

if __name__ == "__main__":
    train()
