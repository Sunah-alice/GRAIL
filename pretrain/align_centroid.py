import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multimodal_encoder import MultimodalEvidenceEncoder
from biencoder import BASE_DIR, DEVICE, BATCH_SIZE

# --- Configuration for Neutral Alignment ---
MASKED_FILE = os.path.join(BASE_DIR, "mmqa_masked_renewal_train.jsonl")
NUM_EPOCHS = 100
PRETRAIN_LR = 1e-3
TEMPERATURE = 0.05
QUEUE_SIZE = 4096

@dataclass
class NeutralSample:
    evidences: List[Dict[str, Any]]
    target: Dict[str, Any]

class NeutralDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.samples: List[NeutralSample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                evs = item.get("evidences", [])
                target = item.get("target")  # target required
                
                # Require a target; a target-only chain is allowed.


                if not target: continue
                # if len(evs) < 1: continue  # (optional) require >=1 evidence
                
                self.samples.append(NeutralSample(evidences=evs, target=target))
    
    def __len__(self) -> int: return len(self.samples)
    def __getitem__(self, idx: int) -> NeutralSample: return self.samples[idx]

def collate_neutral(batch: List[NeutralSample]) -> Dict[str, Any]:
    return {
        "evidences": [b.evidences for b in batch],
        "targets": [b.target for b in batch]
    }

class NeutralAlignmentModel(nn.Module):
    def __init__(self, base_dir: str) -> None:
        super().__init__()
        self.evidence_encoder = MultimodalEvidenceEncoder(base_dir=base_dir)
        self.shared_dim = self.evidence_encoder.shared_dim

    def forward(self, batch: Dict[str, Any]):
        evidences_list = batch["evidences"]
        targets = batch["targets"]
        
        # [OPTIMIZED] Encode Targets Batch
        target_reps, _ = self.evidence_encoder.encode_steps(targets)
        
        flat_all_items = [] # Evidences + Target
        batch_indices = []
        chain_lengths = []
        
        # Flatten Evidences
        flat_evidences = []
        for evs in evidences_list:
            flat_evidences.extend(evs)
            
        # Batch encode all evidences
        flat_ev_reps = torch.empty(0, self.shared_dim, device=target_reps.device)
        if flat_evidences:
            flat_ev_reps, _ = self.evidence_encoder.encode_steps(flat_evidences)
            
        # Re-assemble into per-chain structure
        current_ev_idx = 0
        all_reps_list = []
        
        for i, evs in enumerate(evidences_list):
            num_ev = len(evs)
            
            # 1. Get Evidence Reps
            if num_ev > 0:
                ev_chunk = flat_ev_reps[current_ev_idx : current_ev_idx + num_ev]
                current_ev_idx += num_ev
                
                # 2. Append Target Rep
                # (num_ev, D) + (1, D) -> (num_ev + 1, D)
                combined = torch.cat([ev_chunk, target_reps[i].unsqueeze(0)], dim=0)
            else:
                # Only Target
                combined = target_reps[i].unsqueeze(0)
                
            all_reps_list.append(combined)
            batch_indices.extend([i] * combined.size(0))
            chain_lengths.append(combined.size(0))
            
        flat_tensor = torch.cat(all_reps_list, dim=0) # (Total_N, D)
        
        return flat_tensor, batch_indices, chain_lengths

def train_neutral():
    print("Loading Neutral Alignment Dataset (Target Included)...")
    dataset = NeutralDataset(MASKED_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_neutral)
    
    model = NeutralAlignmentModel(base_dir=BASE_DIR)
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR)
    centroid_queue = torch.empty(0, model.shared_dim, device=DEVICE)
    
    print("Starting Neutral Projection Alignment (Centering Logic with Target)...")
    
    # Accumulation settings
    ACCUM_STEPS = 4
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        # optimizer.zero_grad() # [FIX] Remove this to allow accumulation (Handled inside loop)
        
        pbar = tqdm(dataloader, desc=f"Neutral Epoch {epoch}")
        
        for step, batch in enumerate(pbar):
            # 1. Encode (Total_N, D) - Includes Evidences AND Targets
            flat_tensor, batch_indices, chain_lengths = model(batch)
            batch_indices_tensor = torch.tensor(batch_indices, device=DEVICE)
            
            # 2. Compute Centroids (Mean of Evidences + Target)
            centroids = []
            start = 0
            for length in chain_lengths:
                centroids.append(flat_tensor[start : start+length].mean(dim=0))
                start += length
            centroid_batch = torch.stack(centroids) # (B, D)
            
            # 3. MSE Loss: Pull each item (Ev or Tgt) to its chain centroid
            target_centroids = centroid_batch[batch_indices_tensor]
            loss_mse = nn.functional.mse_loss(flat_tensor, target_centroids)
            
            # 4. Contrastive Loss
            flat_norm = torch.nn.functional.normalize(flat_tensor, p=2, dim=-1)
            centroid_norm = torch.nn.functional.normalize(centroid_batch, p=2, dim=-1)

            logits = (flat_norm @ centroid_norm.T) / TEMPERATURE
            
            if centroid_queue.numel() > 0:
                queue_logits = (flat_norm @ centroid_queue.T) / TEMPERATURE
                logits = torch.cat([logits, queue_logits], dim=1)
            
            loss_cont = nn.functional.cross_entropy(logits, batch_indices_tensor)
            
            loss = 0.1 * loss_mse + 0.9 * loss_cont
            loss = loss / ACCUM_STEPS
            loss.backward()
            
            # [FIX] Update on every ACCUM_STEPS OR at the end of epoch
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
                
                with torch.no_grad():
                    centroid_queue = torch.cat([centroid_queue, centroid_norm.detach()], dim=0)[-QUEUE_SIZE:]
            
            total_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix({"loss": f"{loss.item() * ACCUM_STEPS:.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
        
    ckpt_path = os.path.join(BASE_DIR, "alignment_centroid.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved Centroid alignment checkpoint: {ckpt_path}")

if __name__ == "__main__":
    train_neutral()