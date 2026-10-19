"""
Stage 2 training: GRAIL gap-aware retrieval (paper Section 4.2).

Trains a UnifiedBiEncoder on the evidence-set-completion objective with
in-batch InfoNCE. The query-steering option is selected with --s2:
  A = Query-only, B = Additive, C = GRAIL (context-subtractive query steering)

Stage 1 (alignment) initialization is selected with --s1:
  0 = none (from scratch)
  1 = Centroid        (alignment_centroid.pt)
  2 = CLS             (alignment_cls.pt)
  3 = Query-evidence  (alignment_query_evidence.pt)   [DEFAULT]
  5 = Query-set       (alignment_query_set.pt)

Default paper configuration: --s1 3 --s2 C --freeze
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from biencoder import BASE_DIR, DEVICE, MaskedChainDataset
from grail_model import UnifiedBiEncoder, collate_fn

# Checkpoint naming: biencoder_{S1}{S2}_{freeze|full}[_dy].pt  (100 epochs, fixed)
#   S1 (Stage-1 alignment): 0=none 1=Centroid 2=CLS 3=Query-evidence(default) 5=Query-set
#   S2 (query steering):    A=Query-only  B=Additive  C=GRAIL(gap-aware)
#   _dy = dynamic temperature (Option C only)
STEP1_MAPPING = {
    0: None,
    1: "alignment_centroid",        # Centroid
    2: "alignment_cls",             # CLS
    3: "alignment_query_evidence",  # Query-evidence (default)
    5: "alignment_query_set",       # Query-set
}


def train_grail(args):
    run_id = f"{args.s1}{args.s2}"
    print(f"Starting Stage 2 (GRAIL) | Option: {args.s2} | Init: Stage1 Option {args.s1}")

    model = UnifiedBiEncoder(BASE_DIR, step2_option=args.s2)
    model.to(DEVICE)

    # Load Stage 1 alignment checkpoint if requested.
    if args.s1 > 0:
        fname = STEP1_MAPPING[args.s1]
        load_path = os.path.join(BASE_DIR, f"{fname}.pt")
        if os.path.exists(load_path):
            print(f"Loading Stage 1 checkpoint: {load_path}")
            state = torch.load(load_path, map_location=DEVICE)
            model.load_state_dict(state, strict=False)
        else:
            print(f"Warning: Stage 1 checkpoint not found: {load_path} (training from scratch)")

    # Fine-tuning strategy: optionally freeze encoders, always train reasoning heads.
    encoder_params = []
    head_params = []
    for n, p in model.named_parameters():
        if "question_encoder" in n or "evidence_encoder" in n:
            if args.freeze:
                p.requires_grad = False
            else:
                p.requires_grad = True
                encoder_params.append(p)
        else:
            p.requires_grad = True
            head_params.append(p)

    if args.freeze:
        print("Encoder freeze enabled: only reasoning heads are trained.")
        optimizer = torch.optim.AdamW(head_params, lr=args.lr)
    else:
        print("Full fine-tuning enabled (differential LR).")
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": 2e-5},
            {"params": head_params, "lr": args.lr},
        ])

    data_file = args.data_file if args.data_file else os.path.join(BASE_DIR, "mmqa_masked_renewal_train.jsonl")
    print(f"Training data: {data_file}")
    dataset = MaskedChainDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Run {run_id} Ep{epoch}")
        for batch in pbar:
            optimizer.zero_grad()
            out = model(batch["questions"], batch["evidences"], batch["targets"])
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch}/{args.epochs}] train loss: {avg_loss:.4f}")

    # Save final checkpoint: biencoder_{S1}{S2}_{freeze|full}[_dy].pt
    suffix = "_freeze" if args.freeze else "_full"
    dy_tag = "_dy" if args.s2 == "C" else ""
    save_name = f"biencoder_{run_id}{suffix}{dy_tag}.pt"
    torch.save(model.state_dict(), os.path.join(BASE_DIR, save_name))
    print(f"Saved final checkpoint: {save_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Core options
    parser.add_argument("--s1", type=int, required=True, help="Stage 1 alignment init (0,1,2,3,5)")
    parser.add_argument("--s2", type=str, required=True, choices=["A", "B", "C"],
                        help="Query steering: A=Query-only, B=Additive, C=GRAIL")
    parser.add_argument("--freeze", action="store_true", help="Freeze pre-aligned encoders")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)

    # Data
    parser.add_argument("--data_file", type=str, default=None, help="Custom training data file")

    args = parser.parse_args()
    train_grail(args)
