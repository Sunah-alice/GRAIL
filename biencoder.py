"""
Shared bi-encoder library for GRAIL.

Provides the common constants, dataset, question encoder, and the Query-only
bi-encoder (BiEncoder) used as the retrieval backbone / baseline (paper
"Query Only"). GRAIL's steering model (Options A/B/C) lives in grail_model.py.

Dataset format (one JSON object per line):
  {
    "qid": str,
    "question": str,
    "answer": [str, ...],
    "target":    {"modality": "text|table|image", "id": ...},
    "evidences": [{"modality": "...", "id": ...}, ...]
  }

BiEncoder:
  - Question encoder: BGE-Large backbone (frozen) + projection -> R^D_shared
  - Evidence encoder: MultimodalEvidenceEncoder (text/table/image -> R^D_shared)
  - Attention pooling over evidences with the question as query
  - Set representation: H_set = LayerNorm(context_pooled + question_emb)

Imported as a library; not meant to be run directly.
"""

import json
import math
import os
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel

from multimodal_encoder import MultimodalEvidenceEncoder


# Data/checkpoint directory. Defaults to this file's folder; override with MMQA_DATA_DIR.
BASE_DIR = os.environ.get("MMQA_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))

# Text / question encoder backbone (same family as the text evidence encoder).
MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32
MAX_Q_LEN = 64
SHARED_DIM = None  # None => use text embedding dim from MultimodalEvidenceEncoder
RUN_TAG = ""  # Optional tag to distinguish different runs (e.g., 'proj20', 'proj50')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------- Dataset -----------------


@dataclass
class MaskedSample:
    qid: str
    question: str
    target: Dict[str, Any]
    evidences: List[Dict[str, Any]]


class MaskedChainDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.samples: List[MaskedSample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                qid = item["qid"]
                question = item.get("question") or ""
                target = item["target"]
                evidences = item["evidences"]
                # Skip samples with no evidence.
                if not evidences:
                    continue
                self.samples.append(
                    MaskedSample(
                        qid=qid, question=question, target=target, evidences=evidences
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MaskedSample:
        return self.samples[idx]


# ----------------- Question Encoder -----------------


class QuestionEncoder(nn.Module):
    """
    Embed the question with a pretrained Transformer and project to the shared space.
    """

    def __init__(self, model_name: str, shared_dim: int) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size
        self.proj = nn.Linear(hidden_size, shared_dim)
        self.layer_norm = nn.LayerNorm(shared_dim)

        # Backbone frozen; only the projection is trained.
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        texts: list of strings
        return: (B, D_shared)
        """
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_Q_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
            # Use the CLS token.
            cls = out.last_hidden_state[:, 0, :]  # (B, H)

        z = self.proj(cls)
        z = self.layer_norm(z)
        z = torch.nn.functional.normalize(z, p=2, dim=-1)
        return z


# ----------------- Bi-encoder Model -----------------


class BiEncoder(nn.Module):
    def __init__(self, base_dir: str, shared_dim: Optional[int] = None, **kwargs) -> None:
        super().__init__()

        # evidence encoder
        self.evidence_encoder = MultimodalEvidenceEncoder(
            base_dir=base_dir, shared_dim=shared_dim, **kwargs
        )
        self.shared_dim = self.evidence_encoder.shared_dim

        # question encoder
        self.question_encoder = QuestionEncoder(MODEL_NAME, self.shared_dim)

        # attention pooling parameters (single-head)
        self.attention_scale = 1.0 / math.sqrt(self.shared_dim)

        # Normalize the final set representation.
        self.layer_norm_set = nn.LayerNorm(self.shared_dim)

        # learnable temperature (for InfoNCE)
        self.log_temp = nn.Parameter(torch.tensor(0.0))  # temp = exp(log_temp)

    def encode_context(
        self, batch_questions: List[str], batch_evidences: List[List[Dict[str, Any]]]
    ) -> torch.Tensor:
        """
        batch_questions: list of B strings; batch_evidences: list of B step lists.
        return: (B, D_shared) H_set
        """
        device = next(self.parameters()).device

        # 1) question embeddings
        q_emb = self.question_encoder(batch_questions).to(device)  # (B, D)

        # 2) evidence embeddings + attention pooling (per sample)
        ctx_embs: List[torch.Tensor] = []
        for evidences, q in zip(batch_evidences, q_emb):
            ev_embs, _ = self.evidence_encoder.encode_steps(evidences)
            if ev_embs.numel() == 0:
                # Zero vector when there is no evidence.
                ctx = torch.zeros(self.shared_dim, device=device)
            else:
                # Attention with q as query, evidences as key/value.
                # score_i = (q · e_i) / sqrt(D)
                scores = (ev_embs @ q) * self.attention_scale  # (N,)
                weights = torch.softmax(scores, dim=0)  # (N,)
                ctx = (weights.unsqueeze(-1) * ev_embs).sum(dim=0)  # (D,)

            # Combine question + context into the set representation.
            h = self.layer_norm_set(ctx + q)
            h = torch.nn.functional.normalize(h, p=2, dim=-1)
            ctx_embs.append(h)

        H_set = torch.stack(ctx_embs, dim=0)  # (B, D)
        return H_set

    def encode_targets(self, batch_targets: List[Dict[str, Any]]) -> torch.Tensor:
        """
        batch_targets: [{"modality": "...", "id": ...}, ...]
        return: (B, D_shared)
        """
        device = next(self.parameters()).device
        embs: List[torch.Tensor] = []
        for tgt in batch_targets:
            modality = tgt["modality"]
            _id = tgt["id"]
            # id may be int or str.
            emb = self.evidence_encoder.encode_step(modality, str(_id)).to(device)
            embs.append(emb)
        H_tgt = torch.stack(embs, dim=0)
        return H_tgt

    def forward(
        self,
        batch_questions: List[str],
        batch_evidences: List[List[Dict[str, Any]]],
        batch_targets: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        H_set = self.encode_context(batch_questions, batch_evidences)  # (B, D)
        H_tgt = self.encode_targets(batch_targets)  # (B, D)

        # similarity matrix
        logits = H_set @ H_tgt.T  # (B, B)
        temp = torch.exp(self.log_temp)
        logits = logits / temp

        return {"logits": logits}
