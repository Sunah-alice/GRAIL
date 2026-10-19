"""
GRAIL model definition (shared by training and evaluation).

Defines the UnifiedBiEncoder with three query-steering options:
  - Option A : Query-only (no steering)            -> paper baseline "Query Only"
  - Option B : Additive (q + context)              -> paper baseline "Additive"
  - Option C : GRAIL, context-subtractive query steering -> paper method (Eq. 8)

Option C follows the paper (Eq. 8): the refined query is
    q_gap = q - g * proj_{h_ctx}(q)
followed by LayerNorm and L2-normalization.

This module contains the model only. Training lives in train_grail.py and
evaluation scripts import UnifiedBiEncoder from here.
"""

import math
from typing import List

import torch
from torch import nn

from multimodal_encoder import MultimodalEvidenceEncoder
from biencoder import DEVICE, MAX_Q_LEN, MODEL_NAME

TEMPERATURE = 0.05


def collate_fn(batch):
    questions = [b.question for b in batch]
    evidences = [b.evidences for b in batch]
    targets = [b.target for b in batch]
    return {"questions": questions, "evidences": evidences, "targets": targets}


class QuestionEncoder(nn.Module):
    """Transformer question encoder projected into the shared space.

    Unlike the pre-alignment QuestionEncoder in biencoder.py, the backbone here
    is left trainable so that Stage 2 can optionally fine-tune it.
    """

    def __init__(self, model_name, shared_dim):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.model.config.hidden_size, shared_dim)
        self.layer_norm = nn.LayerNorm(shared_dim)

    def forward(self, texts: List[str]):
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=MAX_Q_LEN, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        out = self.model(**enc)
        cls = out.last_hidden_state[:, 0, :]
        z = self.proj(cls)
        z = self.layer_norm(z)
        return torch.nn.functional.normalize(z, p=2, dim=-1)


class UnifiedBiEncoder(nn.Module):
    """Bi-encoder with selectable query-steering (Options A / B / C).

    Option C is the paper's GRAIL method (subtractive query steering, Eq. 8).
    """

    def __init__(self, base_dir, step2_option="C", **kwargs):
        super().__init__()
        self.step2_option = step2_option

        # 1. Base encoders
        self.evidence_encoder = MultimodalEvidenceEncoder(base_dir=base_dir, **kwargs)
        self.shared_dim = self.evidence_encoder.shared_dim
        self.question_encoder = QuestionEncoder(MODEL_NAME, self.shared_dim)

        # 2. Reasoning modules
        self.d_sqrt = math.sqrt(self.shared_dim)
        self.layer_norm_sat = nn.LayerNorm(self.shared_dim)

        if step2_option == "C":
            # (a) adaptive gap gate: g = sigmoid(W_gap[q; h_ctx])   (Eq. 8)
            self.W_gap = nn.Sequential(
                nn.Linear(self.shared_dim * 2, self.shared_dim // 4),
                nn.ReLU(),
                nn.Linear(self.shared_dim // 4, 1),
                nn.Sigmoid()
            )
            self.layer_norm_req = nn.LayerNorm(self.shared_dim)

            # (b) dynamic temperature scaling (Option C only)
            self.temp_net = nn.Sequential(
                nn.Linear(self.shared_dim * 2, self.shared_dim // 8),
                nn.ReLU(),
                nn.Linear(self.shared_dim // 8, 1)
            )

        if step2_option == "B":
            # Additive baseline: cross-attention over context, then q + context
            self.cross_attention = nn.MultiheadAttention(embed_dim=self.shared_dim, num_heads=8, batch_first=True)
            self.layer_norm = nn.LayerNorm(self.shared_dim)

        self.log_temp = nn.Parameter(torch.tensor(math.log(TEMPERATURE)))

    def get_h_sat(self, q_vec, ev_reps):
        """Context aggregation h_ctx via query-weighted attention (Eq. 6-7)."""
        if not ev_reps:
            return torch.zeros_like(q_vec)
        V = torch.stack(ev_reps)
        scores = (V @ q_vec) / self.d_sqrt
        alpha = torch.softmax(scores, dim=0)
        h_sat = (alpha.unsqueeze(-1) * V).sum(dim=0)
        return self.layer_norm_sat(h_sat)

    def steer_query(self, q_i, h_sat_i):
        """Compute the steered query for a single sample given q and h_ctx.

        Shared by training forward() and by evaluation inference so that the
        two never diverge. Returns (final_q, log_temp_for_this_sample).
        """
        current_log_temp = self.log_temp

        if self.step2_option == "A":
            return q_i, current_log_temp

        if self.step2_option == "B":
            # handled by caller (needs the full ev_reps sequence); see forward().
            raise RuntimeError("Option B steering is handled inline in forward().")

        # Option C: GRAIL subtractive query steering
        if h_sat_i.norm() > 1e-6:
            h_sat_norm = torch.nn.functional.normalize(h_sat_i, p=2, dim=-1)
            proj_i = torch.dot(q_i, h_sat_norm) * h_sat_norm
            g_i = self.W_gap(torch.cat([q_i, h_sat_i], dim=-1))
            q_perp_i = q_i - g_i * proj_i
        else:
            q_perp_i = q_i

        h_req_i = self.layer_norm_req(q_perp_i)
        final_q = torch.nn.functional.normalize(h_req_i, p=2, dim=-1)

        # Dynamic temperature from q and h_ctx.
        dt_i = self.temp_net(torch.cat([q_i, h_sat_i], dim=-1)).squeeze(-1)
        current_log_temp = self.log_temp + dt_i
        return final_q, current_log_temp

    def forward(self, questions, evidences_list, targets):
        Q = self.question_encoder(questions)
        final_q_vecs = []
        final_temps = []
        for i, q_i in enumerate(Q):
            current_log_temp = self.log_temp

            if self.step2_option == "A":
                final_q = q_i
            else:
                evs = evidences_list[i]
                ev_reps = [self.evidence_encoder.encode_step(e["modality"], e["id"]) for e in evs]
                h_sat_i = self.get_h_sat(q_i, ev_reps)

                if self.step2_option == "B":
                    if not ev_reps:
                        final_q = q_i
                    else:
                        H_seq = torch.stack(ev_reps).unsqueeze(0)  # (1, N, D)
                        Q_seq = q_i.unsqueeze(0).unsqueeze(1)       # (1, 1, D)
                        attn_output, _ = self.cross_attention(query=Q_seq, key=H_seq, value=H_seq)
                        final_q = torch.nn.functional.normalize(
                            self.layer_norm(q_i + attn_output.squeeze(0).squeeze(0)), p=2, dim=-1
                        )
                else:  # Option C
                    final_q, current_log_temp = self.steer_query(q_i, h_sat_i)

            final_q_vecs.append(final_q)
            final_temps.append(current_log_temp)

        H_q = torch.stack(final_q_vecs)
        H_t = torch.stack([self.evidence_encoder.encode_step(t["modality"], t["id"]) for t in targets])
        H_t = torch.nn.functional.normalize(H_t, p=2, dim=-1)

        # Per-query dynamic temperature for Option C
        if self.step2_option == "C":
            H_temps = torch.stack(final_temps).unsqueeze(-1)  # (B, 1)
            logits = (H_q @ H_t.T) / torch.exp(H_temps)
        else:
            logits = (H_q @ H_t.T) / torch.exp(self.log_temp)

        labels = torch.arange(len(questions), device=DEVICE)
        loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss}
