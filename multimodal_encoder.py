"""
Multimodal evidence encoder for MMQA.

Loads precomputed text / table / image embeddings (.npy) and projects each
modality's base space into a shared space R^D_shared. Given {"modality", "id"}
it returns the corresponding shared-space vector. Defines the projection
layers only; training happens in the separate training scripts.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn


Modality = Literal["text", "table", "image"]


@dataclass
class EmbeddingStore:
    """id -> vector store backed by an (num_items x dim) weight matrix."""

    ids: List[str]
    id_to_index: Dict[str, int]
    weights: torch.Tensor  # (N, D)

    @property
    def dim(self) -> int:
        return int(self.weights.shape[1])

    @property
    def num_items(self) -> int:
        return int(self.weights.shape[0])

    def get_index(self, item_id: str) -> Optional[int]:
        return self.id_to_index.get(item_id)

    @classmethod
    def from_npy(cls, path: str) -> "EmbeddingStore":
        """Load a dict[id -> np.ndarray] .npy (e.g. image embeddings keyed by id)."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file not found: {path}")

        raw = np.load(path, allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.shape != ():
            raise ValueError(
                f"Expected dict-like npy (object array) at {path}, "
                f"but got array with shape {raw.shape}. "
                "Use from_matrix_with_ids for plain matrices."
            )

        raw = raw.item()

        # Sorted id list for a deterministic index assignment.
        ids = sorted(raw.keys())
        vectors = [np.asarray(raw[_id], dtype=np.float32) for _id in ids]

        weights = torch.from_numpy(np.stack(vectors, axis=0))  # (N, D)
        id_to_index = {id_: i for i, id_ in enumerate(ids)}

        return cls(ids=ids, id_to_index=id_to_index, weights=weights)

    @classmethod
    def from_matrix_with_ids(cls, matrix: np.ndarray, id_order: List[str]) -> "EmbeddingStore":
        """matrix: (N, D); id_order: length-N id list aligned to matrix rows."""
        if matrix.shape[0] != len(id_order):
            raise ValueError(
                f"matrix rows ({matrix.shape[0]}) and id_order length "
                f"({len(id_order)}) do not match"
            )
        weights = torch.from_numpy(matrix.astype(np.float32))
        id_to_index = {id_: i for i, id_ in enumerate(id_order)}
        return cls(ids=id_order, id_to_index=id_to_index, weights=weights)


class MultimodalEvidenceEncoder(nn.Module):
    """Maps text / table / image evidence into the shared space R^D_shared.

    Loads precomputed base embeddings:
        text:  embeddings_text.npy        (id -> R^D_text)
        table: embeddings_table.npy       (id -> R^D_table)
        image: mmqa_image_embeddings.npy  (id -> R^D_image)
    and applies a per-modality projection into R^D_shared. shared_dim defaults
    to the text embedding dimension.
    """

    def __init__(
        self,
        base_dir: str,
        shared_dim: Optional[int] = None,
        freeze_base_embeddings: bool = True,
        use_layernorm: bool = True,
        text_mat_name: str = "embeddings_text.npy",
        text_map_name: str = "map_text_ids.json",
        table_mat_name: str = "embeddings_table.npy",
        table_map_name: str = "map_table_ids.json",
        image_mat_name: str = "mmqa_image_embeddings.npy",
    ) -> None:
        """
        Args:
            base_dir: directory holding the embeddings / id maps.
            shared_dim: shared dimension; defaults to the text embedding dim.
            freeze_base_embeddings: if True, base embeddings are not updated.
            use_layernorm: whether to apply per-modality LayerNorm.
        """
        super().__init__()

        self.base_dir = base_dir
        self.use_layernorm = use_layernorm

        # 1) Load base embeddings.
        # text: embeddings_text.npy + map_text_ids.json
        text_mat_path = os.path.join(base_dir, text_mat_name)
        text_map_path = os.path.join(base_dir, text_map_name)
        if not os.path.exists(text_mat_path) or not os.path.exists(text_map_path):
            raise FileNotFoundError(f"Text embedding or mapping not found: {text_mat_path}")
        
        text_matrix = np.load(text_mat_path)
        with open(text_map_path, "r") as f:
            text_id_map = json.load(f)
        # Build id list ordered by index.
        text_ids = [None] * len(text_id_map)
        for _id, idx in text_id_map.items():
            text_ids[idx] = _id
        self.text_store = EmbeddingStore.from_matrix_with_ids(text_matrix, text_ids)

        # table: embeddings_table.npy + map_table_ids.json
        table_mat_path = os.path.join(base_dir, table_mat_name)
        table_map_path = os.path.join(base_dir, table_map_name)
        if not os.path.exists(table_mat_path) or not os.path.exists(table_map_path):
            raise FileNotFoundError(f"Table embedding or mapping not found: {table_mat_path}")
        
        table_matrix = np.load(table_mat_path)
        with open(table_map_path, "r") as f:
            table_id_map = json.load(f)
        # Build id list ordered by index.
        table_ids = [None] * len(table_id_map)
        for _id, idx in table_id_map.items():
            table_ids[idx] = _id
        self.table_store = EmbeddingStore.from_matrix_with_ids(table_matrix, table_ids)

        # image (dict npy)
        image_path = os.path.join(base_dir, image_mat_name)
        self.image_store = EmbeddingStore.from_npy(image_path)

        # 2) Determine the shared dimension.
        if shared_dim is None:
            shared_dim = self.text_store.dim
        self.shared_dim = int(shared_dim)

        # 3) Wrap base embeddings in nn.Embedding (id->index handled by the store).
        self.text_embedding = nn.Embedding(
            num_embeddings=self.text_store.num_items,
            embedding_dim=self.text_store.dim,
        )
        self.table_embedding = nn.Embedding(
            num_embeddings=self.table_store.num_items,
            embedding_dim=self.table_store.dim,
        )
        self.image_embedding = nn.Embedding(
            num_embeddings=self.image_store.num_items,
            embedding_dim=self.image_store.dim,
        )

        # Initialize with the precomputed embeddings.
        with torch.no_grad():
            self.text_embedding.weight.copy_(self.text_store.weights)
            self.table_embedding.weight.copy_(self.table_store.weights)
            self.image_embedding.weight.copy_(self.image_store.weights)

        if freeze_base_embeddings:
            for p in self.text_embedding.parameters():
                p.requires_grad = False
            for p in self.table_embedding.parameters():
                p.requires_grad = False
            for p in self.image_embedding.parameters():
                p.requires_grad = False

        # 4) Per-modality projection (base_dim -> shared_dim).
        self.text_proj = nn.Linear(self.text_store.dim, self.shared_dim)
        self.table_proj = nn.Linear(self.table_store.dim, self.shared_dim)
        self.image_proj = nn.Linear(self.image_store.dim, self.shared_dim)

        # Optional per-modality LayerNorm.
        self.layer_norm_text = nn.LayerNorm(self.shared_dim)
        self.layer_norm_table = nn.LayerNorm(self.shared_dim)
        self.layer_norm_image = nn.LayerNorm(self.shared_dim)

    # ----------- internal helpers -----------

    def _lookup_index(self, modality: Modality, item_id: str) -> Optional[int]:
        if modality == "text":
            return self.text_store.get_index(item_id)
        if modality == "table":
            return self.table_store.get_index(item_id)
        if modality == "image":
            return self.image_store.get_index(item_id)
        raise ValueError(f"Unknown modality: {modality}")

    def _encode_indices(
        self, modality: Modality, indices: torch.LongTensor
    ) -> torch.Tensor:
        """indices: (B,) -> (B, D_shared)."""
        if modality == "text":
            base = self.text_embedding(indices)  # (B, D_text)
            out = self.text_proj(base)
            if self.use_layernorm:
                out = self.layer_norm_text(out)
        elif modality == "table":
            base = self.table_embedding(indices)
            out = self.table_proj(base)
            if self.use_layernorm:
                out = self.layer_norm_table(out)
        elif modality == "image":
            base = self.image_embedding(indices)
            out = self.image_proj(base)
            if self.use_layernorm:
                out = self.layer_norm_image(out)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # L2 normalize for retrieval-friendly representation
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out

    # ----------- public API -----------

    def encode_step(self, modality: Modality, item_id: str | int) -> torch.Tensor:
        """Encode a single evidence (modality, id) into an R^D_shared vector."""
        if isinstance(item_id, int):
            idx: Optional[int] = item_id
        elif isinstance(item_id, str) and item_id.isdigit() and modality in ["text", "table"]:
            # A digit string like "123" is treated as a direct index.
            idx = int(item_id)
        else:
            idx = self._lookup_index(modality, str(item_id))
            
        if idx is None or idx < 0:
            raise KeyError(f"{modality} id not found in embedding store: {item_id}")
        
        # Range check.
        store = getattr(self, f"{modality}_store")
        if idx >= store.num_items:
            raise IndexError(f"{modality} index {idx} out of range (max {store.num_items-1})")

        device = next(self.parameters()).device
        indices = torch.tensor([idx], dtype=torch.long, device=device)
        emb = self._encode_indices(modality, indices)  # (1, D)
        return emb[0]  # (D,)

    def encode_steps(
        self, steps: List[Dict[str, str | int]]
    ) -> Tuple[torch.Tensor, List[Tuple[Modality, str]]]:
        """
        Encode multiple evidence steps at once (batched).
        """
        device = next(self.parameters()).device
        
        # 1. Group indices by modality
        modality_to_indices: Dict[Modality, List[int]] = {"text": [], "table": [], "image": []}
        step_meta: List[Tuple[Modality, str, int]] = [] # (modality, id_str, original_position)
        
        for i, step in enumerate(steps):
            modality = step["modality"]
            _id_raw = step["id"]
            
            if isinstance(_id_raw, int):
                item_id = str(_id_raw)
                idx = _id_raw
            else:
                item_id = str(_id_raw)
                idx = self._lookup_index(modality, item_id)
            
            if idx is not None:
                modality_to_indices[modality].append(idx)
                step_meta.append((modality, item_id, i))

        if not step_meta:
            return torch.empty(0, self.shared_dim, device=device), []

        # 2. Batch encode each modality
        final_embeddings = torch.zeros((len(step_meta), self.shared_dim), device=device)
        current_idx = 0
        
        results_meta: List[Tuple[Modality, str]] = []
        
        # We need to maintain the original order of 'steps' for some use cases, 
        # but here we can just return them in grouped order if we update meta accordingly,
        # or we can put them back in original order.
        # Let's keep them in original order of input 'steps' that had valid embeddings.
        
        # To make it truly fast, we should batch process modality by modality.
        modality_embs: Dict[Modality, torch.Tensor] = {}
        for mod, indices in modality_to_indices.items():
            if indices:
                idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                modality_embs[mod] = self._encode_indices(mod, idx_tensor) # (N_mod, D)
        
        # Reconstruct in order
        modality_counters = {"text": 0, "table": 0, "image": 0}
        for mod, item_id, _ in step_meta:
            m_idx = modality_counters[mod]
            final_embeddings[current_idx] = modality_embs[mod][m_idx]
            results_meta.append((mod, item_id))
            modality_counters[mod] += 1
            current_idx += 1

        return final_embeddings, results_meta

    def pool_steps(
        self, steps: List[Dict[str, str | int]], pool: str = "mean"
    ) -> torch.Tensor:
        """
        Pool multiple evidence steps into a single set representation (mean pooling).
        """
        embs, _ = self.encode_steps(steps)
        if embs.numel() == 0:
            device = next(self.parameters()).device
            return torch.zeros(self.shared_dim, device=device)

        if pool == "mean":
            return embs.mean(dim=0)
        else:
            raise ValueError(f"Unsupported pooling type: {pool}")


if __name__ == "__main__":
    """
    Quick smoke test:

    python multimodal_encoder.py
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    encoder = MultimodalEvidenceEncoder(base_dir=base_dir, shared_dim=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    # Example: encode one text / table / image item each.
    # Real ids come from map_text_ids.json / map_table_ids.json / mmqa_image_embeddings.npy.
    # Here we just use the first id.
    sample_text_id = encoder.text_store.ids[0]
    sample_table_id = encoder.table_store.ids[0]
    sample_image_id = encoder.image_store.ids[0]

    with torch.no_grad():
        t = encoder.encode_step("text", sample_text_id)
        s = encoder.encode_step("table", sample_table_id)
        i = encoder.encode_step("image", sample_image_id)
        print("text emb:", t.shape)
        print("table emb:", s.shape)
        print("image emb:", i.shape)


