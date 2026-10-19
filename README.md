# GRAIL: Gap-Aware Retrieval for Multimodal Multi-Hop QA

Reference implementation for **"Subtraction Gets You More: Gap-Aware Retrieval
for Multimodal Multi-Hop QA" (GRAIL)**.

GRAIL performs implicit, context-subtractive query steering directly in a
shared multimodal embedding space. Given a question and a partial set of
evidence, it rewrites the query to point toward the *missing* evidence by
subtracting the already-covered context — avoiding the "semantic anchoring"
failure mode of additive / LLM-rewriting retrievers.

This repository contains the full, reproducible pipeline: data construction,
embedding, two-stage training (alignment → gap-aware retrieval), the
task-adaptive hybrid selector, and evaluation.

---

## Repository layout

```
.
├── multimodal_encoder.py        # Projects text/table/image embeddings into a shared space
├── biencoder.py                 # Shared constants, dataset, Query-only BiEncoder (library)
├── grail_model.py               # UnifiedBiEncoder with steering options A / B / C
├── run_all.sh                   # Runs the full pipeline end to end
├── data/                        # Build the training/eval data from raw MultimodalQA
│   ├── 00_build_id_maps.py      # id -> index maps for text / table rows
│   ├── 01_build_raw_gold_chains.py
│   ├── 02_enrich_gold_chains.py
│   ├── 03_build_strict_chains.py
│   └── 04_build_splits.py
├── embeddings/                  # Precompute modality embeddings
│   ├── embed_text_table.py      # BGE-Large (text) + TAPAS (table)
│   └── embed_images.py          # CLIP ViT-B/32 (downloads Wikipedia images)
├── pretrain/                    # Stage 1: Unified Semantic Alignment (4 strategies)
│   ├── align_centroid.py        # s1 = 1  (Centroid)
│   ├── align_cls.py             # s1 = 2  (CLS)
│   ├── align_query_evidence.py  # s1 = 3  (Query-evidence)  [default]
│   └── align_query_set.py       # s1 = 5  (Query-set)
├── train/
│   ├── train_grail.py           # Stage 2: gap-aware retrieval (Options A/B/C)
│   └── train_hybrid_selector.py # Stage 3: task-adaptive hybrid selector
└── eval/
    ├── eval_set_completion.py        # Evidence Set Completion: Recall@K, Delta_esc
    ├── eval_sequential_pool.py       # Sequential Pool: step-combination sweep
    ├── eval_sequential_pool_table.py # Sequential Pool: Set-Recall, Rank Jump, NRM
    ├── eval_baseline_rocchio.py      # Rocchio grid search (baseline)
    └── eval_baseline_rocchio_tuned.py# Tuned Rocchio baseline (beta=gamma=0.25, U=10)
```

Library modules live at the repository root; the runnable scripts under
`data/`, `embeddings/`, `pretrain/`, `train/`, `eval/` add the root to
`sys.path` automatically, so run them from the repository root, e.g.
`python train/train_grail.py ...`.

---

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+, `torch==2.9.0` (CUDA), `transformers==4.57.2`.
A single GPU is sufficient.

---

## Data, embeddings, and checkpoints are NOT included

To keep the repository lightweight, **no data, precomputed embeddings, or model
checkpoints are committed** (they range from hundreds of MB to several GB).
Everything is reproducible by running the pipeline below.

If you would like the exact artifacts used in the paper (processed splits,
embeddings, trained checkpoints), they are available on request — contact
**ec_osa16586@snu.ac.kr**.

> Note: all data-construction steps are deterministic (fixed `seed = 42`, no
> LLM calls). Image embedding is the only step that fetches external resources
> (Wikipedia images), so exact numbers may vary slightly with hardware,
> library versions, and image availability.

---

## Getting the raw data

Download the [MultimodalQA](https://github.com/allenai/multimodalqa) benchmark
and place these files in the repository root (or in `$MMQA_DATA_DIR`):

| File | Description |
|------|-------------|
| `MMQA_train.jsonl`, `MMQA_dev.jsonl` | QA items with metadata / supporting context |
| `MMQA_train_*.json` | Per-question-type QA files (e.g. `MMQA_train_Compare_*.json`) |
| `MMQA_texts.jsonl` | Text documents |
| `MMQA_tables.jsonl` | Tables |
| `MMQA_images.jsonl` | Image metadata (title / url / path) |

All scripts read and write artifacts in the repository root by default. To keep
data elsewhere, set `MMQA_DATA_DIR` (every script honors it):

```bash
export MMQA_DATA_DIR=/path/to/data   # optional; defaults to the repo root
```

## Reproduction pipeline

Run the whole pipeline with one command (GPU required; takes hours):

```bash
bash run_all.sh
```

Or run each stage manually, as described below.

### 1. Build the dataset

```bash
python data/00_build_id_maps.py             # -> map_text_ids.json, map_table_ids.json
python data/01_build_raw_gold_chains.py     # -> raw_gold_chains.jsonl
python data/02_enrich_gold_chains.py        # -> enriched_gold_chains.jsonl
python data/03_build_strict_chains.py       # -> mmqa_strict_multi_hop_chains.jsonl
python data/04_build_splits.py              # -> mmqa_masked_renewal_{train,val,test}.jsonl
```

### 2. Precompute embeddings

```bash
python embeddings/embed_text_table.py       # -> embeddings_text.npy, embeddings_table.npy
python embeddings/embed_images.py           # -> mmqa_image_embeddings.npy (downloads images)
```

### 3. Stage 1 — Unified Semantic Alignment

```bash
python pretrain/align_query_evidence.py     # default alignment (s1 = 3)
# other strategies: align_centroid.py (1), align_cls.py (2), align_query_set.py (5)
```

### 4. Stage 2 — Gap-aware retrieval (GRAIL)

```bash
# Default paper configuration (GRAIL, Query-evidence init, encoders frozen):
python train/train_grail.py --s1 3 --s2 C --freeze

# Baselines use the same command with a different steering option:
python train/train_grail.py --s1 3 --s2 A --freeze   # Query-only
python train/train_grail.py --s1 3 --s2 B --freeze   # Additive
```

### 5. Stage 3 — Task-adaptive hybrid selector

Requires the trained Additive (`biencoder_3B_freeze.pt`) and GRAIL
(`biencoder_3C_freeze_dy.pt`) specialists.

```bash
python train/train_hybrid_selector.py       # -> hybrid_selector.pt
```

### 6. Evaluation

```bash
# Evidence Set Completion (Recall@K, Delta_esc)
python eval/eval_set_completion.py --run_tag 3C_freeze_dy --s2 C --exclude_ctx

# Sequential Pool Construction (Set-Recall, Rank Jump, NRM)
python eval/eval_sequential_pool_table.py
python eval/eval_sequential_pool.py

# Rocchio baselines
python eval/eval_baseline_rocchio_tuned.py
python eval/eval_baseline_rocchio.py        # grid search over beta / step sizes
```

---

## Option and checkpoint naming

Query-steering option (`--s2`):

| `--s2` | Method     | Paper term                     |
|--------|------------|--------------------------------|
| A      | Query-only | "Query Only" baseline          |
| B      | Additive   | "Additive" baseline            |
| C      | GRAIL      | gap-aware subtractive steering |

Stage-1 alignment init (`--s1`):

| `--s1` | Strategy       | Checkpoint                      |
|--------|----------------|---------------------------------|
| 0      | none (scratch) | —                               |
| 1      | Centroid       | `alignment_centroid.pt`         |
| 2      | CLS            | `alignment_cls.pt`              |
| 3      | Query-evidence | `alignment_query_evidence.pt`   |
| 5      | Query-set      | `alignment_query_set.pt`        |

Stage-2 checkpoints follow `biencoder_{S1}{S2}_{freeze|full}[_dy].pt`
(`_dy` = dynamic temperature, Option C only). For example,
`--s1 3 --s2 C --freeze` saves `biencoder_3C_freeze_dy.pt`.

All models are trained for **100 epochs** (fixed).

---

## Not included: LLM baselines

The LLM-based query-regeneration baselines (Closed-Book, IRCoT,
Answer-Augmented; Section 5.3 of the paper) rely on an external `gpt-4o-mini`
API and are out of scope for this code release.

---

## Contact

Questions, data, or checkpoints: **ec_osa16586@snu.ac.kr**
