#!/usr/bin/env bash
#
# End-to-end GRAIL pipeline: data -> embeddings -> alignment -> GRAIL -> hybrid -> eval.
#
# Prerequisites (place in the repo root, or set MMQA_DATA_DIR to point elsewhere):
#   MMQA_train.jsonl, MMQA_dev.jsonl, MMQA_train_*.json
#   MMQA_texts.jsonl, MMQA_tables.jsonl, MMQA_images.jsonl
#
# Run from the repository root:  bash run_all.sh
# Training/embedding steps require a GPU and can take hours.

set -e  # stop on first error
cd "$(dirname "$0")"   # run from the repo root

echo "================ [0/6] Build id maps ================"
python data/00_build_id_maps.py

echo "================ [1/6] Build dataset ================"
python data/01_build_raw_gold_chains.py
python data/02_enrich_gold_chains.py
python data/03_build_strict_chains.py
python data/04_build_splits.py

echo "================ [2/6] Precompute embeddings ================"
python embeddings/embed_text_table.py
python embeddings/embed_images.py          # downloads Wikipedia images

echo "================ [3/6] Stage 1: alignment (default: query-evidence) ================"
python pretrain/align_query_evidence.py
# Other strategies (optional):
# python pretrain/align_centroid.py
# python pretrain/align_cls.py
# python pretrain/align_query_set.py

echo "================ [4/6] Stage 2: train retrievers (s1=3) ================"
python train/train_grail.py --s1 3 --s2 A --freeze   # Query-only baseline
python train/train_grail.py --s1 3 --s2 B --freeze   # Additive baseline
python train/train_grail.py --s1 3 --s2 C --freeze   # GRAIL (main)

echo "================ [5/6] Stage 3: hybrid selector ================"
python train/train_hybrid_selector.py

echo "================ [6/6] Evaluation ================"
python eval/eval_set_completion.py --run_tag 3C_freeze_dy --s2 C --exclude_ctx
python eval/eval_sequential_pool_table.py
python eval/eval_sequential_pool.py
python eval/eval_baseline_rocchio_tuned.py

echo "================ Done. ================"
