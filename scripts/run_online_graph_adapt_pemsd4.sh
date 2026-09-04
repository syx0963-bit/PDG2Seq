#!/usr/bin/env bash
set -euo pipefail

cd /root/PDG2Seq

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

PYTHON_BIN="/root/miniconda3/envs/PDG2SEQ_CU128/bin/python"
LOG_FILE="logs/online_graph_adapt_pemsd4.out"

"${PYTHON_BIN}" -u run.py \
  --dataset PEMSD4 \
  --mode test \
  --device cuda:0 \
  --batch_size 32 \
  --test_model_path experiments/PEMSD4/newinnovation-1-2_DGQ-DGQEnsemble-PeriodicContext-ContextGraphRefine-MetaReliableGraph-PeriodicConsistency_20260813215539/best_test_model.pth \
  --use_dgq true \
  --dgq_eval_ensemble true \
  --dgq_teacher_path ./pre-trained/PEMSD4.pth \
  --use_periodic_context true \
  --use_context_graph_refine true \
  --use_meta_reliable_graph true \
  --meta_auto_features false \
  --use_periodic_consistency true \
  --use_eval_calibration true \
  --select_metric balanced \
  --use_online_adaptation true \
  --online_adapt_lr 0.04 \
  --online_scale_lr 0.08 \
  --online_adapt_decay 0.97 \
  --online_error_decay 0.92 \
  --online_drift_sensitivity 0.7 \
  --online_graph_topk 8 \
  --online_neighbor_expand 0.50 \
  --online_bias_clip 4.0 \
  --online_scale_clip 0.24 \
  --online_warmup_val true \
  --online_overlap_memory false \
  --online_overlap_blend 0.0 \
  | tee "${LOG_FILE}"
