#!/usr/bin/env bash
set -euo pipefail

cd /root/PDG2Seq

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/root/miniconda3/envs/PDG2SEQ_CU128/bin/python"

"${PYTHON_BIN}" -u run.py \
  --dataset PEMSD4 \
  --mode train \
  --device cuda:0 \
  --batch_size 16 \
  --loss_func mape_aware_mae \
  --mape_loss_weight 0.03 \
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
  --use_reliable_invariant_learning true \
  --invariant_loss_weight 0.10 \
  --env_pred_loss_weight 0.50 \
  --env_orth_loss_weight 0.01 \
  --invariant_reliable_topk 8 \
  --env_perturb_node_prob 0.35 \
  --env_perturb_scale 0.20 \
  --env_perturb_bias 0.15 \
  --env_perturb_noise 0.03 \
  --env_perturb_mask_prob 0.05 \
  --env_perturb_periodic_shift_prob 0.20 \
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
  --online_overlap_memory true \
  --online_overlap_blend 1.0
