#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

LOG_FILE="logs/finetune_pemsd4_mae_rmse_w035_bs32.out"
PID_FILE="logs/finetune_pemsd4_mae_rmse_w035_bs32.pid"
PYTHON_BIN="/root/miniconda3/envs/PDG2SEQ_CU128/bin/python"

setsid -f env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "${PYTHON_BIN}" -u run.py \
  --dataset PEMSD4 \
  --mode train \
  --device cuda:0 \
  --batch_size 32 \
  --epochs 80 \
  --early_stop true \
  --early_stop_patience 20 \
  --save_every 5 \
  --lr_init 0.0003 \
  --loss_func mae_rmse \
  --rmse_loss_weight 0.35 \
  --select_metric hybrid \
  --train_init_path experiments/PEMSD4/newinnovation-1-2_DGQ-DGQEnsemble-PeriodicContext-ContextGraphRefine-MetaReliableGraph-PeriodicConsistency_20260804220312/best_test_model.pth \
  --use_dgq true \
  --dgq_eval_ensemble true \
  --dgq_teacher_path ./pre-trained/PEMSD4.pth \
  --use_periodic_context true \
  --use_context_graph_refine true \
  --use_meta_reliable_graph true \
  --meta_auto_features false \
  --use_periodic_consistency true \
  --use_eval_calibration true \
  > "${LOG_FILE}" 2>&1

pgrep -f "run.py --dataset PEMSD4 --mode train .*--loss_func mae_rmse" | tail -n 1 > "${PID_FILE}"
echo "started pid $(cat "${PID_FILE}") log ${LOG_FILE}"
