#!/usr/bin/env bash
set -euo pipefail

cd /root/PDG2Seq

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=logs/finetune_pemsd4_balanced_mapeaware_w003_bs32.out
PIDFILE=logs/finetune_pemsd4_balanced_mapeaware_w003_bs32.pid

nohup /root/miniconda3/envs/PDG2SEQ_CU128/bin/python run.py \
  --dataset PEMSD4 \
  --mode train \
  --device cuda:0 \
  --use_dgq true \
  --dgq_eval_ensemble true \
  --dgq_teacher_path ./pre-trained/PEMSD4.pth \
  --use_periodic_context true \
  --use_context_graph_refine true \
  --use_meta_reliable_graph true \
  --meta_auto_features false \
  --use_signal_decouple false \
  --use_decoder_periodic_context false \
  --use_periodic_consistency true \
  --use_eval_calibration true \
  --select_metric balanced \
  --loss_func mape_aware_mae \
  --mape_loss_weight 0.03 \
  --batch_size 32 \
  --lr_init 0.0005 \
  --epochs 80 \
  --early_stop_patience 20 \
  --save_every 5 \
  --train_init_path experiments/PEMSD4/newinnovation-1-2_DGQ-DGQEnsemble-PeriodicContext-ContextGraphRefine-MetaReliableGraph-PeriodicConsistency_20260804220312/best_test_model.pth \
  > "$OUT" 2>&1 &

echo $! > "$PIDFILE"
echo "started pid $(cat "$PIDFILE")"
echo "stdout log: $OUT"
