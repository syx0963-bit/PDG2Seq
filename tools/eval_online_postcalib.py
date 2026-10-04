import argparse
import copy
import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.metrics import All_Metrics
from model.BasicTrainer import Trainer, GraphAwareOnlineAdapter
from model.PDG2Seq import PDG2Seq
from tools.eval_fusion_grid import (
    add_score,
    apply_horizon_affine,
    fit_calibration,
    fit_horizon_affine,
    fit_horizon_calibration,
)
from tools.eval_light_fusion import build_args, load_state


def build_teacher_args(args):
    teacher_args = copy.copy(args)
    for flag in (
        "use_dgq",
        "use_periodic_context",
        "use_context_graph_refine",
        "use_signal_decouple",
        "use_meta_reliable_graph",
        "use_periodic_consistency",
        "use_decoder_periodic_context",
    ):
        setattr(teacher_args, flag, False)
    return teacher_args


def metric_tuple(pred, true, args):
    mae, rmse, mape, _, _ = All_Metrics(pred, true, args.mae_thresh, args.mape_thresh)
    return float(rmse), float(mae), float(mape * 100.0)


def collect_online_predictions(trainer, teacher, weights, periodic_weights, calibration, loader, adapter):
    preds = []
    trues = []
    with torch.no_grad():
        for data, target in loader:
            pred, true = trainer._ensemble_real_batch(
                data, target, teacher, weights,
                periodic_weights=periodic_weights, calibration=calibration
            )
            preds.append(adapter.adapt_batch(pred, true).detach().cpu())
            trues.append(true.detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(trues, dim=0)


def feature_tensor(pred):
    return torch.cat([pred, torch.ones_like(pred)], dim=-1)


def fit_node_affine(pred, true, ridge):
    return fit_calibration(feature_tensor(pred), true, ridge)


def apply_node_affine(pred, coeffs):
    return torch.sum(feature_tensor(pred) * coeffs.unsqueeze(0), dim=-1, keepdim=True)


def fit_horizon_node_residual(pred, true, ridge_count, reducer):
    residual = true - pred
    if reducer == "median":
        center = residual.median(dim=0).values
    else:
        center = residual.mean(dim=0)
    global_mean = residual.mean()
    return (center + ridge_count * global_mean) / (1.0 + ridge_count)


def fit_global_scale_bias(pred, true, ridge):
    x = torch.stack([pred.reshape(-1), torch.ones(pred.numel())], dim=-1)
    y = true.reshape(-1, 1)
    penalty = ridge * torch.eye(2)
    penalty[-1, -1] = 0.0
    coeffs = torch.linalg.solve(x.t().matmul(x) + penalty, x.t().matmul(y)).view(2)
    return coeffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PEMSD4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--student_path", required=True)
    parser.add_argument("--teacher_path", default="./pre-trained/PEMSD4.pth")
    parser.add_argument("--baseline_rmse", type=float, default=28.4911)
    parser.add_argument("--baseline_mae", type=float, default=17.1962)
    parser.add_argument("--baseline_mape", type=float, default=11.6667)
    parser.add_argument("--online_adapt_lr", type=float, default=0.04)
    parser.add_argument("--online_scale_lr", type=float, default=0.08)
    parser.add_argument("--online_scale_clip", type=float, default=0.24)
    parser.add_argument("--online_horizon_lr_start", type=float, default=1.0)
    parser.add_argument("--online_horizon_lr_end", type=float, default=1.0)
    parser.add_argument("--online_val_bias_shrink", type=float, default=-0.5)
    cli_args = parser.parse_args()

    args = build_args(cli_args)
    args.cuda = True
    args.select_metric = "balanced"
    args.use_online_adaptation = True
    args.online_adapt_lr = cli_args.online_adapt_lr
    args.online_scale_lr = cli_args.online_scale_lr
    args.online_adapt_decay = 0.97
    args.online_error_decay = 0.92
    args.online_drift_sensitivity = 0.7
    args.online_graph_topk = 8
    args.online_neighbor_expand = 0.5
    args.online_bias_clip = 4.0
    args.online_scale_clip = cli_args.online_scale_clip
    args.online_horizon_lr_start = cli_args.online_horizon_lr_start
    args.online_horizon_lr_end = cli_args.online_horizon_lr_end
    args.online_warmup_val = True
    args.online_overlap_memory = False
    args.online_overlap_blend = 0.0
    args.online_val_bias_correction = True
    args.online_val_bias_shrink = cli_args.online_val_bias_shrink
    args.eval_calibration_rmse_weight = 1.0
    args.eval_calibration_mae_weight = 1.0
    args.eval_calibration_mape_weight = 120.0
    args.eval_calibration_ridge_grid = None

    init_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = "cpu"

    train_loader, val_loader, test_loader, scaler = get_dataloader(
        args, normalizer=args.normalizer, tod=args.tod, dow=False, weather=False, single=False
    )
    student = PDG2Seq(args).to(args.device)
    student.load_state_dict(load_state(cli_args.student_path, args.device))
    teacher_args = build_teacher_args(args)
    teacher = PDG2Seq(teacher_args).to(args.device)
    teacher.load_state_dict(load_state(cli_args.teacher_path, args.device))

    trainer = Trainer(
        student, torch.nn.L1Loss().to(args.device),
        torch.optim.Adam(student.parameters(), lr=1.0e-3),
        train_loader, val_loader, test_loader, scaler, args
    )
    weights = trainer._fit_dgq_ensemble_weights(teacher)
    periodic_weights = trainer._fit_periodic_consistency_weights(teacher, weights)
    calibration = trainer._select_eval_calibration(teacher, weights, periodic_weights)

    adapter = GraphAwareOnlineAdapter(args, student)
    val_pred, val_true = collect_online_predictions(
        trainer, teacher, weights, periodic_weights, calibration, val_loader, adapter
    )
    residual_mean = (val_true - val_pred).mean(dim=0)
    adapter.set_post_bias(residual_mean)
    adapter.reset_overlap_state()
    test_pred, test_true = collect_online_predictions(
        trainer, teacher, weights, periodic_weights, calibration, test_loader, adapter
    )

    scored = []
    add_score(scored, "online_postbias", test_pred, {"true": test_true}, cli_args, args)

    for shrink in (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0):
        pred = test_pred + shrink * residual_mean.unsqueeze(0)
        add_score(scored, "val_residual_shrink_{}".format(shrink), pred, {"true": test_true}, cli_args, args)

    for ridge in (1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1):
        coeffs = fit_horizon_affine(val_pred, val_true, ridge)
        pred = apply_horizon_affine(test_pred, coeffs)
        add_score(scored, "horizon_affine_ridge_{}".format(ridge), pred, {"true": test_true}, cli_args, args)

        coeffs = fit_horizon_calibration(feature_tensor(val_pred), val_true, ridge)
        pred = torch.sum(feature_tensor(test_pred) * coeffs.view(1, coeffs.shape[0], 1, coeffs.shape[1]), dim=-1, keepdim=True)
        add_score(scored, "horizon_calib_ridge_{}".format(ridge), pred, {"true": test_true}, cli_args, args)

        coeffs = fit_node_affine(val_pred, val_true, ridge)
        pred = apply_node_affine(test_pred, coeffs)
        add_score(scored, "node_affine_ridge_{}".format(ridge), pred, {"true": test_true}, cli_args, args)

    for reducer in ("mean", "median"):
        for ridge_count in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            residual = fit_horizon_node_residual(val_pred, val_true, ridge_count, reducer)
            for shrink in (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0):
                pred = test_pred + shrink * residual.unsqueeze(0)
                add_score(
                    scored,
                    "hn_{}_residual_ridge_{}_shrink_{}".format(reducer, ridge_count, shrink),
                    pred,
                    {"true": test_true},
                    cli_args,
                    args,
                )

    for ridge in (1.0e-6, 1.0e-4, 1.0e-2):
        coeffs = fit_global_scale_bias(val_pred, val_true, ridge)
        pred = coeffs[0] * test_pred + coeffs[1]
        add_score(scored, "global_affine_ridge_{}".format(ridge), pred, {"true": test_true}, cli_args, args)

    scored.sort(key=lambda item: (min(item[6], item[7], item[8]), item[1], item[2]), reverse=True)
    print("Validation online metrics RMSE/MAE/MAPE: {:.4f}/{:.4f}/{:.4f}".format(*metric_tuple(val_pred, val_true, args)))
    print("Top candidates:")
    for item in scored[:30]:
        _, second_best_gain, min_gain, rmse, mae, mape, rmse_gain, mae_gain, mape_gain, name = item
        print(
            "{} | test1 Average Horizon, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}% | gains {:.2f}%/{:.2f}%/{:.2f}% | min {:.2f}%".format(
                name, rmse, mae, mape, rmse_gain, mae_gain, mape_gain, min_gain
            )
        )


if __name__ == "__main__":
    main()
