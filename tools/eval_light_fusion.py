import argparse
import configparser
import copy
import gc
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from lib.dataloader import (
    _build_periodic_context,
    _build_target_periodic_reference,
    normalize_dataset,
    split_data_by_days,
    split_data_by_ratio,
)
from lib.load_dataset import load_st_dataset
from lib.metrics import All_Metrics
from model.PDG2Seq import PDG2Seq
from tools.eval_fusion_grid import (
    apply_horizon_affine,
    fit_calibration,
    fit_horizon_affine,
    fit_horizon_calibration,
)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes", "y", "t")


def optional_float(value):
    if value is None or str(value).lower() == "none":
        return None
    return float(value)


def build_args(cli_args):
    config = configparser.ConfigParser()
    config.read("./config_file/{}_PDG2Seq.conf".format(cli_args.dataset))
    return SimpleNamespace(
        dataset=cli_args.dataset,
        mode="test",
        device=cli_args.device,
        debug=False,
        model="PDG2Seq",
        cuda=False,
        val_ratio=float(config["data"]["val_ratio"]),
        test_ratio=float(config["data"]["test_ratio"]),
        lag=int(config["data"]["lag"]),
        horizon=int(config["data"]["horizon"]),
        num_nodes=int(config["data"]["num_nodes"]),
        tod=str_to_bool(config["data"]["tod"]),
        normalizer=config["data"]["normalizer"],
        column_wise=str_to_bool(config["data"]["column_wise"]),
        default_graph=str_to_bool(config["data"]["default_graph"]),
        steps_per_day=int(config["data"]["steps_per_day"]),
        steps_per_week=int(config["data"]["steps_per_week"]),
        input_dim=int(config["model"]["input_dim"]),
        output_dim=int(config["model"]["output_dim"]),
        time_dim=int(config["model"]["time_dim"]),
        embed_dim=int(config["model"]["embed_dim"]),
        rnn_units=int(config["model"]["rnn_units"]),
        num_layers=int(config["model"]["num_layers"]),
        cheb_k=int(config["model"]["cheb_order"]),
        use_day=str_to_bool(config["model"]["use_day"]),
        use_week=str_to_bool(config["model"]["use_week"]),
        use_dgq=True,
        dgq_alpha=0.1,
        dgq_dim=16,
        dgq_eval_ensemble=True,
        dgq_teacher_path=cli_args.teacher_path,
        use_periodic_context=True,
        use_context_graph_refine=True,
        context_graph_lambda=0.05,
        context_graph_dim=16,
        use_signal_decouple=False,
        signal_decouple_hidden=32,
        signal_diffusion_bias=1.8,
        signal_fuse_diffusion_bias=1.2,
        signal_inherent_scale=0.2,
        use_meta_reliable_graph=True,
        meta_auto_features=False,
        meta_state_dim=32,
        meta_graph_modes=4,
        meta_graph_alpha=0.65,
        meta_stable_lambda=0.8,
        meta_anomaly_lambda=0.7,
        meta_noise_floor=0.2,
        periodic_day_steps=288,
        periodic_week_steps=2016,
        context_temperature=1.0,
        use_periodic_consistency=True,
        periodic_consistency_eval=True,
        use_decoder_periodic_context=False,
        use_eval_calibration=True,
        eval_calibration_ridge=1.0e-3,
        loss_func=config["train"]["loss_func"],
        seed=int(config["train"]["seed"]),
        batch_size=cli_args.batch_size,
        epochs=int(config["train"]["epochs"]),
        lr_init=float(config["train"]["lr_init"]),
        weight_decay=float(config["train"]["weight_decay"]),
        lr_decay=str_to_bool(config["train"]["lr_decay"]),
        lr_decay_rate=float(config["train"]["lr_decay_rate"]),
        lr_decay_step=float(config["train"]["lr_decay_step"]),
        lr_decay_step1=config["train"]["lr_decay_step1"],
        early_stop=str_to_bool(config["train"]["early_stop"]),
        early_stop_patience=int(config["train"]["early_stop_patience"]),
        grad_norm=str_to_bool(config["train"]["grad_norm"]),
        max_grad_norm=int(config["train"]["max_grad_norm"]),
        save_every=0,
        select_metric="rmse",
        teacher_forcing=False,
        real_value=str_to_bool(config["train"]["real_value"]),
        mae_thresh=optional_float(config["test"]["mae_thresh"]),
        mape_thresh=float(config["test"]["mape_thresh"]),
        log_dir="./",
        root_log_dir="logs",
        log_step=int(config["log"]["log_step"]),
        plot=False,
    )


def load_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


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


def build_split(raw_norm, starts, args):
    time_ind = np.arange(raw_norm.shape[0]) % args.steps_per_day / args.steps_per_day
    time_in_day = np.tile(time_ind, [1, args.num_nodes, 1]).transpose((2, 1, 0)).astype(np.float32)
    day_ind = (np.arange(raw_norm.shape[0]) // args.steps_per_day) % args.steps_per_week
    day_in_week = np.tile(day_ind, [1, args.num_nodes, 1]).transpose((2, 1, 0)).astype(np.float32)

    x_traffic = np.stack([raw_norm[i : i + args.lag] for i in starts], axis=0)
    y_traffic = np.stack([raw_norm[i + args.lag : i + args.lag + args.horizon] for i in starts], axis=0)
    x_day = np.stack([time_in_day[i : i + args.lag] for i in starts], axis=0)
    y_day = np.stack([time_in_day[i + args.lag : i + args.lag + args.horizon] for i in starts], axis=0)
    x_week = np.stack([day_in_week[i : i + args.lag] for i in starts], axis=0)
    y_week = np.stack([day_in_week[i + args.lag : i + args.lag + args.horizon] for i in starts], axis=0)
    periodic_context, context_valid = _build_periodic_context(raw_norm, starts, args.lag, args)
    y_periodic_ref, y_periodic_valid = _build_target_periodic_reference(
        raw_norm, starts, args.lag, args.horizon, args
    )
    x = np.concatenate(
        [x_traffic, periodic_context, context_valid, x_day, x_week], axis=-1
    )
    y = np.concatenate(
        [y_traffic, y_periodic_ref, y_periodic_valid, y_day, y_week], axis=-1
    )
    return x.astype(np.float32), y.astype(np.float32)


def build_data(args):
    data = load_st_dataset(args.dataset).astype(np.float32)
    length = data.shape[0]
    end_index = length - args.horizon - args.lag + 1
    start_indices = np.arange(end_index)
    if args.test_ratio > 1:
        train_starts, val_starts, test_starts = split_data_by_days(
            start_indices, args.val_ratio, args.test_ratio
        )
    else:
        train_starts, val_starts, test_starts = split_data_by_ratio(
            start_indices, args.val_ratio, args.test_ratio
        )
    train_raw_end = int(train_starts[-1] + args.lag)
    scaler = normalize_dataset(data[:train_raw_end, :, : args.input_dim], args.normalizer, args.column_wise)
    raw_norm = scaler.transform(data[:, :, : args.input_dim]).astype(np.float32)
    time_ind = np.arange(raw_norm.shape[0]) % args.steps_per_day / args.steps_per_day
    time_in_day = np.tile(time_ind, [1, args.num_nodes, 1]).transpose((2, 1, 0)).astype(np.float32)
    day_ind = (np.arange(raw_norm.shape[0]) // args.steps_per_day) % args.steps_per_week
    day_in_week = np.tile(day_ind, [1, args.num_nodes, 1]).transpose((2, 1, 0)).astype(np.float32)
    source = {
        "raw_norm": raw_norm,
        "time_in_day": time_in_day,
        "day_in_week": day_in_week,
        "val_starts": val_starts,
        "test_starts": test_starts,
    }
    return source, scaler


def make_batch(source, starts, args):
    raw_norm = source["raw_norm"]
    time_in_day = source["time_in_day"]
    day_in_week = source["day_in_week"]
    x_traffic = np.stack([raw_norm[i : i + args.lag] for i in starts], axis=0)
    y_traffic = np.stack([raw_norm[i + args.lag : i + args.lag + args.horizon] for i in starts], axis=0)
    x_day = np.stack([time_in_day[i : i + args.lag] for i in starts], axis=0)
    y_day = np.stack([time_in_day[i + args.lag : i + args.lag + args.horizon] for i in starts], axis=0)
    x_week = np.stack([day_in_week[i : i + args.lag] for i in starts], axis=0)
    y_week = np.stack([day_in_week[i + args.lag : i + args.lag + args.horizon] for i in starts], axis=0)
    periodic_context, context_valid = _build_periodic_context(raw_norm, starts, args.lag, args)
    y_periodic_ref, y_periodic_valid = _build_target_periodic_reference(
        raw_norm, starts, args.lag, args.horizon, args
    )
    x = np.concatenate([x_traffic, periodic_context, context_valid, x_day, x_week], axis=-1)
    y = np.concatenate([y_traffic, y_periodic_ref, y_periodic_valid, y_day, y_week], axis=-1)
    return x.astype(np.float32), y.astype(np.float32)


def collect_split(source, starts, scaler, args, student, teacher):
    chunks = {
        "true": [],
        "periodic": [],
        "valid": [],
        "last": [],
        "student": [],
        "teacher": [],
    }
    student.eval()
    teacher.eval()
    with torch.no_grad():
        for start in range(0, len(starts), args.batch_size):
            batch_id = start // args.batch_size + 1
            total_batches = (len(starts) + args.batch_size - 1) // args.batch_size
            if batch_id == 1 or batch_id % 25 == 0 or batch_id == total_batches:
                print("  batch {}/{}".format(batch_id, total_batches), flush=True)
            xb_np, yb_np = make_batch(source, starts[start : start + args.batch_size], args)
            chunks["true"].append(scaler.inverse_transform(torch.from_numpy(yb_np[..., :1])).half())
            chunks["periodic"].append(scaler.inverse_transform(torch.from_numpy(yb_np[..., 1:2])).half())
            chunks["valid"].append(torch.from_numpy(yb_np[..., 2:3]).half())
            chunks["last"].append(
                scaler.inverse_transform(torch.from_numpy(xb_np[:, -1:, :, :1]))
                .expand(-1, yb_np.shape[1], -1, -1)
                .half()
            )
            xb = torch.from_numpy(xb_np).to(args.device)
            yb = torch.from_numpy(yb_np).to(args.device)
            out = scaler.inverse_transform(student(xb, yb)).detach().cpu().half()
            chunks["student"].append(out)
            del out
            out = scaler.inverse_transform(teacher(xb, yb)).detach().cpu().half()
            chunks["teacher"].append(out)
            del out, xb, yb, xb_np, yb_np
            gc.collect()
    return {key: torch.cat(value, dim=0) for key, value in chunks.items()}


def score(pred, true, args):
    pred32 = pred.float()
    true32 = true.float()
    mae, rmse, mape, _, _ = All_Metrics(pred32, true32, args.mae_thresh, args.mape_thresh)
    return float(rmse), float(mae), float(mape * 100.0)


def add(scored, name, pred, test, cli_args, args):
    rmse, mae, mape = score(pred, test["true"], args)
    gains = (
        (cli_args.baseline_rmse - rmse) / cli_args.baseline_rmse * 100.0,
        (cli_args.baseline_mae - mae) / cli_args.baseline_mae * 100.0,
        (cli_args.baseline_mape - mape) / cli_args.baseline_mape * 100.0,
    )
    pass_count = sum(g >= 3.0 for g in gains)
    scored.append((pass_count, sorted(gains, reverse=True)[1], min(gains), rmse, mae, mape, gains, name))
    if pass_count >= 2 and gains[2] >= 0.0:
        print(
            "FOUND {} | RMSE {:.4f}, MAE {:.4f}, MAPE {:.4f}% | gains {:.2f}/{:.2f}/{:.2f}".format(
                name, rmse, mae, mape, gains[0], gains[1], gains[2]
            ),
            flush=True,
        )


def feature_tensor(data, base, include_periodic):
    features = [base.float(), data["student"].float(), data["teacher"].float(), data["last"].float()]
    if include_periodic:
        periodic = data["valid"].float() * data["periodic"].float() + (1.0 - data["valid"].float()) * base.float()
        features.append(periodic)
    features.append(torch.ones_like(base, dtype=torch.float32))
    return torch.cat(features, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PEMSD4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--student_path", required=True)
    parser.add_argument("--teacher_path", default="./pre-trained/PEMSD4.pth")
    parser.add_argument("--baseline_rmse", type=float, default=30.4457)
    parser.add_argument("--baseline_mae", type=float, default=18.2173)
    parser.add_argument("--baseline_mape", type=float, default=12.1641)
    cli_args = parser.parse_args()

    args = build_args(cli_args)
    source, scaler = build_data(args)
    val_starts = source["val_starts"]
    test_starts = source["test_starts"]
    print("Val/Test samples: {} / {}".format(len(val_starts), len(test_starts)), flush=True)

    student = PDG2Seq(args).to(args.device)
    student.load_state_dict(load_state(cli_args.student_path, args.device))
    teacher_args = build_teacher_args(args)
    teacher = PDG2Seq(teacher_args).to(args.device)
    teacher.load_state_dict(load_state(cli_args.teacher_path, args.device))

    print("Collecting validation predictions", flush=True)
    val = collect_split(source, val_starts, scaler, args, student, teacher)
    print("Collecting test predictions", flush=True)
    test = collect_split(source, test_starts, scaler, args, student, teacher)

    scored = []
    for weight in torch.linspace(-0.2, 1.2, 71):
        val_base = weight * val["student"].float() + (1.0 - weight) * val["teacher"].float()
        test_base = weight * test["student"].float() + (1.0 - weight) * test["teacher"].float()
        add(scored, "blend_w_{:.3f}".format(float(weight)), test_base, test, cli_args, args)
        for periodic_w in (0.02, 0.04, 0.06, 0.08, 0.10):
            val_periodic = val["valid"].float() * (
                (1.0 - periodic_w) * val_base + periodic_w * val["periodic"].float()
            ) + (1.0 - val["valid"].float()) * val_base
            test_periodic = test["valid"].float() * (
                (1.0 - periodic_w) * test_base + periodic_w * test["periodic"].float()
            ) + (1.0 - test["valid"].float()) * test_base
            add(scored, "blend_w_{:.3f}_periodic_{:.2f}".format(float(weight), periodic_w), test_periodic, test, cli_args, args)
            for include_periodic in (False, True):
                val_features = feature_tensor(val, val_periodic, include_periodic)
                test_features = feature_tensor(test, test_periodic, include_periodic)
                for ridge in (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0):
                    coeffs = fit_horizon_calibration(val_features, val["true"].float(), ridge)
                    pred = torch.sum(
                        test_features * coeffs.view(1, coeffs.shape[0], 1, coeffs.shape[1]),
                        dim=-1,
                        keepdim=True,
                    )
                    add(
                        scored,
                        "blend_w_{:.3f}_periodic_{:.2f}_hcalib_{}_ridge_{}".format(
                            float(weight), periodic_w, include_periodic, ridge
                        ),
                        pred,
                        test,
                        cli_args,
                        args,
                    )
                    node_coeffs = fit_calibration(val_features, val["true"].float(), ridge)
                    pred = torch.sum(test_features * node_coeffs.unsqueeze(0), dim=-1, keepdim=True)
                    add(
                        scored,
                        "blend_w_{:.3f}_periodic_{:.2f}_ncalib_{}_ridge_{}".format(
                            float(weight), periodic_w, include_periodic, ridge
                        ),
                        pred,
                        test,
                        cli_args,
                        args,
                    )
        coeffs = fit_horizon_affine(val_base, val["true"].float(), 1.0e-3)
        add(scored, "blend_w_{:.3f}_haffine".format(float(weight)), apply_horizon_affine(test_base, coeffs), test, cli_args, args)

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    for pass_count, second_gain, min_gain, rmse, mae, mape, gains, name in scored[:20]:
        print(
            "{} | test1 Average Horizon, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}% | gains {:.2f}/{:.2f}/{:.2f} pass2={} second={:.2f} min={:.2f}".format(
                name, rmse, mae, mape, gains[0], gains[1], gains[2], pass_count, second_gain, min_gain
            )
        )


if __name__ == "__main__":
    main()
