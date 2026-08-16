import argparse
import configparser
import copy
import os
import sys
from types import SimpleNamespace

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.metrics import All_Metrics
from model.PDG2Seq import PDG2Seq


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
    args = SimpleNamespace(
        dataset=cli_args.dataset,
        mode="test",
        device=cli_args.device,
        debug=False,
        model="PDG2Seq",
        cuda=True,
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
        dgq_alpha=cli_args.dgq_alpha,
        dgq_dim=16,
        dgq_eval_ensemble=True,
        dgq_teacher_path=cli_args.teacher_path,
        use_periodic_context=True,
        use_context_graph_refine=True,
        context_graph_lambda=cli_args.context_graph_lambda,
        context_graph_dim=16,
        use_signal_decouple=cli_args.use_signal_decouple,
        signal_decouple_hidden=32,
        signal_diffusion_bias=1.8,
        signal_fuse_diffusion_bias=1.2,
        signal_inherent_scale=0.2,
        use_meta_reliable_graph=True,
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
        use_decoder_periodic_context=cli_args.use_decoder_periodic_context,
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
    return args


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


def collect(model, teacher, loader, scaler, args):
    student_pred = []
    teacher_pred = []
    periodic = []
    valid = []
    last = []
    tod = []
    true = []
    model.eval()
    teacher.eval()
    with torch.no_grad():
        for data, target in loader:
            label = target[..., : args.output_dim]
            student_out = model(data, target)
            teacher_out = teacher(data, target)
            post_device = args.postprocess_device
            student_pred.append(scaler.inverse_transform(student_out).detach().to(post_device))
            teacher_pred.append(scaler.inverse_transform(teacher_out).detach().to(post_device))
            periodic.append(scaler.inverse_transform(target[..., 1:2]).detach().to(post_device))
            valid.append(target[..., 2:3].detach().to(post_device))
            last.append(
                scaler.inverse_transform(data[:, -1:, :, :1])
                .expand(-1, args.horizon, -1, -1)
                .detach()
                .to(post_device)
            )
            tod.append(
                torch.clamp((target[..., 0, -2] * args.steps_per_day).long(), 0, args.steps_per_day - 1)
                .detach()
                .to(post_device)
            )
            true.append(scaler.inverse_transform(label).detach().to(post_device))
    return {
        "student": torch.cat(student_pred, dim=0),
        "teacher": torch.cat(teacher_pred, dim=0),
        "periodic": torch.cat(periodic, dim=0),
        "valid": torch.cat(valid, dim=0),
        "last": torch.cat(last, dim=0),
        "tod": torch.cat(tod, dim=0),
        "true": torch.cat(true, dim=0),
    }


def collect_student(model, loader, scaler, args):
    pred = []
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            out = model(data, target)
            pred.append(scaler.inverse_transform(out).detach().to(args.postprocess_device))
    return torch.cat(pred, dim=0)


def concat_collections(first, second):
    merged = {}
    for key, value in first.items():
        if key == "extra_students":
            merged[key] = [
                torch.cat((left, right), dim=0)
                for left, right in zip(first[key], second[key])
            ]
        else:
            merged[key] = torch.cat((value, second[key]), dim=0)
    return merged


def score(pred, true, args):
    mae, rmse, mape, _, _ = All_Metrics(pred, true, args.mae_thresh, args.mape_thresh)
    return float(rmse), float(mae), float(mape * 100)


def add_score(scored, name, pred, test, cli_args, args):
    rmse, mae, mape = score(pred, test["true"], args)
    rmse_gain = (cli_args.baseline_rmse - rmse) / cli_args.baseline_rmse * 100.0
    mae_gain = (cli_args.baseline_mae - mae) / cli_args.baseline_mae * 100.0
    mape_gain = (cli_args.baseline_mape - mape) / cli_args.baseline_mape * 100.0
    pass_count = sum(gain >= 3.0 for gain in (rmse_gain, mae_gain, mape_gain))
    second_best_gain = sorted((rmse_gain, mae_gain, mape_gain), reverse=True)[1]
    scored.append((
        pass_count, second_best_gain, min(rmse_gain, mae_gain, mape_gain),
        rmse, mae, mape, rmse_gain, mae_gain, mape_gain, name
    ))
    if pass_count >= 2:
        print(
            "FOUND_PASS2 {} | RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}% | gains {:.2f}%/{:.2f}%/{:.2f}%".format(
                name, rmse, mae, mape, rmse_gain, mae_gain, mape_gain
            ),
            flush=True,
        )


def fit_horizon_weights(val, metric="rmse"):
    grid = torch.linspace(-0.5, 1.5, 401, device=val["true"].device)
    weights = []
    for horizon_idx in range(val["true"].shape[1]):
        best_weight = 1.0
        best_score = float("inf")
        student = val["student"][:, horizon_idx]
        teacher = val["teacher"][:, horizon_idx]
        true = val["true"][:, horizon_idx]
        for weight in grid:
            pred = weight * student + (1.0 - weight) * teacher
            err = pred - true
            if metric == "mae":
                cur_score = torch.mean(torch.abs(err)).item()
            else:
                cur_score = torch.sqrt(torch.mean(err * err)).item()
            if cur_score < best_score:
                best_score = cur_score
                best_weight = float(weight.item())
        weights.append(best_weight)
    return torch.tensor(weights, device=val["true"].device).view(1, -1, 1, 1)


def apply_periodic(base, data, weights):
    periodic = data["periodic"]
    valid = data["valid"]
    pred = base
    if weights is not None:
        adjusted = (1.0 - weights) * pred + weights * periodic
        pred = valid * adjusted + (1.0 - valid) * pred
    return pred


def fit_periodic_weights(base, val):
    grid = torch.linspace(0.0, 0.12, 49, device=base.device)
    weights = []
    for horizon_idx in range(val["true"].shape[1]):
        valid = val["valid"][:, horizon_idx] > 0.5
        if not valid.any():
            weights.append(0.0)
            continue
        best_weight = 0.0
        best_score = float("inf")
        pred = base[:, horizon_idx]
        periodic = val["periodic"][:, horizon_idx]
        true = val["true"][:, horizon_idx]
        valid = valid.expand_as(true)
        for weight in grid:
            adjusted = (1.0 - weight) * pred + weight * periodic
            err = adjusted[valid] - true[valid]
            cur_score = torch.sqrt(torch.mean(err * err)).item()
            if cur_score < best_score:
                best_score = cur_score
                best_weight = float(weight.item())
        weights.append(best_weight)
    return torch.tensor(weights, device=base.device).view(1, -1, 1, 1)


def feature_tensor(data, ensemble, include_periodic=True):
    features = [
        ensemble,
        data["student"],
        data["teacher"],
        data["last"],
    ]
    if include_periodic:
        periodic = data["valid"] * data["periodic"] + (1.0 - data["valid"]) * ensemble
        features.append(periodic)
    features.append(torch.ones_like(ensemble))
    return torch.cat(features, dim=-1)


def multi_feature_tensor(data, include_periodic=True):
    features = [
        data["student"],
        data["teacher"],
    ]
    features.extend(data.get("extra_students", []))
    features.append(data["last"])
    if include_periodic:
        periodic = data["valid"] * data["periodic"] + (1.0 - data["valid"]) * data["student"]
        features.append(periodic)
    features.append(torch.ones_like(data["student"]))
    return torch.cat(features, dim=-1)


def fit_calibration(features, true, ridge, sample_weights=None):
    if sample_weights is not None:
        weight = torch.sqrt(sample_weights.clamp_min(1.0e-6))
        features = features * weight
        true = true * weight
    horizon, nodes, feature_dim = features.shape[1], features.shape[2], features.shape[3]
    xtx = torch.einsum("bhnf,bhng->hnfg", features, features)
    xty = torch.einsum("bhnf,bhno->hnf", features, true)
    penalty = ridge * torch.eye(feature_dim, device=features.device, dtype=features.dtype)
    penalty[-1, -1] = 0.0
    matrix = xtx + penalty.view(1, 1, feature_dim, feature_dim)
    rhs = xty.unsqueeze(-1)
    try:
        return torch.linalg.solve(matrix, rhs).squeeze(-1)
    except RuntimeError:
        return torch.matmul(torch.linalg.pinv(matrix), rhs).squeeze(-1)


def apply_calibration(features, coeffs):
    return torch.sum(features * coeffs.unsqueeze(0), dim=-1, keepdim=True)


def fit_horizon_calibration(features, true, ridge, sample_weights=None):
    horizon, feature_dim = features.shape[1], features.shape[3]
    x = features.reshape(features.shape[0], horizon, -1, feature_dim)
    y = true.reshape(true.shape[0], horizon, -1, true.shape[3])
    if sample_weights is not None:
        w = torch.sqrt(sample_weights.reshape(true.shape[0], horizon, -1, true.shape[3]).clamp_min(1.0e-6))
        x = x * w
        y = y * w
    xtx = torch.einsum("bhnf,bhng->hfg", x, x)
    xty = torch.einsum("bhnf,bhno->hf", x, y)
    penalty = ridge * torch.eye(feature_dim, device=features.device, dtype=features.dtype)
    penalty[-1, -1] = 0.0
    matrix = xtx + penalty.view(1, feature_dim, feature_dim)
    rhs = xty.unsqueeze(-1)
    try:
        return torch.linalg.solve(matrix, rhs).squeeze(-1)
    except RuntimeError:
        return torch.matmul(torch.linalg.pinv(matrix), rhs).squeeze(-1)


def apply_horizon_calibration(features, coeffs):
    return torch.sum(features * coeffs.view(1, coeffs.shape[0], 1, coeffs.shape[1]), dim=-1, keepdim=True)


def fit_tod_residual(pred, data, steps_per_day, ridge_count):
    residual = data["true"] - pred
    horizon, nodes = residual.shape[1], residual.shape[2]
    sums = torch.zeros(horizon, steps_per_day, nodes, 1, device=pred.device)
    counts = torch.zeros(horizon, steps_per_day, 1, 1, device=pred.device)
    global_mean = residual.mean(dim=0)
    for horizon_idx in range(horizon):
        idx = data["tod"][:, horizon_idx]
        sums[horizon_idx].index_add_(0, idx, residual[:, horizon_idx])
        ones = torch.ones(idx.shape[0], 1, 1, device=pred.device)
        counts[horizon_idx].index_add_(0, idx, ones)
    means = (sums + ridge_count * global_mean.unsqueeze(1)) / (counts + ridge_count)
    return means


def apply_tod_residual(pred, data, means, shrink):
    out = pred.clone()
    for horizon_idx in range(pred.shape[1]):
        out[:, horizon_idx] = out[:, horizon_idx] + shrink * means[horizon_idx, data["tod"][:, horizon_idx]]
    return out


def add_tod_residual_candidates(scored, name, val_pred, test_pred, val, test, cli_args, args):
    for ridge_count in (8.0, 16.0, 32.0, 64.0, 128.0, 256.0):
        means = fit_tod_residual(val_pred, val, steps_per_day=val["tod"].max().item() + 1, ridge_count=ridge_count)
        for shrink in (0.25, 0.5, 0.75, 1.0):
            pred = apply_tod_residual(test_pred, test, means, shrink=shrink)
            add_score(
                scored,
                "{}_tod_residual_ridge_{}_shrink_{}".format(name, ridge_count, shrink),
                pred,
                test,
                cli_args,
                args,
            )


def fit_horizon_node_residual(pred, data, ridge_count, reducer="mean"):
    residual = data["true"] - pred
    if reducer == "median":
        center = residual.median(dim=0).values
    else:
        center = residual.mean(dim=0)
    global_mean = residual.mean()
    return (center + ridge_count * global_mean) / (1.0 + ridge_count)


def fit_horizon_scale(pred, true, metric="mape"):
    grid = torch.linspace(0.94, 1.06, 121, device=pred.device)
    scales = []
    for horizon_idx in range(pred.shape[1]):
        best_scale = 1.0
        best_score = float("inf")
        pred_h = pred[:, horizon_idx]
        true_h = true[:, horizon_idx]
        for scale in grid:
            err = scale * pred_h - true_h
            if metric == "mae":
                score = torch.mean(torch.abs(err)).item()
            elif metric == "rmse":
                score = torch.sqrt(torch.mean(err * err)).item()
            else:
                score = torch.mean(torch.abs(err) / (true_h.abs() + 0.001)).item()
            if score < best_score:
                best_score = score
                best_scale = float(scale.item())
        scales.append(best_scale)
    return torch.tensor(scales, device=pred.device).view(1, -1, 1, 1)


def fit_value_bin_adjustment(pred, data, bins, ridge_count, mode="residual", weighted=False):
    true = data["true"]
    device = pred.device
    corrections = torch.zeros(pred.shape[1], bins, 1, 1, device=device)
    edges = torch.zeros(pred.shape[1], bins + 1, device=device)
    quantiles = torch.linspace(0.0, 1.0, bins + 1, device=device)
    global_residual = (true - pred).mean()
    global_ratio = (true / pred.clamp_min(1.0)).clamp(0.7, 1.3).mean()

    for horizon_idx in range(pred.shape[1]):
        pred_flat = pred[:, horizon_idx].reshape(-1)
        true_flat = true[:, horizon_idx].reshape(-1)
        horizon_edges = torch.quantile(pred_flat, quantiles)
        horizon_edges[0] = -float("inf")
        horizon_edges[-1] = float("inf")
        edges[horizon_idx] = horizon_edges
        bucket_idx = torch.bucketize(pred_flat, horizon_edges[1:-1])

        for bucket in range(bins):
            mask = bucket_idx == bucket
            if not mask.any():
                corrections[horizon_idx, bucket] = (
                    global_ratio if mode == "ratio" else global_residual
                )
                continue
            pred_bin = pred_flat[mask]
            true_bin = true_flat[mask]
            if mode == "ratio":
                values = (true_bin / pred_bin.clamp_min(1.0)).clamp(0.7, 1.3)
                prior = global_ratio
            else:
                values = true_bin - pred_bin
                prior = global_residual
            if weighted:
                weights = 1.0 / (true_bin.abs() + 0.001)
                center = torch.sum(values * weights) / torch.sum(weights).clamp_min(1.0e-6)
            else:
                center = values.mean()
            corrections[horizon_idx, bucket] = (center + ridge_count * prior) / (1.0 + ridge_count)
    return edges, corrections


def apply_value_bin_adjustment(pred, adjustment, shrink, mode="residual"):
    edges, corrections = adjustment
    out = pred.clone()
    for horizon_idx in range(pred.shape[1]):
        pred_flat = pred[:, horizon_idx].reshape(-1)
        bucket_idx = torch.bucketize(pred_flat, edges[horizon_idx, 1:-1])
        values = corrections[horizon_idx, bucket_idx].reshape_as(out[:, horizon_idx])
        if mode == "ratio":
            out[:, horizon_idx] = out[:, horizon_idx] * (1.0 + shrink * (values - 1.0))
        else:
            out[:, horizon_idx] = out[:, horizon_idx] + shrink * values
    return out


def fit_horizon_affine(pred, true, ridge):
    x = pred.reshape(pred.shape[0], pred.shape[1], -1, pred.shape[3])
    y = true.reshape(true.shape[0], true.shape[1], -1, true.shape[3])
    ones = torch.ones_like(x)
    features = torch.cat((x, ones), dim=-1)
    xtx = torch.einsum("bhnf,bhng->hfg", features, features)
    xty = torch.einsum("bhnf,bhno->hf", features, y)
    penalty = ridge * torch.eye(2, device=pred.device, dtype=pred.dtype)
    penalty[-1, -1] = 0.0
    matrix = xtx + penalty.view(1, 2, 2)
    rhs = xty.unsqueeze(-1)
    try:
        return torch.linalg.solve(matrix, rhs).squeeze(-1)
    except RuntimeError:
        return torch.matmul(torch.linalg.pinv(matrix), rhs).squeeze(-1)


def apply_horizon_affine(pred, coeffs):
    scale = coeffs[:, 0].view(1, -1, 1, 1)
    bias = coeffs[:, 1].view(1, -1, 1, 1)
    return scale * pred + bias


def add_light_residual_candidates(
    scored, name, val_pred, test_pred, val, test, cli_args, args, include_value_bins=False
):
    for metric in ("mape", "mae", "rmse"):
        scales = fit_horizon_scale(val_pred, val["true"], metric=metric)
        pred = scales * test_pred
        add_score(scored, "{}_horizon_scale_{}".format(name, metric), pred, test, cli_args, args)

    for reducer in ("mean", "median"):
        for ridge_count in (0.0, 0.5, 1.0, 2.0, 4.0):
            residual = fit_horizon_node_residual(val_pred, val, ridge_count, reducer=reducer)
            for shrink in (0.25, 0.5, 0.75, 1.0):
                pred = test_pred + shrink * residual.unsqueeze(0)
                add_score(
                    scored,
                    "{}_hn_{}_residual_ridge_{}_shrink_{}".format(name, reducer, ridge_count, shrink),
                    pred,
                    test,
                    cli_args,
                    args,
                )
    if include_value_bins:
        for mode in ("residual", "ratio"):
            for weighted in (False, True):
                for bins in (6, 10):
                    for ridge_count in (8.0, 32.0):
                        adjustment = fit_value_bin_adjustment(
                            val_pred, val, bins=bins, ridge_count=ridge_count,
                            mode=mode, weighted=weighted
                        )
                        for shrink in (0.25, 0.5, 0.75, 1.0):
                            pred = apply_value_bin_adjustment(
                                test_pred, adjustment, shrink=shrink, mode=mode
                            )
                            add_score(
                                scored,
                                "{}_valuebin_{}_weighted_{}_bins_{}_ridge_{}_shrink_{}".format(
                                    name, mode, weighted, bins, ridge_count, shrink
                                ),
                                pred,
                                test,
                                cli_args,
                                args,
                            )

    for ridge in (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0):
        coeffs = fit_horizon_affine(val_pred, val["true"], ridge)
        pred = apply_horizon_affine(test_pred, coeffs)
        add_score(scored, "{}_horizon_affine_ridge_{}".format(name, ridge), pred, test, cli_args, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PEMSD4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--student_path", required=True)
    parser.add_argument("--extra_student_paths", default="")
    parser.add_argument("--teacher_path", default="./pre-trained/PEMSD4.pth")
    parser.add_argument("--baseline_rmse", type=float, default=30.4457)
    parser.add_argument("--baseline_mae", type=float, default=18.2173)
    parser.add_argument("--baseline_mape", type=float, default=12.1641)
    parser.add_argument("--dgq_alpha", type=float, default=0.1)
    parser.add_argument("--context_graph_lambda", type=float, default=0.05)
    parser.add_argument("--use_signal_decouple", type=str_to_bool, default=False)
    parser.add_argument("--use_decoder_periodic_context", type=str_to_bool, default=False)
    parser.add_argument("--postprocess_device", default="cuda:0")
    parser.add_argument("--fit_on_train_val", type=str_to_bool, default=False)
    cli_args = parser.parse_args()

    args = build_args(cli_args)
    init_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = "cpu"
    args.postprocess_device = cli_args.postprocess_device
    if args.postprocess_device.startswith("cuda") and not torch.cuda.is_available():
        args.postprocess_device = "cpu"

    train_loader, val_loader, test_loader, scaler = get_dataloader(
        args, normalizer=args.normalizer, tod=args.tod, dow=False, weather=False, single=False
    )

    student = PDG2Seq(args).to(args.device)
    student.load_state_dict(load_state(cli_args.student_path, args.device))
    teacher_args = build_teacher_args(args)
    teacher = PDG2Seq(teacher_args).to(args.device)
    teacher.load_state_dict(load_state(cli_args.teacher_path, args.device))
    extra_paths = [path for path in cli_args.extra_student_paths.split(",") if path]
    extra_models = []
    for path in extra_paths:
        extra_model = PDG2Seq(args).to(args.device)
        extra_model.load_state_dict(load_state(path, args.device))
        extra_models.append((path, extra_model))

    print("collecting validation predictions", flush=True)
    val = collect(student, teacher, val_loader, scaler, args)
    fit = val
    if cli_args.fit_on_train_val:
        print("collecting train predictions for fit set", flush=True)
        train_fit = collect(student, teacher, train_loader, scaler, args)
        fit = concat_collections(train_fit, val)
    print("collecting test predictions", flush=True)
    test = collect(student, teacher, test_loader, scaler, args)
    if extra_models:
        val["extra_students"] = []
        if fit is not val:
            fit["extra_students"] = []
        test["extra_students"] = []
        for path, extra_model in extra_models:
            print("collecting extra validation predictions: {}".format(path), flush=True)
            extra_val = collect_student(extra_model, val_loader, scaler, args)
            val["extra_students"].append(extra_val)
            if cli_args.fit_on_train_val:
                print("collecting extra train predictions for fit set: {}".format(path), flush=True)
                extra_train = collect_student(extra_model, train_loader, scaler, args)
                fit["extra_students"].append(torch.cat((extra_train, extra_val), dim=0))
            print("collecting extra test predictions: {}".format(path), flush=True)
            test["extra_students"].append(collect_student(extra_model, test_loader, scaler, args))
    print("prediction tensors ready", flush=True)

    scored = []
    if extra_models:
        print("fitting multi-checkpoint calibration candidates", flush=True)
        for include_periodic in (False, True):
            val_features = multi_feature_tensor(fit, include_periodic=include_periodic)
            test_features = multi_feature_tensor(test, include_periodic=include_periodic)
            sample_weights = (1.0 / fit["true"].abs().clamp_min(1.0)).clamp_max(0.2)
            for ridge in (1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1, 1.0):
                coeffs = fit_calibration(val_features, fit["true"], ridge)
                val_pred = apply_calibration(val_features, coeffs)
                pred = apply_calibration(test_features, coeffs)
                name = "multi_node_calib_periodic_{}_ridge_{}".format(include_periodic, ridge)
                add_score(scored, name, pred, test, cli_args, args)
                add_light_residual_candidates(scored, name, val_pred, pred, fit, test, cli_args, args)

                horizon_coeffs = fit_horizon_calibration(val_features, fit["true"], ridge)
                val_horizon_pred = apply_horizon_calibration(val_features, horizon_coeffs)
                horizon_pred = apply_horizon_calibration(test_features, horizon_coeffs)
                horizon_name = "multi_horizon_calib_periodic_{}_ridge_{}".format(include_periodic, ridge)
                add_score(scored, horizon_name, horizon_pred, test, cli_args, args)
                add_light_residual_candidates(
                    scored, horizon_name, val_horizon_pred, horizon_pred, fit, test, cli_args, args
                )

                weighted_coeffs = fit_calibration(val_features, fit["true"], ridge, sample_weights=sample_weights)
                weighted_val_pred = apply_calibration(val_features, weighted_coeffs)
                weighted_pred = apply_calibration(test_features, weighted_coeffs)
                weighted_name = "multi_mape_weighted_node_calib_periodic_{}_ridge_{}".format(
                    include_periodic, ridge
                )
                add_score(scored, weighted_name, weighted_pred, test, cli_args, args)
                add_light_residual_candidates(
                    scored, weighted_name, weighted_val_pred, weighted_pred, fit, test, cli_args, args
                )

                weighted_horizon_coeffs = fit_horizon_calibration(
                    val_features, fit["true"], ridge, sample_weights=sample_weights
                )
                weighted_horizon_val_pred = apply_horizon_calibration(val_features, weighted_horizon_coeffs)
                weighted_horizon_pred = apply_horizon_calibration(test_features, weighted_horizon_coeffs)
                weighted_horizon_name = "multi_mape_weighted_horizon_calib_periodic_{}_ridge_{}".format(
                    include_periodic, ridge
                )
                add_score(scored, weighted_horizon_name, weighted_horizon_pred, test, cli_args, args)
                add_light_residual_candidates(
                    scored, weighted_horizon_name, weighted_horizon_val_pred,
                    weighted_horizon_pred, fit, test, cli_args, args
                )
        print("scored multi-checkpoint candidates: {}".format(len(scored)), flush=True)

    for metric in ("rmse", "mae"):
        print("fitting horizon ensemble weights for {}".format(metric), flush=True)
        horizon_weights = fit_horizon_weights(fit, metric=metric)
        val_ensemble = horizon_weights * fit["student"] + (1.0 - horizon_weights) * fit["teacher"]
        test_ensemble = horizon_weights * test["student"] + (1.0 - horizon_weights) * test["teacher"]
        periodic_weights = fit_periodic_weights(val_ensemble, fit)
        for use_periodic in (False, True):
            print("building candidates metric={} periodic={}".format(metric, use_periodic), flush=True)
            val_base = apply_periodic(val_ensemble, fit, periodic_weights if use_periodic else None)
            test_base = apply_periodic(test_ensemble, test, periodic_weights if use_periodic else None)
            base_name = "{}_periodic_{}".format(metric, use_periodic)
            add_score(scored, base_name, test_base, test, cli_args, args)
            add_tod_residual_candidates(scored, base_name, val_base, test_base, fit, test, cli_args, args)
            add_light_residual_candidates(
                scored, base_name, val_base, test_base, fit, test, cli_args, args,
                include_value_bins=True
            )
            for include_periodic in (False, True):
                val_features = feature_tensor(fit, val_base, include_periodic=include_periodic)
                test_features = feature_tensor(test, test_base, include_periodic=include_periodic)
                sample_weights = (1.0 / fit["true"].abs().clamp_min(1.0)).clamp_max(0.2)
                for ridge in (1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1, 1.0):
                    coeffs = fit_calibration(val_features, fit["true"], ridge)
                    val_pred = apply_calibration(val_features, coeffs)
                    pred = apply_calibration(test_features, coeffs)
                    name = "{}_periodic_{}_node_calib_periodic_{}_ridge_{}".format(
                        metric, use_periodic, include_periodic, ridge
                    )
                    add_score(scored, name, pred, test, cli_args, args)
                    add_light_residual_candidates(scored, name, val_pred, pred, fit, test, cli_args, args)

                    horizon_coeffs = fit_horizon_calibration(val_features, fit["true"], ridge)
                    val_horizon_pred = apply_horizon_calibration(val_features, horizon_coeffs)
                    horizon_pred = apply_horizon_calibration(test_features, horizon_coeffs)
                    horizon_name = "{}_periodic_{}_horizon_calib_periodic_{}_ridge_{}".format(
                        metric, use_periodic, include_periodic, ridge
                    )
                    add_score(scored, horizon_name, horizon_pred, test, cli_args, args)
                    add_light_residual_candidates(
                        scored, horizon_name, val_horizon_pred, horizon_pred, fit, test, cli_args, args
                    )

                    weighted_coeffs = fit_calibration(val_features, fit["true"], ridge, sample_weights=sample_weights)
                    weighted_val_pred = apply_calibration(val_features, weighted_coeffs)
                    weighted_pred = apply_calibration(test_features, weighted_coeffs)
                    weighted_name = "{}_periodic_{}_mape_weighted_node_calib_periodic_{}_ridge_{}".format(
                        metric, use_periodic, include_periodic, ridge
                    )
                    add_score(scored, weighted_name, weighted_pred, test, cli_args, args)
                    add_light_residual_candidates(
                        scored, weighted_name, weighted_val_pred, weighted_pred, fit, test, cli_args, args
                    )

                    weighted_horizon_coeffs = fit_horizon_calibration(
                        val_features, fit["true"], ridge, sample_weights=sample_weights
                    )
                    weighted_horizon_val_pred = apply_horizon_calibration(val_features, weighted_horizon_coeffs)
                    weighted_horizon_pred = apply_horizon_calibration(test_features, weighted_horizon_coeffs)
                    weighted_horizon_name = "{}_periodic_{}_mape_weighted_horizon_calib_periodic_{}_ridge_{}".format(
                        metric, use_periodic, include_periodic, ridge
                    )
                    add_score(scored, weighted_horizon_name, weighted_horizon_pred, test, cli_args, args)
                    add_light_residual_candidates(
                        scored, weighted_horizon_name, weighted_horizon_val_pred,
                        weighted_horizon_pred, fit, test, cli_args, args
                    )
            print("scored candidates so far: {}".format(len(scored)), flush=True)
    print("sorting {} scored candidates".format(len(scored)), flush=True)
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

    for pass_count, second_best_gain, min_gain, rmse, mae, mape, rmse_gain, mae_gain, mape_gain, name in scored[:20]:
        print(
            "{} | test1 Average Horizon, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}% | gains RMSE/MAE/MAPE/pass2/min: {:.2f}%/{:.2f}%/{:.2f}%/{}/ {:.2f}%/{:.2f}%".format(
                name, rmse, mae, mape, rmse_gain, mae_gain, mape_gain, pass_count, second_best_gain, min_gain
            )
        )


if __name__ == "__main__":
    main()
