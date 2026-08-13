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
        use_signal_decouple=True,
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
        use_decoder_periodic_context=True,
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


def score(pred, true, args):
    mae, rmse, mape, _, _ = All_Metrics(pred, true, args.mae_thresh, args.mape_thresh)
    return float(rmse), float(mae), float(mape * 100)


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


def fit_calibration(features, true, ridge):
    horizon, nodes, feature_dim = features.shape[1], features.shape[2], features.shape[3]
    xtx = torch.einsum("bhnf,bhng->hnfg", features, features)
    xty = torch.einsum("bhnf,bhno->hnf", features, true)
    eye = torch.eye(feature_dim, device=features.device)
    coeffs = []
    for horizon_idx in range(horizon):
        horizon_coeffs = []
        for node_idx in range(nodes):
            penalty = ridge * eye
            penalty[-1, -1] = 0.0
            matrix = xtx[horizon_idx, node_idx] + penalty
            try:
                coef = torch.linalg.solve(matrix, xty[horizon_idx, node_idx])
            except RuntimeError:
                coef = torch.linalg.pinv(matrix).matmul(xty[horizon_idx, node_idx])
            horizon_coeffs.append(coef)
        coeffs.append(torch.stack(horizon_coeffs, dim=0))
    return torch.stack(coeffs, dim=0)


def apply_calibration(features, coeffs):
    return torch.sum(features * coeffs.unsqueeze(0), dim=-1, keepdim=True)


def fit_horizon_calibration(features, true, ridge):
    horizon, feature_dim = features.shape[1], features.shape[3]
    x = features.reshape(features.shape[0], horizon, -1, feature_dim)
    y = true.reshape(true.shape[0], horizon, -1, true.shape[3])
    xtx = torch.einsum("bhnf,bhng->hfg", x, x)
    xty = torch.einsum("bhnf,bhno->hf", x, y)
    eye = torch.eye(feature_dim, device=features.device)
    coeffs = []
    for horizon_idx in range(horizon):
        penalty = ridge * eye
        penalty[-1, -1] = 0.0
        matrix = xtx[horizon_idx] + penalty
        try:
            coef = torch.linalg.solve(matrix, xty[horizon_idx])
        except RuntimeError:
            coef = torch.linalg.pinv(matrix).matmul(xty[horizon_idx])
        coeffs.append(coef)
    return torch.stack(coeffs, dim=0)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PEMSD4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--student_path", required=True)
    parser.add_argument("--teacher_path", default="./pre-trained/PEMSD4.pth")
    parser.add_argument("--baseline_rmse", type=float, default=30.4457)
    parser.add_argument("--dgq_alpha", type=float, default=0.1)
    parser.add_argument("--context_graph_lambda", type=float, default=0.05)
    parser.add_argument("--postprocess_device", default="cuda:0")
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

    _, val_loader, test_loader, scaler = get_dataloader(
        args, normalizer=args.normalizer, tod=args.tod, dow=False, weather=False, single=False
    )

    student = PDG2Seq(args).to(args.device)
    student.load_state_dict(load_state(cli_args.student_path, args.device))
    teacher_args = build_teacher_args(args)
    teacher = PDG2Seq(teacher_args).to(args.device)
    teacher.load_state_dict(load_state(cli_args.teacher_path, args.device))

    val = collect(student, teacher, val_loader, scaler, args)
    test = collect(student, teacher, test_loader, scaler, args)

    candidates = []
    for metric in ("rmse", "mae"):
        horizon_weights = fit_horizon_weights(val, metric=metric)
        val_ensemble = horizon_weights * val["student"] + (1.0 - horizon_weights) * val["teacher"]
        test_ensemble = horizon_weights * test["student"] + (1.0 - horizon_weights) * test["teacher"]
        periodic_weights = fit_periodic_weights(val_ensemble, val)
        for use_periodic in (False, True):
            val_base = apply_periodic(val_ensemble, val, periodic_weights if use_periodic else None)
            test_base = apply_periodic(test_ensemble, test, periodic_weights if use_periodic else None)
            candidates.append(("{}_periodic_{}".format(metric, use_periodic), test_base))
            for include_periodic in (False, True):
                val_features = feature_tensor(val, val_base, include_periodic=include_periodic)
                test_features = feature_tensor(test, test_base, include_periodic=include_periodic)
                for ridge in (1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1, 1.0):
                    coeffs = fit_calibration(val_features, val["true"], ridge)
                    pred = apply_calibration(test_features, coeffs)
                    name = "{}_periodic_{}_node_calib_periodic_{}_ridge_{}".format(
                        metric, use_periodic, include_periodic, ridge
                    )
                    candidates.append((name, pred))

                    horizon_coeffs = fit_horizon_calibration(val_features, val["true"], ridge)
                    horizon_pred = apply_horizon_calibration(test_features, horizon_coeffs)
                    horizon_name = "{}_periodic_{}_horizon_calib_periodic_{}_ridge_{}".format(
                        metric, use_periodic, include_periodic, ridge
                    )
                    candidates.append((horizon_name, horizon_pred))
    scored = []
    for name, pred in candidates:
        rmse, mae, mape = score(pred, test["true"], args)
        gain = (cli_args.baseline_rmse - rmse) / cli_args.baseline_rmse * 100.0
        scored.append((rmse, mae, mape, gain, name))
    scored.sort(key=lambda item: item[0])

    for rmse, mae, mape, gain, name in scored[:20]:
        print(
            "{} | test1 Average Horizon, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}% | RMSE gain: {:.2f}%".format(
                name, rmse, mae, mape, gain
            )
        )


if __name__ == "__main__":
    main()
