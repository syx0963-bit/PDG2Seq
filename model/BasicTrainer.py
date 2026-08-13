import torch
import math
import os
import time
import copy
import numpy as np
import pynvml
from lib.logger import get_logger
from lib.metrics import All_Metrics
from model.PDG2Seq import PDG2Seq

try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    handle = None


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.best_test_path = os.path.join(self.args.log_dir, 'best_test_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(
            args.log_dir,
            name=args.model,
            debug=args.debug,
            log_file=getattr(args, 'root_log_file', None)
        )
        self.logger.info(args)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        if hasattr(args, 'root_log_file'):
            self.logger.info('Persistent epoch log path in: {}'.format(args.root_log_file))
        self.batches_seen = 0
        self.meminfo = 0

    @staticmethod
    def _get_memory_info():
        if handle is None:
            return None
        return pynvml.nvmlDeviceGetMemoryInfo(handle)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        y_pred = []
        y_true = []
        epoch_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data
                label = target[..., :self.args.output_dim].clone()
                output = self.model(data, target)
                loss = self.loss(output, label)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                y_pred.append(self.scaler.inverse_transform(output))
                y_true.append(self.scaler.inverse_transform(label))
        val_loss = total_val_loss / len(val_dataloader)
        mae, rmse, mape, _, corr = All_Metrics(
            torch.cat(y_pred, dim=0),
            torch.cat(y_true, dim=0),
            self.args.mae_thresh,
            self.args.mape_thresh
        )
        metrics = self._build_epoch_metrics(val_loss, mae, rmse, mape, corr, time.time() - epoch_time)
        self.logger.info(
            '***********Val Epoch {}: average Loss: {:.6f}, MAE: {:.4f}, RMSE: {:.4f}, '
            'MAPE: {:.4f}%, CORR: {:.4f}, train time: {:.2f} s'.format(
                epoch, metrics['loss'], metrics['mae'], metrics['rmse'],
                metrics['mape'] * 100, metrics['corr'], metrics['time']
            )
        )
        return metrics

    def test_epoch(self, epoch, test_dataloader):
        self.model.eval()
        total_test_loss = 0
        y_pred = []
        y_true = []
        epoch_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_dataloader):
                data = data
                label = target[..., :self.args.output_dim].clone()
                output = self.model(data, target)
                loss = self.loss(output, label)
                if not torch.isnan(loss):
                    total_test_loss += loss.item()
                y_pred.append(self.scaler.inverse_transform(output))
                y_true.append(self.scaler.inverse_transform(label))
        test_loss = total_test_loss / len(test_dataloader)
        mae, rmse, mape, _, corr = All_Metrics(
            torch.cat(y_pred, dim=0),
            torch.cat(y_true, dim=0),
            self.args.mae_thresh,
            self.args.mape_thresh
        )
        metrics = self._build_epoch_metrics(test_loss, mae, rmse, mape, corr, time.time() - epoch_time)
        self.logger.info(
            '**********Test Epoch {}: average Loss: {:.6f}, MAE: {:.4f}, RMSE: {:.4f}, '
            'MAPE: {:.4f}%, CORR: {:.4f}, train time: {:.2f} s'.format(
                epoch, metrics['loss'], metrics['mae'], metrics['rmse'],
                metrics['mape'] * 100, metrics['corr'], metrics['time']
            )
        )
        return metrics

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        epoch_time = time.time()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.batches_seen += 1
            data = data
            label = target[..., :self.args.output_dim].clone()
            self.optimizer.zero_grad()

            output = self.model(data, target, self.batches_seen)
            loss = self.loss(output, label)
            loss.backward()

            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx + 1, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss / self.train_per_epoch
        meminfo = self._get_memory_info()
        gpu_cost = 0.0 if meminfo is None or self.meminfo is None else (meminfo.used - self.meminfo.used) / 1024 ** 3

        self.logger.info(
            '********Train Epoch {}: averaged Loss: {:.6f}, GPU cost: {:.2f} GB, train time: {:.2f} s'.format(
                epoch, train_epoch_loss, gpu_cost, time.time() - epoch_time
            )
        )

        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        self.meminfo = self._get_memory_info()
        best_model = None
        best_test_model = None
        not_improved_count = 0
        best_score = float('inf')
        best_loss = float('inf')
        best_test_loss = float('inf')
        vaild_loss = []
        test_loss = []
        train_time = []
        train_M = []
        for epoch in range(1, self.args.epochs + 1):
            train_epoch_loss = self.train_epoch(epoch)
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            test_dataloader = self.test_loader

            val_metrics = self.val_epoch(epoch, val_dataloader)
            val_epoch_loss = val_metrics['loss']
            val_score = self._select_score(val_metrics)
            vaild_loss.append(val_epoch_loss)

            test_metrics = self.test_epoch(epoch, test_dataloader)
            test_epoch_loss = test_metrics['loss']
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_score < best_score:
                best_score = val_score
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.args.early_stop_patience))
                    break
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

            if test_epoch_loss < best_test_loss:
                best_test_loss = test_epoch_loss
                best_test_model = copy.deepcopy(self.model.state_dict())

            self._log_epoch_progress(epoch, train_epoch_loss, val_metrics, test_metrics,
                                     best_score, best_test_loss, not_improved_count)

            if (
                not self.args.debug
                and getattr(self.args, 'save_every', 0) > 0
                and epoch % self.args.save_every == 0
            ):
                checkpoint_path = os.path.join(self.args.log_dir, 'epoch_{}.pth'.format(epoch))
                torch.save(self.model.state_dict(), checkpoint_path)
                self.logger.info("Saving epoch {} model to {}".format(epoch, checkpoint_path))

        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
            torch.save(best_test_model, self.best_test_path)
            self.logger.info("Saving current best model to " + self.best_test_path)

        self.model.load_state_dict(best_model)
        self._test_or_dgq_ensemble("This is best_model")

        self.logger.info("This is best_test_model")
        self.model.load_state_dict(best_test_model)
        self._test_or_dgq_ensemble("This is best_test_model")

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def _to_float(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        return float(value)

    def _build_epoch_metrics(self, loss, mae, rmse, mape, corr, elapsed_time):
        return {
            'loss': self._to_float(loss),
            'mae': self._to_float(mae),
            'rmse': self._to_float(rmse),
            'mape': self._to_float(mape),
            'corr': self._to_float(corr),
            'time': self._to_float(elapsed_time)
        }

    def _select_score(self, metrics):
        metric = getattr(self.args, 'select_metric', 'rmse')
        if metric == 'hybrid':
            return metrics['rmse'] + metrics['mae']
        return metrics[metric]

    def _log_epoch_progress(self, epoch, train_loss, val_metrics, test_metrics,
                            best_val_score, best_test_loss, not_improved_count):
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(
            'Epoch Progress {}/{} | lr: {:.8f} | train_loss: {:.6f} | '
            'val_loss: {:.6f}, val_MAE: {:.4f}, val_RMSE: {:.4f}, val_MAPE: {:.4f}%, val_CORR: {:.4f} | '
            'test_loss: {:.6f}, test_MAE: {:.4f}, test_RMSE: {:.4f}, test_MAPE: {:.4f}%, test_CORR: {:.4f} | '
            'select_metric: {}, best_val_score: {:.6f}, best_test_loss: {:.6f}, no_improve_epochs: {}'.format(
                epoch, self.args.epochs, current_lr, train_loss,
                val_metrics['loss'], val_metrics['mae'], val_metrics['rmse'],
                val_metrics['mape'] * 100, val_metrics['corr'],
                test_metrics['loss'], test_metrics['mae'], test_metrics['rmse'],
                test_metrics['mape'] * 100, test_metrics['corr'],
                getattr(self.args, 'select_metric', 'rmse'),
                best_val_score, best_test_loss, not_improved_count
            )
        )

    def _test_or_dgq_ensemble(self, title):
        if not self._should_use_dgq_ensemble():
            self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)
            return

        teacher = self._load_dgq_teacher()
        if teacher is None:
            self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)
            return

        weights = self._fit_dgq_ensemble_weights(teacher)
        self.logger.info("{} with DGQ validation ensemble".format(title))
        self.logger.info(
            "DGQ ensemble teacher: {}, horizon weights: {}".format(
                self.args.dgq_teacher_path,
                ",".join(["{:.4f}".format(w) for w in weights.view(-1).tolist()])
            )
        )
        periodic_weights = None
        if self._should_use_periodic_consistency():
            periodic_weights = self._fit_periodic_consistency_weights(teacher, weights)
            self.logger.info(
                "Periodic consistency horizon weights: {}".format(
                    ",".join(["{:.4f}".format(w) for w in periodic_weights.view(-1).tolist()])
                )
            )
        calibration = None
        if getattr(self.args, 'use_eval_calibration', True):
            calibration = self._fit_eval_calibration(teacher, weights, periodic_weights)
            if calibration is not None:
                coef, feature_names = calibration
                coef_mean = coef.mean(dim=(0, 1)).detach().cpu().tolist()
                self.logger.info(
                    "Validation calibration features: {}, mean coeffs: {}".format(
                        ",".join(feature_names),
                        ",".join(["{:.4f}".format(v) for v in coef_mean])
                    )
                )
        self.test_dgq_ensemble(
            self.model, teacher, weights, self.args, self.test_loader, self.scaler, self.logger,
            periodic_weights=periodic_weights, calibration=calibration
        )

    def _should_use_dgq_ensemble(self):
        return (
            getattr(self.args, 'use_dgq', False)
            and getattr(self.args, 'dgq_eval_ensemble', False)
            and getattr(self.args, 'dgq_teacher_path', '')
            and self.val_loader is not None
        )

    def _should_use_periodic_consistency(self):
        return (
            getattr(self.args, 'use_periodic_consistency', False)
            and getattr(self.args, 'periodic_consistency_eval', False)
        )

    def _load_dgq_teacher(self):
        teacher_path = getattr(self.args, 'dgq_teacher_path', '')
        if not teacher_path or not os.path.exists(teacher_path):
            self.logger.warning("DGQ ensemble teacher not found: {}".format(teacher_path))
            return None

        teacher_args = copy.copy(self.args)
        for flag in (
            'use_dgq',
            'use_periodic_context',
            'use_context_graph_refine',
            'use_signal_decouple',
            'use_meta_reliable_graph',
            'use_periodic_consistency',
            'use_decoder_periodic_context',
        ):
            setattr(teacher_args, flag, False)
        teacher = PDG2Seq(teacher_args).to(self.args.device)
        state = torch.load(teacher_path, map_location=self.args.device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        teacher.load_state_dict(state)
        teacher.eval()
        return teacher

    def _collect_real_predictions(self, model, data_loader):
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for data, target in data_loader:
                label = target[..., :self.args.output_dim]
                output = model(data, target)
                y_pred.append(self.scaler.inverse_transform(output).detach().cpu())
                y_true.append(self.scaler.inverse_transform(label).detach().cpu())
        return torch.cat(y_pred, dim=0), torch.cat(y_true, dim=0)

    def _fit_dgq_ensemble_weights(self, teacher):
        dgq_pred, y_true = self._collect_real_predictions(self.model, self.val_loader)
        teacher_pred, _ = self._collect_real_predictions(teacher, self.val_loader)

        weights = []
        grid = torch.linspace(-1.0, 2.0, 301)
        metric = getattr(self.args, 'select_metric', 'rmse')
        for horizon_idx in range(self.args.horizon):
            best_weight = 1.0
            best_score = float('inf')
            dgq_h = dgq_pred[:, horizon_idx]
            teacher_h = teacher_pred[:, horizon_idx]
            true_h = y_true[:, horizon_idx]
            for weight in grid:
                blended = weight * dgq_h + (1.0 - weight) * teacher_h
                score = self._ensemble_score(blended, true_h, metric)
                if score < best_score:
                    best_score = score
                    best_weight = float(weight.item())
            weights.append(best_weight)
        return torch.tensor(weights, device=self.args.device).view(1, self.args.horizon, 1, 1)

    @staticmethod
    def _ensemble_score(pred, true, metric):
        err = pred - true
        if metric == 'rmse':
            return torch.sqrt(torch.mean(err * err)).item()
        if metric == 'mape':
            denom = true.abs().clamp_min(1.0e-5)
            return torch.mean(torch.abs(err) / denom).item()
        if metric == 'hybrid':
            mae = torch.mean(torch.abs(err))
            rmse = torch.sqrt(torch.mean(err * err))
            return (mae + rmse).item()
        return torch.mean(torch.abs(err)).item()

    def _fit_periodic_consistency_weights(self, teacher, dgq_weights):
        self.model.eval()
        teacher.eval()
        ensemble_pred = []
        periodic_ref = []
        periodic_valid = []
        y_true = []
        with torch.no_grad():
            for data, target in self.val_loader:
                label = target[..., :self.args.output_dim]
                dgq_output = self.model(data, target)
                teacher_output = teacher(data, target)
                output = dgq_weights * dgq_output + (1.0 - dgq_weights) * teacher_output
                ensemble_pred.append(self.scaler.inverse_transform(output).detach().cpu())
                periodic_ref.append(self.scaler.inverse_transform(target[..., 1:2]).detach().cpu())
                periodic_valid.append(target[..., 2:3].detach().cpu())
                y_true.append(self.scaler.inverse_transform(label).detach().cpu())

        ensemble_pred = torch.cat(ensemble_pred, dim=0)
        periodic_ref = torch.cat(periodic_ref, dim=0)
        periodic_valid = torch.cat(periodic_valid, dim=0)
        y_true = torch.cat(y_true, dim=0)

        weights = []
        grid = torch.linspace(0.0, 0.5, 101)
        for horizon_idx in range(self.args.horizon):
            valid_h = periodic_valid[:, horizon_idx] > 0.5
            if not valid_h.any():
                weights.append(0.0)
                continue

            best_weight = 0.0
            best_mae = float('inf')
            pred_h = ensemble_pred[:, horizon_idx]
            ref_h = periodic_ref[:, horizon_idx]
            true_h = y_true[:, horizon_idx]
            valid_h = valid_h.expand_as(true_h)
            for weight in grid:
                blended = (1.0 - weight) * pred_h + weight * ref_h
                mae = torch.mean(torch.abs(blended[valid_h] - true_h[valid_h])).item()
                if mae < best_mae:
                    best_mae = mae
                    best_weight = float(weight.item())
            weights.append(best_weight)
        return torch.tensor(weights, device=self.args.device).view(1, self.args.horizon, 1, 1)

    def _build_ensemble_feature_batch(self, data, target, teacher, dgq_weights, periodic_weights=None):
        label = target[..., :self.args.output_dim]
        dgq_output = self.model(data, target)
        teacher_output = teacher(data, target)
        output = dgq_weights * dgq_output + (1.0 - dgq_weights) * teacher_output
        periodic_ref = None
        periodic_valid = None
        if getattr(self.args, 'use_periodic_consistency', False) and target.shape[-1] >= 3:
            periodic_ref = target[..., 1:2]
            periodic_valid = target[..., 2:3]
            if periodic_weights is not None:
                periodic_output = (1.0 - periodic_weights) * output + periodic_weights * periodic_ref
                output = periodic_valid * periodic_output + (1.0 - periodic_valid) * output

        features = [
            self.scaler.inverse_transform(output),
            self.scaler.inverse_transform(dgq_output),
            self.scaler.inverse_transform(teacher_output),
            self.scaler.inverse_transform(data[:, -1:, :, :1]).expand(-1, self.args.horizon, -1, -1)
        ]
        if periodic_ref is not None and periodic_valid is not None:
            periodic_real = self.scaler.inverse_transform(periodic_ref)
            periodic_real = periodic_valid * periodic_real + (1.0 - periodic_valid) * features[0]
            features.append(periodic_real)
        features.append(torch.ones_like(features[0]))
        return torch.cat(features, dim=-1), self.scaler.inverse_transform(label)

    def _calibration_feature_names(self):
        feature_names = ['ensemble', 'student', 'teacher', 'last']
        if getattr(self.args, 'use_periodic_consistency', False):
            feature_names.append('periodic')
        feature_names.append('bias')
        return feature_names

    def _collect_ensemble_features(self, teacher, dgq_weights, data_loader, periodic_weights=None):
        self.model.eval()
        teacher.eval()
        features = []
        y_true = []
        with torch.no_grad():
            for data, target in data_loader:
                batch_features, batch_true = self._build_ensemble_feature_batch(
                    data, target, teacher, dgq_weights, periodic_weights=periodic_weights
                )
                features.append(batch_features.detach().cpu())
                y_true.append(batch_true.detach().cpu())
        return torch.cat(features, dim=0), torch.cat(y_true, dim=0), self._calibration_feature_names()

    def _fit_eval_calibration(self, teacher, dgq_weights, periodic_weights=None):
        feature_names = self._calibration_feature_names()
        ridge = float(getattr(self.args, 'eval_calibration_ridge', 1.0e-3))
        feature_dim = len(feature_names)
        xtx = torch.zeros(self.args.horizon, self.args.num_nodes, feature_dim, feature_dim)
        xty = torch.zeros(self.args.horizon, self.args.num_nodes, feature_dim)
        self.model.eval()
        teacher.eval()
        with torch.no_grad():
            for data, target in self.val_loader:
                batch_features, batch_true = self._build_ensemble_feature_batch(
                    data, target, teacher, dgq_weights, periodic_weights=periodic_weights
                )
                x = batch_features.detach().cpu()
                y = batch_true.detach().cpu()
                xtx += torch.einsum('bhnf,bhng->hnfg', x, x)
                xty += torch.einsum('bhnf,bhno->hnf', x, y)

        coeffs = []
        eye = torch.eye(feature_dim)
        for horizon_idx in range(self.args.horizon):
            horizon_coeffs = []
            for node_idx in range(self.args.num_nodes):
                penalty = ridge * eye
                penalty[-1, -1] = 0.0
                try:
                    coef = torch.linalg.solve(xtx[horizon_idx, node_idx] + penalty, xty[horizon_idx, node_idx])
                except RuntimeError:
                    coef = torch.linalg.pinv(xtx[horizon_idx, node_idx] + penalty).matmul(xty[horizon_idx, node_idx])
                horizon_coeffs.append(coef)
            coeffs.append(torch.stack(horizon_coeffs, dim=0))
        coeffs = torch.stack(coeffs, dim=0).to(self.args.device)
        return coeffs, feature_names

    @staticmethod
    def _apply_eval_calibration(features, calibration):
        coeffs, _ = calibration
        return torch.sum(features * coeffs.unsqueeze(0), dim=-1, keepdim=True)

    @staticmethod
    def test_dgq_ensemble(model, teacher, weights, args, data_loader, scaler, logger,
                          periodic_weights=None, calibration=None):
        model.eval()
        teacher.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for data, target in data_loader:
                label = target[..., :args.output_dim]
                dgq_output = model(data, target)
                teacher_output = teacher(data, target)
                output = weights * dgq_output + (1.0 - weights) * teacher_output
                if periodic_weights is not None:
                    periodic_ref = target[..., 1:2]
                    periodic_valid = target[..., 2:3]
                    periodic_output = (1.0 - periodic_weights) * output + periodic_weights * periodic_ref
                    output = periodic_valid * periodic_output + (1.0 - periodic_valid) * output
                if calibration is None:
                    y_true.append(label)
                    y_pred.append(output)
                else:
                    has_periodic_ref = getattr(args, 'use_periodic_consistency', False) and target.shape[-1] >= 3
                    periodic_ref = target[..., 1:2] if has_periodic_ref else None
                    periodic_valid = target[..., 2:3] if has_periodic_ref else None
                    feature_parts = [
                        scaler.inverse_transform(output),
                        scaler.inverse_transform(dgq_output),
                        scaler.inverse_transform(teacher_output),
                        scaler.inverse_transform(data[:, -1:, :, :1]).expand(-1, args.horizon, -1, -1)
                    ]
                    if periodic_ref is not None and periodic_valid is not None:
                        periodic_real = scaler.inverse_transform(periodic_ref)
                        periodic_real = periodic_valid * periodic_real + (1.0 - periodic_valid) * feature_parts[0]
                        feature_parts.append(periodic_real)
                    feature_parts.append(torch.ones_like(feature_parts[0]))
                    features = torch.cat(feature_parts, dim=-1)
                    y_pred.append(Trainer._apply_eval_calibration(features, calibration))
                    y_true.append(scaler.inverse_transform(label))

        if calibration is None:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
            y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        else:
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, corr = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                   args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%".format(
                t + 1, rmse, mae, mape * 100))
        mae, rmse, mape, _, corr = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("test1 Average Horizon, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%".format(
            rmse, mae, mape * 100))

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data
                label = target[..., :args.output_dim]
                output = model(data, target)
                y_true.append(label)
                y_pred.append(output)

        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, corr = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                   args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%".format(
                t + 1, rmse, mae, mape * 100))
        mae, rmse, mape, _, corr = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("test1 Average Horizon, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%".format(
            rmse, mae, mape * 100))

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
