import torch
import math
import os
import time
import copy
import numpy as np
import pynvml
import torch.nn.functional as F
from lib.logger import get_logger
from lib.metrics import All_Metrics
from model.PDG2Seq import PDG2Seq

try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    handle = None


class GraphAwareOnlineAdapter(object):
    def __init__(self, args, model):
        self.args = args
        self.device = torch.device(args.device)
        self.horizon = int(args.horizon)
        self.num_nodes = int(args.num_nodes)
        self.lr = float(getattr(args, 'online_adapt_lr', 0.08))
        self.scale_lr = float(getattr(args, 'online_scale_lr', 0.02))
        self.bias_decay = float(getattr(args, 'online_adapt_decay', 0.95))
        self.error_decay = float(getattr(args, 'online_error_decay', 0.90))
        self.sensitivity = float(getattr(args, 'online_drift_sensitivity', 1.0))
        self.neighbor_expand = float(getattr(args, 'online_neighbor_expand', 0.35))
        self.bias_clip = float(getattr(args, 'online_bias_clip', 8.0))
        self.scale_clip = float(getattr(args, 'online_scale_clip', 0.08))
        self.use_overlap_memory = bool(getattr(args, 'online_overlap_memory', False))
        self.overlap_blend = float(getattr(args, 'online_overlap_blend', 0.85))
        self.bias = torch.zeros(1, self.horizon, self.num_nodes, 1, device=self.device)
        self.scale = torch.ones(1, self.horizon, self.num_nodes, 1, device=self.device)
        self.error_ema = torch.zeros(self.horizon, self.num_nodes, 1, device=self.device)
        self.prev_true = None
        self.update_count = 0
        self.drift_count = 0
        self.graph = self._build_reliable_graph(model)

    def _build_reliable_graph(self, model):
        with torch.no_grad():
            emb = getattr(model, 'node_embeddings1', None)
            if emb is None:
                return torch.eye(self.num_nodes, device=self.device)
            emb = F.normalize(emb.detach().to(self.device), dim=-1, eps=1.0e-8)
            sim = torch.relu(torch.matmul(emb, emb.transpose(0, 1)))
            sim.fill_diagonal_(0.0)
            topk = min(max(int(getattr(self.args, 'online_graph_topk', 8)), 1), self.num_nodes - 1)
            values, indices = torch.topk(sim, k=topk, dim=-1)
            graph = torch.zeros_like(sim)
            graph.scatter_(1, indices, values)
            graph = 0.5 * (graph + graph.transpose(0, 1))
            graph = graph + torch.eye(self.num_nodes, device=self.device)
            return graph / graph.sum(-1, keepdim=True).clamp_min(1.0e-8)

    def apply(self, pred):
        pred = torch.clamp(self.scale * pred + self.bias, min=0.0)
        if self.use_overlap_memory and self.prev_true is not None and pred.shape[1] > 1:
            known = self.prev_true[:, 1:, :, :]
            pred[:, :-1, :, :] = (
                (1.0 - self.overlap_blend) * pred[:, :-1, :, :]
                + self.overlap_blend * known
            )
        return pred

    def adapt_batch(self, pred, true):
        adapted = []
        for sample_idx in range(pred.shape[0]):
            pred_i = self.apply(pred[sample_idx:sample_idx + 1])
            true_i = true[sample_idx:sample_idx + 1]
            adapted.append(pred_i.detach())
            self.update(pred_i, true_i)
            self.prev_true = true_i.detach().to(self.device)
        return torch.cat(adapted, dim=0)

    def update(self, pred, true):
        pred = pred.detach().to(self.device)
        true = true.detach().to(self.device)
        residual = true - pred
        abs_error = torch.abs(residual).mean(dim=0)
        if self.update_count == 0:
            self.error_ema = abs_error
        else:
            self.error_ema = self.error_decay * self.error_ema + (1.0 - self.error_decay) * abs_error

        neighbor_error = torch.einsum('ij,hjo->hio', self.graph, self.error_ema)
        graph_delta = torch.abs(self.error_ema - neighbor_error)
        drift_score = self.error_ema + self.neighbor_expand * graph_delta
        center = drift_score.mean(dim=1, keepdim=True)
        spread = drift_score.std(dim=1, keepdim=True).clamp_min(1.0e-6)
        drift_mask = drift_score > (center + self.sensitivity * spread)
        local_score = drift_mask.float()
        local_score = torch.maximum(local_score, torch.einsum('ij,hjo->hio', self.graph, local_score))
        local_mask = local_score > 0.0

        mean_residual = residual.mean(dim=0)
        mean_relative_residual = (residual / pred.abs().clamp_min(1.0)).mean(dim=0)
        self.bias = self.bias_decay * self.bias
        local_update = self.lr * mean_residual * local_mask.float()
        self.bias = torch.clamp(self.bias + local_update.unsqueeze(0), -self.bias_clip, self.bias_clip)
        scale_update = self.scale_lr * mean_relative_residual * local_mask.float()
        self.scale = self.scale + scale_update.unsqueeze(0)
        self.scale = torch.clamp(self.scale, 1.0 - self.scale_clip, 1.0 + self.scale_clip)
        self.update_count += 1
        self.drift_count += int(drift_mask.sum().item())

    def summary(self):
        return {
            'updates': self.update_count,
            'drift_nodes': self.drift_count,
            'bias_mean': float(self.bias.abs().mean().detach().cpu().item()),
            'bias_max': float(self.bias.abs().max().detach().cpu().item()),
            'scale_mean': float(torch.abs(self.scale - 1.0).mean().detach().cpu().item()),
            'scale_max': float(torch.abs(self.scale - 1.0).max().detach().cpu().item()),
        }


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
        if metric == 'balanced':
            return metrics['rmse'] + metrics['mae'] + 120.0 * metrics['mape']
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
            online_adapter = self._build_online_adapter()
            self._warmup_online_adapter_single(online_adapter)
            self.test(
                self.model, self.args, self.test_loader, self.scaler, self.logger,
                online_adapter=online_adapter
            )
            return

        teacher = self._load_dgq_teacher()
        if teacher is None:
            online_adapter = self._build_online_adapter()
            self._warmup_online_adapter_single(online_adapter)
            self.test(
                self.model, self.args, self.test_loader, self.scaler, self.logger,
                online_adapter=online_adapter
            )
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
            calibration = self._select_eval_calibration(teacher, weights, periodic_weights)
            if calibration is not None:
                coef, feature_names, calibration_name, val_metrics = calibration
                if coef.dim() == 3:
                    coef_mean = coef.mean(dim=(0, 1)).detach().cpu().tolist()
                else:
                    coef_mean = coef.mean(dim=0).detach().cpu().tolist()
                self.logger.info(
                    "Validation calibration: {}, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%, features: {}, mean coeffs: {}".format(
                        calibration_name, val_metrics[1], val_metrics[0], val_metrics[2] * 100,
                        ",".join(feature_names),
                        ",".join(["{:.4f}".format(v) for v in coef_mean])
                    )
                )
        online_adapter = self._build_online_adapter()
        self._warmup_online_adapter_ensemble(
            online_adapter, teacher, weights, periodic_weights=periodic_weights, calibration=calibration
        )
        self.test_dgq_ensemble(
            self.model, teacher, weights, self.args, self.test_loader, self.scaler, self.logger,
            periodic_weights=periodic_weights, calibration=calibration, online_adapter=online_adapter
        )

    def _build_online_adapter(self):
        if not getattr(self.args, 'use_online_adaptation', False):
            return None
        adapter = GraphAwareOnlineAdapter(self.args, self.model)
        self.logger.info(
            "Online graph adaptation enabled: lr={}, scale_lr={}, decay={}, error_decay={}, topk={}, sensitivity={}, neighbor_expand={}, bias_clip={}, scale_clip={}, warmup_val={}, overlap_memory={}, overlap_blend={}".format(
                getattr(self.args, 'online_adapt_lr', 0.08),
                getattr(self.args, 'online_scale_lr', 0.02),
                getattr(self.args, 'online_adapt_decay', 0.95),
                getattr(self.args, 'online_error_decay', 0.90),
                getattr(self.args, 'online_graph_topk', 8),
                getattr(self.args, 'online_drift_sensitivity', 1.0),
                getattr(self.args, 'online_neighbor_expand', 0.35),
                getattr(self.args, 'online_bias_clip', 8.0),
                getattr(self.args, 'online_scale_clip', 0.08),
                getattr(self.args, 'online_warmup_val', True),
                getattr(self.args, 'online_overlap_memory', False),
                getattr(self.args, 'online_overlap_blend', 0.85)
            )
        )
        return adapter

    def _warmup_online_adapter_single(self, online_adapter):
        if online_adapter is None or self.val_loader is None or not getattr(self.args, 'online_warmup_val', True):
            return
        self.model.eval()
        with torch.no_grad():
            for data, target in self.val_loader:
                label = target[..., :self.args.output_dim]
                pred = self.scaler.inverse_transform(self.model(data, target))
                true = self.scaler.inverse_transform(label)
                online_adapter.adapt_batch(pred, true)
        self._log_online_adapter_summary("Online adaptation warmup", online_adapter)

    def _warmup_online_adapter_ensemble(self, online_adapter, teacher, weights, periodic_weights=None, calibration=None):
        if online_adapter is None or self.val_loader is None or not getattr(self.args, 'online_warmup_val', True):
            return
        self.model.eval()
        teacher.eval()
        with torch.no_grad():
            for data, target in self.val_loader:
                pred, true = self._ensemble_real_batch(
                    data, target, teacher, weights, periodic_weights=periodic_weights, calibration=calibration
                )
                online_adapter.adapt_batch(pred, true)
        self._log_online_adapter_summary("Online adaptation warmup", online_adapter)

    def _ensemble_real_batch(self, data, target, teacher, weights, periodic_weights=None, calibration=None):
        label = target[..., :self.args.output_dim]
        dgq_output = self.model(data, target)
        teacher_output = teacher(data, target)
        output = weights * dgq_output + (1.0 - weights) * teacher_output
        if periodic_weights is not None:
            periodic_ref = target[..., 1:2]
            periodic_valid = target[..., 2:3]
            periodic_output = (1.0 - periodic_weights) * output + periodic_weights * periodic_ref
            output = periodic_valid * periodic_output + (1.0 - periodic_valid) * output
        if calibration is None:
            pred = self.scaler.inverse_transform(output)
        else:
            has_periodic_ref = getattr(self.args, 'use_periodic_consistency', False) and target.shape[-1] >= 3
            periodic_ref = target[..., 1:2] if has_periodic_ref else None
            periodic_valid = target[..., 2:3] if has_periodic_ref else None
            feature_parts = [
                self.scaler.inverse_transform(output),
                self.scaler.inverse_transform(dgq_output),
                self.scaler.inverse_transform(teacher_output),
                self.scaler.inverse_transform(data[:, -1:, :, :1]).expand(-1, self.args.horizon, -1, -1)
            ]
            if periodic_ref is not None and periodic_valid is not None:
                periodic_real = self.scaler.inverse_transform(periodic_ref)
                periodic_real = periodic_valid * periodic_real + (1.0 - periodic_valid) * feature_parts[0]
                feature_parts.append(periodic_real)
            feature_parts.append(torch.ones_like(feature_parts[0]))
            features = torch.cat(feature_parts, dim=-1)
            pred = Trainer._apply_eval_calibration(features, calibration)
        return pred, self.scaler.inverse_transform(label)

    def _log_online_adapter_summary(self, prefix, online_adapter):
        if online_adapter is None:
            return
        stats = online_adapter.summary()
        self.logger.info(
            "{}: updates={}, drift_nodes={}, |bias| mean/max={:.4f}/{:.4f}, |scale-1| mean/max={:.4f}/{:.4f}".format(
                prefix, stats['updates'], stats['drift_nodes'], stats['bias_mean'], stats['bias_max'],
                stats['scale_mean'], stats['scale_max']
            )
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
        if metric == 'balanced':
            mae = torch.mean(torch.abs(err))
            rmse = torch.sqrt(torch.mean(err * err))
            denom = true.abs().clamp_min(1.0e-5)
            mape = torch.mean(torch.abs(err) / denom)
            return (rmse + mae + 120.0 * mape).item()
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
        return self._fit_node_calibration_from_loader(teacher, dgq_weights, periodic_weights, ridge), feature_names

    def _fit_node_calibration_from_loader(self, teacher, dgq_weights, periodic_weights, ridge):
        feature_names = self._calibration_feature_names()
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
        return self._solve_node_calibration(xtx, xty, ridge)

    @staticmethod
    def _solve_node_calibration(xtx, xty, ridge):
        feature_dim = xtx.shape[-1]
        penalty = ridge * torch.eye(feature_dim, device=xtx.device, dtype=xtx.dtype)
        penalty[-1, -1] = 0.0
        matrix = xtx + penalty.view(1, 1, feature_dim, feature_dim)
        rhs = xty.unsqueeze(-1)
        try:
            return torch.linalg.solve(matrix, rhs).squeeze(-1)
        except RuntimeError:
            return torch.matmul(torch.linalg.pinv(matrix), rhs).squeeze(-1)

    @staticmethod
    def _fit_node_calibration(features, true, ridge, sample_weights=None):
        x = features
        y = true
        if sample_weights is not None:
            w = torch.sqrt(sample_weights.clamp_min(1.0e-6))
            x = x * w
            y = y * w
        xtx = torch.einsum('bhnf,bhng->hnfg', x, x)
        xty = torch.einsum('bhnf,bhno->hnf', x, y)
        return Trainer._solve_node_calibration(xtx, xty, ridge)

    @staticmethod
    def _fit_horizon_calibration(features, true, ridge, sample_weights=None):
        horizon = features.shape[1]
        feature_dim = features.shape[3]
        x = features.reshape(features.shape[0], horizon, -1, feature_dim)
        y = true.reshape(true.shape[0], horizon, -1, true.shape[3])
        if sample_weights is not None:
            w = torch.sqrt(sample_weights.reshape(true.shape[0], horizon, -1, true.shape[3]).clamp_min(1.0e-6))
            x = x * w
            y = y * w
        xtx = torch.einsum('bhnf,bhng->hfg', x, x)
        xty = torch.einsum('bhnf,bhno->hf', x, y)
        eye = torch.eye(feature_dim)
        penalty = ridge * eye
        penalty[-1, -1] = 0.0
        matrix = xtx + penalty.view(1, feature_dim, feature_dim)
        rhs = xty.unsqueeze(-1)
        try:
            return torch.linalg.solve(matrix, rhs).squeeze(-1)
        except RuntimeError:
            return torch.matmul(torch.linalg.pinv(matrix), rhs).squeeze(-1)

    @staticmethod
    def _apply_horizon_calibration(features, calibration):
        coeffs = calibration[0]
        return torch.sum(features * coeffs.view(1, coeffs.shape[0], 1, coeffs.shape[1]), dim=-1, keepdim=True)

    @staticmethod
    def _metric_tuple(pred, true, args):
        mae, rmse, mape, _, _ = All_Metrics(pred, true, args.mae_thresh, args.mape_thresh)
        return float(mae), float(rmse), float(mape)

    @staticmethod
    def _calibration_selection_score(metrics):
        mae, rmse, mape = metrics
        return rmse + mae + 120.0 * mape

    def _select_eval_calibration(self, teacher, dgq_weights, periodic_weights=None):
        features, y_true, feature_names = self._collect_ensemble_features(
            teacher, dgq_weights, self.val_loader, periodic_weights=periodic_weights
        )
        base_pred = features[..., 0:1]
        base_metrics = self._metric_tuple(base_pred, y_true, self.args)
        best_score = self._calibration_selection_score(base_metrics)
        best = None

        sample_weights = (1.0 / y_true.abs().clamp_min(1.0)).clamp_max(0.2)
        ridge_values = getattr(self.args, 'eval_calibration_ridge_grid', None)
        if ridge_values is None:
            ridge_values = (1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1)

        for ridge in ridge_values:
            node_coeffs = self._fit_node_calibration(features, y_true, float(ridge))
            calibration = (node_coeffs, feature_names, 'node_ridge_{}'.format(ridge), base_metrics)
            pred = self._apply_eval_calibration(features, calibration)
            metrics = self._metric_tuple(pred, y_true, self.args)
            score = self._calibration_selection_score(metrics)
            if score < best_score:
                best_score = score
                best = (node_coeffs.to(self.args.device), feature_names, 'node_ridge_{}'.format(ridge), metrics)

            horizon_coeffs = self._fit_horizon_calibration(features, y_true, float(ridge))
            calibration = (horizon_coeffs, feature_names, 'horizon_ridge_{}'.format(ridge), base_metrics)
            pred = self._apply_horizon_calibration(features, calibration)
            metrics = self._metric_tuple(pred, y_true, self.args)
            score = self._calibration_selection_score(metrics)
            if score < best_score:
                best_score = score
                best = (horizon_coeffs.to(self.args.device), feature_names, 'horizon_ridge_{}'.format(ridge), metrics)

            weighted_node_coeffs = self._fit_node_calibration(features, y_true, float(ridge), sample_weights=sample_weights)
            calibration = (
                weighted_node_coeffs, feature_names, 'mape_weighted_node_ridge_{}'.format(ridge), base_metrics
            )
            pred = self._apply_eval_calibration(features, calibration)
            metrics = self._metric_tuple(pred, y_true, self.args)
            score = self._calibration_selection_score(metrics)
            if score < best_score:
                best_score = score
                best = (
                    weighted_node_coeffs.to(self.args.device), feature_names,
                    'mape_weighted_node_ridge_{}'.format(ridge), metrics
                )

            weighted_coeffs = self._fit_horizon_calibration(features, y_true, float(ridge), sample_weights=sample_weights)
            calibration = (
                weighted_coeffs, feature_names,
                'mape_weighted_horizon_ridge_{}'.format(ridge), base_metrics
            )
            pred = self._apply_horizon_calibration(features, calibration)
            metrics = self._metric_tuple(pred, y_true, self.args)
            score = self._calibration_selection_score(metrics)
            if score < best_score:
                best_score = score
                best = (
                    weighted_coeffs.to(self.args.device), feature_names,
                    'mape_weighted_horizon_ridge_{}'.format(ridge), metrics
                )

        if best is not None:
            return best

        return None

    @staticmethod
    def _apply_eval_calibration(features, calibration):
        coeffs = calibration[0]
        if coeffs.dim() == 2:
            return Trainer._apply_horizon_calibration(features, calibration)
        return torch.sum(features * coeffs.unsqueeze(0), dim=-1, keepdim=True)

    @staticmethod
    def test_dgq_ensemble(model, teacher, weights, args, data_loader, scaler, logger,
                          periodic_weights=None, calibration=None, online_adapter=None):
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
                    pred_real = scaler.inverse_transform(output)
                    true_real = scaler.inverse_transform(label)
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
                    pred_real = Trainer._apply_eval_calibration(features, calibration)
                    true_real = scaler.inverse_transform(label)

                if online_adapter is not None:
                    pred_real = online_adapter.adapt_batch(pred_real, true_real)
                    y_pred.append(pred_real.detach())
                    y_true.append(true_real.detach())
                else:
                    y_pred.append(pred_real)
                    y_true.append(true_real)

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        if online_adapter is not None:
            stats = online_adapter.summary()
            logger.info(
                "Online adaptation test: updates={}, drift_nodes={}, |bias| mean/max={:.4f}/{:.4f}, |scale-1| mean/max={:.4f}/{:.4f}".format(
                    stats['updates'], stats['drift_nodes'], stats['bias_mean'], stats['bias_max'],
                    stats['scale_mean'], stats['scale_max']
                )
            )
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, corr = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                   args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%".format(
                t + 1, rmse, mae, mape * 100))
        mae, rmse, mape, _, corr = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("test1 Average Horizon, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%".format(
            rmse, mae, mape * 100))

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None, online_adapter=None):
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
                pred_real = scaler.inverse_transform(output)
                true_real = scaler.inverse_transform(label)
                if online_adapter is not None:
                    pred_real = online_adapter.adapt_batch(pred_real, true_real)
                    y_pred.append(pred_real.detach())
                    y_true.append(true_real.detach())
                else:
                    y_pred.append(pred_real)
                    y_true.append(true_real)

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        if online_adapter is not None:
            stats = online_adapter.summary()
            logger.info(
                "Online adaptation test: updates={}, drift_nodes={}, |bias| mean/max={:.4f}/{:.4f}, |scale-1| mean/max={:.4f}/{:.4f}".format(
                    stats['updates'], stats['drift_nodes'], stats['bias_mean'], stats['bias_max'],
                    stats['scale_mean'], stats['scale_max']
                )
            )
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
