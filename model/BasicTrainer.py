import torch
import math
import os
import time
import copy
import numpy as np
import pynvml
from lib.logger import get_logger
from lib.metrics import All_Metrics

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
            vaild_loss.append(val_epoch_loss)

            test_metrics = self.test_epoch(epoch, test_dataloader)
            test_epoch_loss = test_metrics['loss']
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
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
                                     best_loss, best_test_loss, not_improved_count)

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
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

        self.logger.info("This is best_test_model")
        self.model.load_state_dict(best_test_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

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

    def _log_epoch_progress(self, epoch, train_loss, val_metrics, test_metrics,
                            best_val_loss, best_test_loss, not_improved_count):
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(
            'Epoch Progress {}/{} | lr: {:.8f} | train_loss: {:.6f} | '
            'val_loss: {:.6f}, val_MAE: {:.4f}, val_RMSE: {:.4f}, val_MAPE: {:.4f}%, val_CORR: {:.4f} | '
            'test_loss: {:.6f}, test_MAE: {:.4f}, test_RMSE: {:.4f}, test_MAPE: {:.4f}%, test_CORR: {:.4f} | '
            'best_val_loss: {:.6f}, best_test_loss: {:.6f}, no_improve_epochs: {}'.format(
                epoch, self.args.epochs, current_lr, train_loss,
                val_metrics['loss'], val_metrics['mae'], val_metrics['rmse'],
                val_metrics['mape'] * 100, val_metrics['corr'],
                test_metrics['loss'], test_metrics['mae'], test_metrics['rmse'],
                test_metrics['mape'] * 100, test_metrics['corr'],
                best_val_loss, best_test_loss, not_improved_count
            )
        )

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
