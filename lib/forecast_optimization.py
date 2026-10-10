"""Forecast loss balancing and averaged weights, without architecture changes."""
import torch


def configure_forecast_training(model, finetune_core=False):
    """Train only modules used by direct inference; preserve the long prior."""
    core_names = {'node_embeddings1', 'T_i_D_emb1', 'T_i_D_emb2', 'D_i_W_emb1', 'D_i_W_emb2'}
    head, core = [], []
    for name, parameter in model.named_parameters():
        is_head = name.startswith('horizon_forecaster.')
        is_core = finetune_core and (name.startswith('encoder.') or name in core_names)
        parameter.requires_grad_(is_head or is_core)
        if is_head:
            head.append(parameter)
        elif is_core:
            core.append(parameter)
    return head, core


def balanced_forecast_loss(pred, labels, true_real, teacher, std,
                           mse_weight, mape_weight, distill_weight, near_weight=1.0, rmse_weight=0.0):
    if near_weight <= 0:
        raise ValueError('Horizon weights must be positive')
    diff = pred.float()-labels.float()
    weights = torch.linspace(near_weight, 1.0, pred.shape[1], device=pred.device)
    weights = (weights/weights.mean()).view(1, -1, 1, 1)
    valid = true_real > 0
    percentage = diff.abs()*std/true_real.clamp_min(1)
    mape = (percentage*weights)[valid].mean() if valid.any() else diff.sum()*0
    parts = dict(mae=(diff.abs()*weights).mean(), mse=(diff.square()*weights).mean(),
                 mape=mape, distill=((pred.float()-teacher.float()).square()*weights).mean())
    parts['rmse'] = (parts['mse']+1e-8).sqrt()
    total = parts['mae']+mse_weight*parts['mse']+mape_weight*mape+distill_weight*parts['distill']
    total = total+rmse_weight*parts['rmse']
    return total, parts


def chronological_sample_weights(count, recent_fraction=0.0, recent_multiplier=1.0):
    if not 0 <= recent_fraction <= 1 or recent_multiplier < 1:
        raise ValueError('Invalid recent training sampling settings')
    weights = torch.ones(count)
    recent_count = int(count*recent_fraction)
    if recent_count:
        weights[-recent_count:] = recent_multiplier
    return weights


class ForecastEMA:
    def __init__(self, module, decay):
        if not 0 < decay < 1:
            raise ValueError('EMA decay must lie between zero and one')
        self.decay, self.updates = decay, 0
        self.shadow = {name:value.detach().clone() for name,value in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module):
        self.updates += 1
        for name, value in module.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].lerp_(value.detach(), 1-self.decay)
            else:
                self.shadow[name].copy_(value)

    def state_dict(self):
        return dict(decay=self.decay, updates=self.updates, shadow=self.shadow)

    def load_state_dict(self, state):
        self.decay, self.updates = state['decay'], state['updates']
        for name,value in state['shadow'].items():
            self.shadow[name].copy_(value)


def validation_score(metrics, reference=None):
    if reference is None:
        return metrics['rmse']+metrics['mae']+1.2*metrics['mape']
    ratios = [metrics[key]/reference[key] for key in ('rmse', 'mae', 'mape')]
    return max(ratios)+0.05*(sum(ratios)/len(ratios)-1)
