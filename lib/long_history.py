"""Chronological context construction with train-only normalization."""
import csv

import numpy as np
import torch


def split_origins(length, lag=12, horizon=12, val_ratio=0.2, test_ratio=0.2):
    count = length-lag-horizon+1
    train_end = count-int(count*(val_ratio+test_ratio))
    val_end = count-int(count*test_ratio)
    return (torch.arange(lag, train_end+lag-horizon+1),
            torch.arange(train_end+lag, val_end+lag-horizon+1),
            torch.arange(val_end+lag, count+lag)), train_end+lag-1


def physical_graph(csv_path, nodes):
    graph = torch.eye(nodes)
    with open(csv_path) as file:
        for row in csv.DictReader(file):
            first, second = int(row['from']), int(row['to'])
            graph[first, second] = graph[second, first] = 1
    return graph/graph.sum(-1, keepdim=True)


def long_windows(raw, ends, length):
    indices = ends[:, None]-length+torch.arange(length, device=raw.device)[None]
    available = indices >= 0
    values = raw[indices.clamp_min(0)]*available.unsqueeze(-1)
    return values, available.unsqueeze(-1), ends-length


def context_batch(raw, origins, lag=12, horizon=12, day=288, week=2016):
    """Target traffic is never placed in inference inputs; calendar is known."""
    history_indices = origins[:, None]-lag+torch.arange(lag, device=raw.device)[None]
    history = raw[history_indices]
    future_indices = origins[:, None]+torch.arange(horizon, device=raw.device)[None]
    references, future_references, flags, similarities = [], [], [], []
    for period in (day, week):
        valid = history_indices[:, 0] >= period
        reference = raw[(history_indices-period).clamp_min(0)]*valid[:, None, None]
        numerator = (history*reference).sum(1)
        denominator = (history.square().sum(1)*reference.square().sum(1)).sqrt().clamp_min(1e-8)
        similarities.append((numerator/denominator).masked_fill(~valid[:, None], -10000))
        references.append(reference)
        future_references.append(raw[(future_indices-period).clamp_min(0)]*valid[:, None, None])
        flags.append(valid)
    weights = torch.softmax(torch.stack(similarities, -1), -1)
    periodic_history = sum(reference*weights[:, None, :, index] for index, reference in enumerate(references))
    periodic_future = sum(reference*weights[:, None, :, index] for index, reference in enumerate(future_references))
    valid = flags[0] | flags[1]
    batch, _, nodes = history.shape
    calendars = lambda indices: torch.stack(((indices % day).float()/day,
                                              (indices//day % 7).float()), -1)[:, :, None].expand(-1, -1, nodes, -1)
    source = torch.cat((history.unsqueeze(-1), periodic_history.unsqueeze(-1),
                        valid[:, None, None, None].expand(batch, lag, nodes, 1), calendars(history_indices)), -1)
    target = torch.cat((torch.zeros(batch, horizon, nodes, 1, device=raw.device), periodic_future.unsqueeze(-1),
                        valid[:, None, None, None].expand(batch, horizon, nodes, 1), calendars(future_indices)), -1)
    periodic = torch.stack((future_references[0], future_references[1],
        flags[0][:, None, None].expand(batch, horizon, nodes),
        flags[1][:, None, None].expand(batch, horizon, nodes)), -1)
    return source, target, periodic


def structured_mask(history, available, patch_size, steps_per_day, generator=None):
    """Mask whole nodes, contiguous time spans and repeated periodic fragments."""
    batch, length, nodes = history.shape
    patches = length//patch_size
    rand = lambda shape: torch.rand(shape, device=history.device, generator=generator)
    patch_mask = rand((batch, patches, nodes)) < 0.25
    patch_mask |= (rand((batch, 1, nodes)) < 0.12).expand(-1, patches, -1)
    starts = (rand((batch, 1, 1))*(patches-8)).long()
    positions = torch.arange(patches, device=history.device)[None, :, None]
    patch_mask |= ((positions >= starts) & (positions < starts+8)).expand(-1, -1, nodes)
    patches_per_day = steps_per_day//patch_size
    phase = (rand((batch, 1, 1))*patches_per_day).long()
    periodic = ((positions % patches_per_day)-phase).abs() < 2
    patch_mask |= periodic.expand(-1, -1, nodes)
    return patch_mask.repeat_interleave(patch_size, 1) & available.expand_as(history)
