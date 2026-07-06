import torch
import numpy as np
import torch.utils.data
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
import os

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    # return data, scaler
    return scaler

def split_data_by_days(data, val_days, test_days, interval=30):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    x = -T*test_days
    test_data = data[-int(T*test_days):]
    val_data = data[-int(T*(test_days + val_days)): -int(T*test_days)]
    train_data = data[:-int(T*(test_days + val_days))]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def _build_periodic_context(raw_norm, start_indices, window, args):
    periodic_context = np.zeros((len(start_indices), window, raw_norm.shape[1], raw_norm.shape[2]), dtype=np.float32)
    context_valid = np.zeros((len(start_indices), window, raw_norm.shape[1], 1), dtype=np.float32)
    temperature = max(float(args.context_temperature), 1.0e-6)

    def cosine_similarity(cur_seq, ref_seq):
        cur_flat = np.transpose(cur_seq, (1, 0, 2)).reshape(cur_seq.shape[1], -1)
        ref_flat = np.transpose(ref_seq, (1, 0, 2)).reshape(ref_seq.shape[1], -1)
        numerator = np.sum(cur_flat * ref_flat, axis=-1)
        denominator = np.linalg.norm(cur_flat, axis=-1) * np.linalg.norm(ref_flat, axis=-1)
        return numerator / np.clip(denominator, 1.0e-8, None)

    for sample_idx, start_idx in enumerate(start_indices):
        current = raw_norm[start_idx:start_idx + window]
        candidates = []
        scores = []

        if start_idx >= args.periodic_day_steps:
            day_seq = raw_norm[start_idx - args.periodic_day_steps:start_idx - args.periodic_day_steps + window]
            candidates.append(day_seq)
            scores.append(cosine_similarity(current, day_seq))

        if start_idx >= args.periodic_week_steps:
            week_seq = raw_norm[start_idx - args.periodic_week_steps:start_idx - args.periodic_week_steps + window]
            candidates.append(week_seq)
            scores.append(cosine_similarity(current, week_seq))

        if not candidates:
            continue

        context_valid[sample_idx] = 1.0
        if len(candidates) == 1:
            periodic_context[sample_idx] = candidates[0]
            continue

        score_stack = np.stack(scores, axis=-1) / temperature
        score_stack = score_stack - np.max(score_stack, axis=-1, keepdims=True)
        weights = np.exp(score_stack)
        weights = weights / np.clip(np.sum(weights, axis=-1, keepdims=True), 1.0e-8, None)

        fused = np.zeros_like(candidates[0], dtype=np.float32)
        for candidate_idx, candidate in enumerate(candidates):
            fused += candidate * weights[:, candidate_idx][None, :, None]
        periodic_context[sample_idx] = fused

    return periodic_context, context_valid


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True, seed=None):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last,
                                             generator=generator)
    return dataloader


def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    data = load_st_dataset(args.dataset).astype(np.float32)        # B, N, D

    L, N, F = data.shape

    t = args.steps_per_day
    # numerical time_in_day
    time_ind    = [i%t / t for i in range(data.shape[0])]
    time_ind    = np.array(time_ind)
    time_in_day = np.tile(time_ind, [1, N, 1]).transpose((2, 1, 0))
    # numerical day_in_week
    day_in_week = [(i // t)%args.steps_per_week for i in range(data.shape[0])]
    day_in_week = np.array(day_in_week)
    day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((2, 1, 0))
    end_index = L - args.horizon - args.lag + 1
    start_indices = np.arange(end_index)

    #spilit dataset by days or by ratio
    if args.test_ratio > 1:
        train_starts, val_starts, test_starts = split_data_by_days(start_indices, args.val_ratio, args.test_ratio)
    else:
        train_starts, val_starts, test_starts = split_data_by_ratio(start_indices, args.val_ratio, args.test_ratio)

    train_raw_end = int(train_starts[-1] + args.lag) if len(train_starts) > 0 else L
    scaler = normalize_dataset(data[:train_raw_end, ..., :args.input_dim], normalizer, args.column_wise)
    raw_norm = scaler.transform(data[..., :args.input_dim]).astype(np.float32)

    x_traffic, y_traffic = Add_Window_Horizon(raw_norm, args.lag, args.horizon, single)
    x_day, y_day = Add_Window_Horizon(time_in_day.astype(np.float32), args.lag, args.horizon, single)
    x_week, y_week = Add_Window_Horizon(day_in_week.astype(np.float32), args.lag, args.horizon, single)

    if args.use_periodic_context and args.use_context_graph_refine:
        periodic_context, context_valid = _build_periodic_context(raw_norm, start_indices, args.lag, args)
        x = np.concatenate([x_traffic, periodic_context, context_valid, x_day, x_week], axis=-1)
    else:
        x = np.concatenate([x_traffic, x_day, x_week], axis=-1)
    y = np.concatenate([y_traffic, y_day, y_week], axis=-1)

    x_train, x_val, x_test = x[train_starts], x[val_starts], x[test_starts]
    y_train, y_val, y_test = y[train_starts], y[val_starts], y[test_starts]

    print('Train: ', x_train.shape, y_train.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    ##############get dataloader######################
    train_dataloader = data_loader(x_train, y_train, args.batch_size, shuffle=True, drop_last=True, seed=args.seed)
    if len(x_val[...,0]) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=False, seed=args.seed)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False, seed=args.seed)
    return train_dataloader, val_dataloader, test_dataloader, scaler

def get_adjacency_matrix2(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A


if __name__ == '__main__':
    import argparse
    #MetrLA 207; BikeNYC 128; SIGIR_solar 137; SIGIR_electric 321
    DATASET = 'SIGIR_electric'
    if DATASET == 'MetrLA':
        NODE_NUM = 207
    elif DATASET == 'BikeNYC':
        NODE_NUM = 128
    elif DATASET == 'SIGIR_solar':
        NODE_NUM = 137
    elif DATASET == 'SIGIR_electric':
        NODE_NUM = 321
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True)
