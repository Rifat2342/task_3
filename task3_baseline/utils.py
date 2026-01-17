import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_list(items):
    if items is None:
        return []
    if isinstance(items, (list, tuple)):
        return list(items)
    return [item.strip() for item in items.split(",") if item.strip()]


def parse_scenarios(scenarios):
    return parse_list(scenarios)


def parse_tasks(tasks):
    return parse_list(tasks)


def denormalize_values(values, mean, std):
    if mean is None or std is None:
        return values
    return values * std + mean


def compute_srs_metrics(preds, targets, target_mean=None, target_std=None):
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    if target_mean is not None and target_std is not None:
        mean = torch.tensor(target_mean, dtype=preds.dtype)
        std = torch.tensor(target_std, dtype=preds.dtype)
        preds = denormalize_values(preds, mean, std)
        targets = denormalize_values(targets, mean, std)

    errors = preds - targets
    mae = torch.mean(torch.abs(errors)).item()
    rmse = torch.sqrt(torch.mean(errors ** 2)).item()

    return {"mae": mae, "rmse": rmse}
