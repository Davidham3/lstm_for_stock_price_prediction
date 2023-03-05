# -*- coding:utf-8 -*-

import math
import torch
import numpy as np
from sklearn.metrics import mean_squared_error


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(
        np.abs((y_true - y_pred) / (y_true)) * 100
    )  # some issues with zero denominator


def calculate_scores(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r = np.corrcoef(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"rmse": rmse, "R": r[0, 1], "mape": mape}


def evaluate(pl_net, ts_data_module):
    preds, labels = [], []
    with torch.no_grad():
        for x, y in ts_data_module.test_dataloader():
            prediction = pl_net(x) * (ts_data_module.train_y_max - ts_data_module.train_y_min) + ts_data_module.train_y_min
            preds.append(prediction)
            labels.append(y)
    return calculate_scores(torch.cat(labels, dim=0).numpy(), torch.cat(preds, dim=0).numpy())