# -*- coding:utf-8 -*-

import math
import torch
import numpy as np
from sklearn.metrics import mean_squared_error


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(np.abs((y_true - y_pred) / (y_true)) * 100)


def calculate_scores(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r = np.corrcoef(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"rmse": rmse, "R": r[0, 1], "mape": mape}
