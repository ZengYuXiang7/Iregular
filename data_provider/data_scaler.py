# coding : utf-8
# Author : yuxiang Zeng
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataScalerStander:
    def __init__(self, y, config=None):
        self.config = config
        y = self.__check_input__(y)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim > 2:
            y = y.reshape(-1, y.shape[-1])

        self.scaler = StandardScaler()
        self.scaler.fit(y)

    def transform(self, y):
        y = self.__check_input__(y)
        orig_shape = y.shape
        if y.ndim == 1:
            return self.scaler.transform(y.reshape(-1, 1))
        elif y.ndim > 2:
            return self.scaler.transform(y.reshape(-1, y.shape[-1])).reshape(orig_shape)
        return self.scaler.transform(y)

    def inverse_transform(self, y):
        y = self.__check_input__(y)
        orig_shape = y.shape
        if y.ndim == 1:
            return self.scaler.inverse_transform(y.reshape(-1, 1))
        elif y.ndim > 2:
            return self.scaler.inverse_transform(y.reshape(-1, y.shape[-1])).reshape(orig_shape)
        return self.scaler.inverse_transform(y)

    def __check_input__(self, y):
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        return y.astype(float)


class DataScalerMinMax:
    def __init__(self, y, config=None):
        self.config = config
        y = self.__check_input__(y)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim > 2:
            y = y.reshape(-1, y.shape[-1])

        self.scaler = MinMaxScaler()
        self.scaler.fit(y)

    def transform(self, y):
        y = self.__check_input__(y)
        orig_shape = y.shape
        if y.ndim == 1:
            return self.scaler.transform(y.reshape(-1, 1))
        elif y.ndim > 2:
            return self.scaler.transform(y.reshape(-1, y.shape[-1])).reshape(orig_shape)
        return self.scaler.transform(y)

    def inverse_transform(self, y):
        y = self.__check_input__(y)
        orig_shape = y.shape
        if y.ndim == 1:
            return self.scaler.inverse_transform(y.reshape(-1, 1))
        elif y.ndim > 2:
            return self.scaler.inverse_transform(y.reshape(-1, y.shape[-1])).reshape(orig_shape)
        return self.scaler.inverse_transform(y)

    def __check_input__(self, y):
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        return y.astype(float)


class GlobalStandardScaler:
    def __init__(self, y, config=None):
        self.config = config
        y = self.__check_input__(y)

        self.mean = y.mean()
        self.std = y.std()
        if self.std == 0:
            self.std = 1

    def transform(self, y):
        y = self.__check_input__(y)
        return (y - self.mean) / self.std

    def inverse_transform(self, y):
        y = self.__check_input__(y)
        return y * self.std + self.mean

    def __check_input__(self, y):
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        return y.astype(float)


class GlobalMinMaxScaler:
    def __init__(self, y, config=None):
        self.config = config
        y = self.__check_input__(y)

        self.min = y.min()
        self.max = y.max()
        if self.max - self.min == 0:
            self.max += 1e-6

    def transform(self, y):
        y = self.__check_input__(y)
        return (y - self.min) / (self.max - self.min)

    def inverse_transform(self, y):
        y = self.__check_input__(y)
        return y * (self.max - self.min) + self.min

    def __check_input__(self, y):
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        return y.astype(float)


class NoneScaler:
    def __init__(self, y, config=None):
        self.config = config

    def transform(self, y):
        return y

    def inverse_transform(self, y):
        return y


def get_scaler(y, config, selected_method=None):
    method = selected_method if selected_method else config.scaler_method
    if method == 'stander':
        return DataScalerStander(y, config)
    elif method == 'minmax':
        return DataScalerMinMax(y, config)
    elif method == 'globalstander':
        return GlobalStandardScaler(y, config)
    elif method == 'globalminmax':
        return GlobalMinMaxScaler(y, config)
    elif method == 'none':
        return NoneScaler(y, config)
    else:
        raise NotImplementedError(f"Scaler method '{method}' is not supported.")