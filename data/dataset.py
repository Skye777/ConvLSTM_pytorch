from torch.utils import data
import netCDF4 as nc
from numpy import array
import torch as t
import numpy as np


class SST(data.Dataset):

    def __init__(self, root_dir, lags, steps, channels, height, width, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.lags = lags
        self.steps = steps
        self.channels = channels
        self.height = height
        self.width = width

    def __getitem__(self, item):
        nc_obj = nc.Dataset(self.root_dir)
        sst = nc_obj.variables['sst'][:]
        sst_nino34 = sst[::, 72:108:, 175:255:]

        X, y = list(), list()
        for s in range(len(sst_nino34) - self.lags - self.steps + 1):
            X.append(sst_nino34[s:s + self.lags])
            y.append(sst_nino34[s + self.lags:s + self.lags + self.steps])
        # print(np.array(X[item]).shape)
        input, target = array(X[item]).reshape(self.lags, self.channels, self.height, self.width), \
                        array(y[item]).reshape(self.steps, self.channels, self.height, self.width)
        return input, target

    def __len__(self):
        return 1993

