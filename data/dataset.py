from torch.utils import data
import netCDF4 as nc
from numpy import array
import torch as t
import numpy as np


class SST(data.Dataset):

    def __init__(self, root_dir, lags, steps, channels, height, width, transform=None, train=True, test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.lags = lags
        self.steps = steps
        self.channels = channels
        self.height = height
        self.width = width
        self.train = train
        self.test = test
        self.input, self.target = list(), list()

        nc_obj = nc.Dataset(self.root_dir)
        sst = nc_obj.variables['sst'][:]
        sst_nino34 = sst[::, 72:108:, 175:255:]

        for s in range(len(sst_nino34) - self.lags - self.steps + 1):
            self.input.append(sst_nino34[s:s + self.lags])
            self.target.append(sst_nino34[s + self.lags:s + self.lags + self.steps])
        # print(np.array(X[item]).shape)

        self.num_sample = len(self.input)

    def __getitem__(self, item):

        if self.train:
            x, y = self.input[:int(0.7 * self.num_sample)], self.target[:int(0.7 * self.num_sample)]
        else:
            x, y = self.input[int(0.7 * self.num_sample):], self.target[int(0.7 * self.num_sample):]

        input, target = array(x[item]).reshape(self.lags, self.channels, self.height, self.width), \
                        array(y[item]).reshape(self.steps, self.channels, self.height, self.width)
        return input, target

    def __len__(self):

        if self.train:
            return int(0.7 * self.num_sample)
        else:
            return self.num_sample - int(0.7 * self.num_sample)

