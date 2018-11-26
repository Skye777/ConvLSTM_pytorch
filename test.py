import torch.nn as nn
from torch.autograd import Variable
import torch as t
import netCDF4 as nc
from convlstm import *
import numpy as np
from numpy import array


def supervised_samples(sst, lags, steps):
    X, y = list(), list()
    for s in range(len(sst)-lags-steps+1):
        X.append(sst[s:s+lags])
        y.append(sst[s+lags:s+lags+steps])
    return t.from_array(X), t.from_array(y)
# rnn1 = nn.ConvLSTM(3, 10, 2, kernal_size=3)
# input1 = Variable(torch.randn(4, 10, 3, 25, 25))
# h01 = Variable(torch.randn(2, 10, 10, 25, 25))
# c01 = Variable(torch.randn(2, 10, 10, 25, 25))
# output1, hn1 = rnn1(input1, (h01, c01))
#
# print('input1:{}'.format(input1.size()))
# print('output1:{}'.format(output1.size()))
# print('hn1:{}'.format(hn1[0][0].size()))


# hyper parameter setting
height = 36
width = 80
channels = 1
lags = 12
steps = 12
batch_size = 3

# load data
nc_obj = nc.Dataset("sst.mon.mean1850-2015.nc")
sst = nc_obj.variables['sst'][:]
sst_all = sst[::, 72:108:, 175:255:]

X, y = supervised_samples(sst_all, lags, steps)
print(np.array(X).shape)
print(np.array(y).shape)

train_X = X[:-120].view(len(X[:120]), lags, channels, height, width)
# train_X, train_y = array(X[:-120]).reshape(len(X[:-120]), lags, 12*3, 50*2-20, 1), array(y[:-120]).reshape(len(y[:-120]), steps, 12*3, 50*2-20, 1)
model = ConvLSTM(input_size=(height, width),
                 input_dim=channels,
                 hidden_dim=[16, 16, 16, 1],
                 kernel_size=(3, 3),
                 num_layers=4,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)

# input = Variable(torch.randn(batch_size, steps, channels, height, width))
output1, output2 = model(train_X)
# output3 = conv3d(output1)
# print(output1)
# print(output2)
print(np.array(output1).shape)
print(np.array(output2).shape)
# print(np.array(output3).shape)

