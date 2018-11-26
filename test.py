import torch.nn as nn
from torch.autograd import Variable
import torch
from convlstm import *
import numpy as np

# rnn1 = nn.ConvLSTM(3, 10, 2, kernal_size=3)
# input1 = Variable(torch.randn(4, 10, 3, 25, 25))
# h01 = Variable(torch.randn(2, 10, 10, 25, 25))
# c01 = Variable(torch.randn(2, 10, 10, 25, 25))
# output1, hn1 = rnn1(input1, (h01, c01))
#
# print('input1:{}'.format(input1.size()))
# print('output1:{}'.format(output1.size()))
# print('hn1:{}'.format(hn1[0][0].size()))

height = 10
width = 20
channels = 3
steps = 12
batch_size = 3

model = ConvLSTM(input_size=(height, width),
                 input_dim=channels,
                 hidden_dim=32,
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)
conv3d = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3)

input = Variable(torch.randn(batch_size, steps, channels, height, width))
output1, output2 = model(input)
output3 = conv3d(output1)
# print(output1)
# print(output2)
print(np.array(output1).shape)
print(np.array(output2).shape)
print(np.array(output3).shape)

