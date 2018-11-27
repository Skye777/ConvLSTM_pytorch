import torch as t
from model.convlstm import *
import numpy as np
from data import dataset
from torch.utils.data import DataLoader


# hyper parameter setting
height = 36
width = 80
channels = 1
lags = 12
steps = 12
batch_size = 4
n_epochs = 100
use_cuda = False

# load data
train_dataset = dataset.SST('./data/sst.mon.mean1850-2015.nc', lags, steps, channels, height, width)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

# train_X = X[:-120].view(len(X[:120]), lags, channels, height, width)
model = ConvLSTM(input_size=(height, width),
                 input_dim=channels,
                 hidden_dim=[16, 16, 16, 1],
                 kernel_size=(3, 3),
                 num_layers=4,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)

criterion = t.nn.MSELoss()
optimizer = t.optim.Adam(model.parameters())

if use_cuda:
    model.cuda()
    criterion.cuda()

for epoch in range(n_epochs):

    print("Epoch: [{}/{}]".format(epoch, n_epochs))

    for ii, (input, target) in enumerate(train_dataloader):

        input = t.Tensor(input)
        target = t.Tensor(target)

        if use_cuda:
            input.cuda()
            target.cuda()

        # print(np.array(input).shape)
        output, state = model(input)
        output = output[0]
        loss = criterion(output, target)
        print("Batch:[{}], Loss:{}".format(ii, loss.item()))
        loss.backward()
        optimizer.step()


