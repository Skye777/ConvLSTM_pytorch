import torch as t
from model import *
import numpy as np
from data import dataset
from torch.utils.data import DataLoader
from config import DefaultConfig


def val(model, dataloader, criterion):

    # set to evaluation mode
    model.eval()

    val_loss_list = list()
    for ii, (val_input, val_target) in enumerate(dataloader):
        val_input = t.Tensor(val_input)
        val_target = t.Tensor(val_target)

        if opt.use_cuda:
            val_input.cuda()
            val_target.cuda()

        val_output, val_state = model(val_input)
        val_output = val_output[0]
        loss = criterion(val_output, val_target)
        val_loss_list.append(loss.item())

    model.train()
    val_loss = np.average(val_loss_list)

    return val_loss


def train(opt):

    # define and load model
    model = ConvLSTM(input_size=(opt.height, opt.width),
                     input_dim=opt.channels,
                     hidden_dim=[16, 16, 16, 1],
                     kernel_size=(3, 3),
                     num_layers=4,
                     batch_first=True,
                     bias=True,
                     return_all_layers=False)

    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_cuda:
        model.cuda()

    # load data
    train_dataset = dataset.SST(opt.data_root, opt.lags, opt.steps, opt.channels, opt.height, opt.width, train=True)
    val_dataset = dataset.SST(opt.data_root, opt.lags, opt.steps, opt.channels, opt.height, opt.width, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    # objective function and optimizer
    criterion = t.nn.MSELoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    previous_loss = 1e100

    for epoch in range(opt.n_epochs):

        print("Epoch: [{}/{}]".format(epoch, opt.n_epochs))

        for ii, (input, target) in enumerate(train_dataloader):

            input = t.Tensor(input)
            target = t.Tensor(target)

            if opt.use_cuda:
                input.cuda()
                target.cuda()

            # print(np.array(input).shape)
            optimizer.zero_grad()
            output, state = model(input)
            output = output[0]
            loss = criterion(output, target)
            print("Batch:[{}], Loss:{}".format(ii, loss.item()))
            loss.backward()
            optimizer.step()

        model.save()

        val_loss = val(model, val_dataloader, criterion)
        print("Epoch:[{}], Val_Loss:{}, lr:{}".format(epoch, val_loss, lr))

        if val_loss > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = val_loss


opt = DefaultConfig()
train(opt)



