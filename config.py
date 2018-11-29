import torch as t


class DefaultConfig(object):
    env = 'default' # visdom env
    vis_port = 8097
    model = ''
    data_root = './data/sst.mon.mean1850-2015.nc'
    load_model_path = None

    height = 36
    width = 80
    channels = 1
    lags = 12
    steps = 12
    batch_size = 4
    n_epochs = 100
    lr = 0.1
    lr_decay = 0.95
    use_cuda = False
