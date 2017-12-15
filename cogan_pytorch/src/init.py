from torch import nn
import torch.nn.init as init
import numpy as np


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)


def xavier_weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0.1)