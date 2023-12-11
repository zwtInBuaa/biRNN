import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import pandas as pd

def to_var(var):
    if torch.is_tensor(var):#变为变量
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda('cuda:0')
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var

def stop_gradient(x):
    if isinstance(x, float):
        return x
    if isinstance(x, tuple):
        return tuple(map(lambda y: Variable(y.data), x))
    return Variable(x.data)

def zero_var(sz):
    x = Variable(torch.zeros(sz))
    if torch.cuda.is_available():
        x = x.cuda('cuda:0')
    return x
