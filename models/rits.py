import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import utils
import argparse
import data_loader


from sklearn import metrics

SEQ_LEN =456#48小时采样49次
RNN_HID_SIZE =176#hide层的维度

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)#最小返回0
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log() #Lout

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class FeatureRegression(nn.Module):#对角限制为0
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size)) #其作用将一个不可训练的类型为Tensor的参数转化为可训练的类型为parameter的参数
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size) #eye对角线全一，此时此矩阵对角线为0，其余为一
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)# 随机生成一个实数,"_"重新赋值
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b) #w*x+b 限制w对角线为0
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)#应该就是在内存中定一个常量，同时，模型保存和加载的时候可以写入和读出

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))#第一维度的大小
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True: #
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))#限制w仅有对角线的值？
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))#relu和0找最大时间衰减参数
        gamma = torch.exp(-gamma)
        return gamma   #


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.build()
    def build(self):
        self.rnn_cell = nn.LSTMCell(161 * 2, RNN_HID_SIZE) #xc和m拼接
        self.rnn_cell1 = nn.LSTMCell(RNN_HID_SIZE, RNN_HID_SIZE)
        self.rnn_cell2 = nn.LSTMCell(RNN_HID_SIZE, RNN_HID_SIZE)
        #self.rnn_cell3 = nn.LSTMCell(RNN_HID_SIZE, RNN_HID_SIZE)
        self.temp_decay_h = TemporalDecay(input_size = 161, output_size = RNN_HID_SIZE, diag = False)#h时间衰减
        self.temp_decay_x = TemporalDecay(input_size = 161, output_size =161, diag = True)#时间衰减参数
        self.hist_reg = nn.Linear(RNN_HID_SIZE, 161)#线性变换
        self.feat_reg = FeatureRegression(161) #特征矩阵
        self.weight_combine = nn.Linear(161 * 2, 161)#bata权重
        self.dropout = nn.Dropout(p = 0.25)#训练过程中以概率P随机的将参数置0，其中P为置0的概率
        self.out = nn.Linear(RNN_HID_SIZE, 1)
    def forward(self, data, direct):
        # Original sequence with 24 time steps
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']
        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']
        labels = data['labels'].view(-1, 1)#reshape成一列，行数自适应
        is_train = data['is_train'].view(-1, 1)#reshape成一列，行数自适应
        h = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))#变量 上一层隐藏状态
        c = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))#细胞状态
        h0 = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))  # 变量 上一层隐藏状态
        c0 = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))  # 细胞状态
        h1 = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))  # 变量 上一层隐藏状态
        c1 = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))  # 细胞状态
        if torch.cuda.is_available():
            h, c = h.cuda('cuda:0'), c.cuda('cuda:0')
            h0, c0 = h0.cuda('cuda:0'), c0.cuda('cuda:0')
            h1, c1 = h1.cuda('cuda:0'), c1.cuda('cuda:0')
        x_loss = 0.0
        y_loss = 0.0
        imputations = []
        for t in range(SEQ_LEN):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            gamma_h = self.temp_decay_h(d)#h的时间衰减参数
            gamma_x = self.temp_decay_x(d) #根据delta更新时间衰减参数
            h = h * gamma_h#h*其时间衰减参数
            x_h = self.hist_reg(h) #x的估计
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5) #基于历史估计的损耗（对不缺失的值
            x_c =  m * x +  (1 - m) * x_h  #x补码
            z_h = self.feat_reg(x_c) #基于特征的估计

            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)#基于特征估计的损耗
            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))   #权重bata
            c_h = alpha * z_h + (1 - alpha) * x_h #结合更新
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)#联合更新之后的损耗
            c_c = m * x + (1 - m) * c_h #补码输入c——c

            inputs = torch.cat([c_c, m], dim = 1) #将c_c和m以第一维进行拼接输入
            h0, c0 = self.rnn_cell(inputs, (h0, c0))
            #h1, c1 = self.rnn_cell1(h0, (h1, c1))
            h, c = self.rnn_cell2(h0, (h, c)) #使用lstm更新隐藏层h和c细胞状态

            imputations.append(c_c.unsqueeze(dim = 1))#保存估计值
        imputations = torch.cat(imputations, dim = 1)#拼接估计值成为一个tensor

        y_h = self.out(h)#linear预测标签
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False) #
        y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)#lout

        y_h = F.sigmoid(y_h)#softmax归一到0到1

        return {'loss': x_loss / SEQ_LEN , 'predictions': y_h,\
                'imputations': imputations, 'labels': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}

    def run_on_batch(self, data, optimizer): #调用模型并定义优化器
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()#梯度置零
            ret['loss'].backward()#反向传播计算梯度
            optimizer.step()#更新参数

        return ret
