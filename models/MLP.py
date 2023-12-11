import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math

SEQ_LEN =20

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.hidden=nn.Sequential(
            nn.Linear(SEQ_LEN*2,128),
            nn.Sigmoid(),
            nn.Linear(128, 64), # 第二个隐藏层为64个神经元
            nn.Sigmoid(),
            nn.Linear(64, 32),  # 第三个隐藏层为32个神经元
            nn.Sigmoid(),
            nn.Linear(32, SEQ_LEN),  # 第三个隐藏层为32个神经元
            
        )

    def forward(self, data, direct):
        # Original sequence with 24 time steps


        values = data[direct]['values']
        #print(values.size())
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        x_loss = 0.0

        imputations = []

        for t in range(72):
            x = values[:, :, t]
            m = masks[:, :, t]
            d = deltas[:, :, t]
            inputs = torch.cat([x, m], dim=1)

            #gamma_xf = vare_forward[t]
            #gamma_xb=vare_backward[t]
            x_c=self.hidden(inputs)
            #print(x_c.size())

            x_loss += torch.sum(torch.abs(x - x_c) * m) / (torch.sum(m) + 1e-5)

            imputations.append(x_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)
        a,b,c=imputations.size()
        imputations = imputations.reshape(a, c, b)
        #imputations=imputations.reshape(64,10,72)
        #print(imputations.size())
        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

       # y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)
       # y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)

        #y_h = F.sigmoid(y_h)

        #*self.impute_weight + y_loss * self.label_weight
        return {'loss': x_loss / SEQ_LEN,\
                'imputations': imputations,\
                'evals': evals, 'eval_masks': eval_masks}

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret