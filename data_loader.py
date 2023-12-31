import os
import time

import ujson as json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MySet(Dataset):
    def __init__(self):
        super(MySet, self).__init__()
        with open('./json/our.json', mode='r', encoding='utf-8') as f:
            self.content = json.load(f)
        # self.content = open('./json/json').readlines()

        indices = np.arange(len(self.content))  # content长度的排列，0到len（content-1）
        val_indices = np.random.choice(indices, len(self.content) // 10)  # //整数除法

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = self.content[idx]  ####
        # rec = json.loads(self.content[idx])
        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1  # 训练集
        return rec


def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict(recs):
        values = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['values'], r)), recs)))
        masks = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['masks'], r)), recs)))
        deltas = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['deltas'], r)), recs)))
        forwards = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['forwards'], r)), recs)))

        evals = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['evals'], r)), recs)))
        eval_masks = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['eval_masks'], r)), recs)))

        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals,
                'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs)))
    ret_dict['is_train'] = torch.FloatTensor(list(map(lambda x: x['is_train'], recs)))

    return ret_dict


def get_loader(batch_size=64, shuffle=True):
    data_set = MySet()
    data_iter = DataLoader(dataset=data_set, \
                           batch_size=batch_size, \
                           num_workers=4, \
                           shuffle=shuffle, \
                           pin_memory=True, \
                           collate_fn=collate_fn
                           )
    data_iterte = DataLoader(dataset=data_set, \
                             batch_size=batch_size, \
                             num_workers=4, \
                             shuffle=False, \
                             pin_memory=True, \
                             collate_fn=collate_fn
                             )

    # n线程数，shuffle打乱数据，collat一个batch的样本打包成一个tensor的结构

    return data_iter, data_iterte
