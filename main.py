import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import datetime
import numpy as np
import time
import utils
import models
import argparse
import data_loader
import pandas as pd
import ujson as json
import math
from sklearn import metrics
import csv
import random
import os


def seed_torch(seed=1000):  # 1029,1030
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别

seed_torch()

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--model', type=str)
args = parser.parse_args()


def train(model):
    mae = []
    mre = []
    auc = []
    df = pd.read_csv(
        "./ours/re15.csv",
        index_col="Datetime",
        parse_dates=True,
    )
    mean = df.describe().loc["mean"].values
    std = df.describe().loc["std"].values

    optimizer = optim.Adam(model.parameters(),
                           lr=0.001)  # Adaptive Moment Estimation自适应矩估计，万金油式的优化器，使用起来非常方便，梯度下降速度快，但是容易在最优值附近震荡
    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False,
                                               threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    data_iter, data_te = data_loader.get_loader(batch_size=args.batch_size)  # 导入数据
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = (
            "./save/" + current_time + "result.txt"
    )
    note = open(foldername, mode='a')
    bestmae = 10000
    besrmse = 10000
    im = []
    ev = []
    for epoch in range(args.epochs):  # 训练
        model.train()  # 启用 batch normalization 和 dropout 保证 BN 层能够用到 每一批数据 的均值和方差随机取一部分 网络连接来训练更新参数
        run_loss = 0.0

        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)  # 变成变量
            ret = model.run_on_batch(data, optimizer)  # 使用lstm训练规定优化器
            run_loss += ret['loss'].data  # 累计损耗
            print('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter),
                                                                          run_loss / (idx + 1.0)))
            note.write(
                '\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter),
                                                                        run_loss / (idx + 1.0)))

        if epoch % 1 == 0:  # 评估
            b, c, imp, eva = evaluate(model, data_te, mean, std)
            note.write(
                '\r MAE:{}'.format(b))
            note.write(
                '\r RMRE:{}'.format(c))
            if bestmae > float(b):
                '''im=ret['imputations'].data.cpu().numpy()
                im=im*std+mean'''
                im = imp
                ev = eva

                bestmae = float(b)
                besrmse = float(c)
                # im=imp
                # ev=imp
    note.write(
        '\r BESTMAE:{}'.format(bestmae))
    note.write(
        '\r BESTRMRE:{}'.format(besrmse))
    note.close()
    print("birnn_n70h176:")
    print(bestmae)
    print(besrmse)
    print(im.shape)
    # np.save('./ours/result/M-RNN-in/70/im',im)
    # np.save('./ours/result/M-RNN-in/70/ev', ev)


# with open('./ours/result/M-RNN-i/in30.csv', 'w', encoding='utf-8', newline='') as f:
#     write = csv.writer(f)
#     write.writerows(im)


def evaluate(model, val_iter, mean, std):
    model.eval()  # 模型中有BatchNormalization和Dropout，在预测时使用model.eval()后会将其关闭以免影响预测结果
    labels = []
    preds = []
    evals = []
    imputations = []
    count = 0

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)
        # pred = ret['predictions'].data.cpu().numpy()#拿到cpu并且转化为numpy的array
        # label = ret['labels'].data.cpu().numpy()
        # is_train = ret['is_train'].data.cpu().numpy()
        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        eval_ = eval_ * std
        imputation = ret['imputations'].data.cpu().numpy()  # 数据导出
        imputation = imputation * std
        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()
        count += np.abs(eval_masks).sum()
        # collect test label & prediction
        # pred = pred[np.where(is_train == 0)]
        # label = label[np.where(is_train == 0)]#测试集计算mae和mre
    # labels += label.tolist()
    # preds += pred.tolist()
    # labels = np.asarray(labels).astype('int32')
    # preds = np.asarray(preds)
    # print('AUC {}'.format(metrics.roc_auc_score(labels, preds)))
    evals = np.asarray(evals)
    imputations = np.asarray(imputations)
    # print(evals)
    # print(imputations)
    print('MAE', np.abs(evals - imputations).sum() / count)
    print('RMSE', math.sqrt(np.sum((evals - imputations) ** 2) / count))

    return np.abs(evals - imputations).sum() / count, math.sqrt(
        np.sum((evals - imputations) ** 2) / count), imputations, evals  # metrics.roc_auc_score(labels, preds),
    # note = open('x.txt', mode='a')
    # note.write('\n')
    # note.write('AUC {}'.format(metrics.roc_auc_score(labels, preds))+'\n')
    # note.write('MAE' + str(np.abs(evals - imputations).mean())+'\n')
    # note.write('MRE'+str(np.abs(evals - imputations).sum() / np.abs(evals).sum())+'\n')
    # note.write('\n')
    # note.close()


def run():
    model = getattr(models, args.model).Model()  # 指定调用model

    if torch.cuda.is_available():
        model = model.cuda('cuda:0')

    train(model)


if __name__ == '__main__':
    run()
