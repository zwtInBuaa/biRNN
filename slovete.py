import pandas as pd
import numpy as np
import json
import math

dic = {
    "forward": {},
    "backward": {},
    "label": 1,

}
'''values: list, indicating x_t \in R^d (after elimination)
    * masks: list, indicating m_t \in R^d
    * deltas: list, indicating \delta_t \in R^d
    * forwards: list, the forward imputation, only used in GRU_D, can be any numbers in our model
    * evals: list, indicating x_t \in R^d (before elimination)
    * eval_masks: list, indicating whether each value is an imputation ground-truth'''
fordic = {
    "values": [],
    "masks": [],
    "deltas": [],
    "forward": [],
    "evals": [],
    "eval_mask": []

}
df = pd.read_csv(
    "./ours/re15.csv",
    index_col="Datetime",
    parse_dates=True,
)
'''df = pd.read_csv(
            "./ours/50/test/truth.csv",
            index_col="Datetime",
            parse_dates=True,
        )'''
df_get = pd.read_csv(
    "./ours/miss30.csv",
    index_col="Datetime",
    parse_dates=True,
)
evalvalues = df.values  # 实际
mean = df.describe().loc["mean"].values
std = df.describe().loc["std"].values

# print(mean,std)
evalmasks = ~np.isnan(np.array(evalvalues))
evalmasks1 = evalmasks.astype('float')
evalmasks = evalmasks.astype('float')
evalvalues = df.fillna(0).values
evalvalues = (evalvalues - mean) / std
evalvalues1 = evalvalues
evalvalues = evalvalues * evalmasks

value = df_get.values  # 缺失
forwardvalue = pd.DataFrame(value).fillna(method='ffill').fillna(0.0).values

masks = ~np.isnan(np.array(value))
masks1 = masks.astype('int32')
masks = masks.astype('float')
value = df_get.fillna(0).values
value = (value - mean) / std
value = value * masks

evalmasks = evalmasks - masks

forwardvalue = (forwardvalue - mean) / std


# print(masks)
# print(len(value))

def delt(masks):
    [T, D] = masks.shape
    deltas = []

    for t in range(T):
        if t == 0:
            deltas.append(np.ones(D, dtype='float'))
        else:
            deltas.append(np.ones(D, dtype='float') + (1 - masks[t]) * deltas[-1])
    return deltas


# print(deltas)
deltas = delt(masks)


def split(value, start, end):
    values = []
    for i in range(start, end + 1):
        values.append(value[i])
    return np.array(values)


diclist = []
dictest = []

dic = {
    "forward": [],
    "backward": [],
    "label": 1,

}
n = 456
for i in range(0, len(value) - n + 1):

    #  if i ==0:

    '''for j in range(i,i+n):



        fordic["values"] = list(value[j])
        fordic["masks"] = list(masks[j])
        # fordic["deltas"]=list(deltas[i])
        fordic["evals"] = list(evalvalues[j])
        fordic["eval_masks"] = list(evalmasks[j])
        fordic["forwards"] = list(forwardvalue[j])




        bakdic["values"] = list(value[i+n-1 - j+i])
        bakdic["masks"] = list(masks[i+n-1 - j+i])
        # bakdic["deltas"]=list(deltas[i][::-1])
        bakdic["evals"] = list(evalvalues[i+n-1 - j+i])
        bakdic["eval_masks"] = list(evalmasks[i+n-1 - j+i])
        bakdic["forwards"] = list(forwardvalue[i+n-1 - j+i])
        dic["forward"].append(fordic)
        dic["backward"].append(bakdic)

    v = split(masks, i , i+n-1)
    k = delt(v)
   # print(len(k))
    for j in range(0, n):
        dic["forward"][j]["deltas"] = list(k[j])
        # dic["backward"][i]["deltas"] = list(k[i][::-1])
    v = v[::-1]
    k = delt(v)
    for j in range(0, n):
        dic["backward"][j]["deltas"] = list(k[j])
    dictest.append(dic)'''
    # else:
    for j in range(i, i + n):
        fordic = {
            "values": [],
            "masks": [],
            "deltas": [],
            "forwards": [],
            "evals": [],
            "eval_masks": []

        }
        bakdic = {
            "values": [],
            "masks": [],
            "deltas": [],
            "forwards": [],
            "evals": [],
            "eval_masks": []
        }
        fordic["values"] = list(value[j])
        fordic["masks"] = list(masks[j])
        # fordic["deltas"]=list(deltas[i])
        fordic["evals"] = list(evalvalues[j])
        fordic["eval_masks"] = list(evalmasks[j])
        fordic["forwards"] = list(forwardvalue[j])

        bakdic["values"] = list(value[i + n - 1 - j + i])
        bakdic["masks"] = list(masks[i + n - 1 - j + i])
        # bakdic["deltas"]=list(deltas[i][::-1])
        bakdic["evals"] = list(evalvalues[i + n - 1 - j + i])
        bakdic["eval_masks"] = list(evalmasks[i + n - 1 - j + i])
        bakdic["forwards"] = list(forwardvalue[i + n - 1 - j + i])
        dic["forward"].append(fordic)
        dic["backward"].append(bakdic)

    v = split(masks, i, i + n - 1)
    k = delt(v)
    # print(len(k))
    for j in range(0, n):
        dic["forward"][j]["deltas"] = list(k[j])
        # dic["backward"][i]["deltas"] = list(k[i][::-1])
    v = v[::-1]
    k = delt(v)
    for j in range(0, n):
        dic["backward"][j]["deltas"] = list(k[j])
    diclist.append(dic)

    dic = {
        "forward": [],
        "backward": [],
        "label": 1,

    }

print(len(diclist))
# print(len(dictest))


with open("./json/our.json", mode='w', encoding='utf-8') as f:
    json.dump(diclist, f)
