import torch
import sys, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datareader
import sharedutils
import graphgallery as gg
import graphgallery.functional as gf
from sklearn import metrics
from thop import profile
from thop import clever_format

def fgcn_run(dataname="cora", random_seed=2020, train_epoch=100, lr=0.2, dropout=0.5):
    rd = datareader.ReadData()
    data = rd.read_raw_data(dataname)
    tag = data.graph.y
    mat_index = np.arange(0, len(tag))
    tr_ind, val_ind, ts_ind = rd.get_random_ind_tensor(mat_index, tag, 0.2)
    print("length of all,tr,val,ts: ", len(tag), len(tr_ind), len(val_ind), len(ts_ind))

    gg.set_backend('torch')
    fgcn = gg.gallery.FastGCN(data.graph, device="cuda:0").process().build(dropout=dropout, lr=lr)
    fgcn.train(tr_ind, val_ind, epochs=train_epoch)
    result = fgcn.test(ts_ind)
    return float(result.accuracy)

def model_startup(dataname, runs, **model_params):
    """模型初始化"""
    allconfs, conf = sharedutils.read_config("./models_conf.json"), {}
    conf.update(allconfs['shared_conf'])
    conf.update(allconfs['fgcn'][dataname])
    conf.update(model_params)

    scores = []
    for run in range(runs):
        score = fgcn_run(dataname = dataname,
                   train_epoch   = conf["train_epoch"], 
                   lr            = conf["learning_rate"], 
                   dropout       = conf["dropout"])
        scores.append(score)
        if score <= 0.7: break
        if score <= 0.8 and dataname != "citeseer": break
    return np.mean(scores), np.std(scores)


def search_params(dataname, log_dir):
    learning_rate = np.linspace(0.001, 0.2, 20)
    dropout = np.array([0.5])
    params_set = {"learning_rate": learning_rate, "dropout": dropout}

    best_score, std, best_params = sharedutils.grid_search(dataname, 10, params_set, model_startup)
    msg = "fgcn; %s; best_score, std, best_params %s %s %s\n" % (dataname, best_score, std, best_params)
    print(msg)
    sharedutils.add_log(os.path.join(log_dir, "fgcn_search.log"), msg)



if __name__ == '__main__':
    print('* Set parameters in models_conf.json, such as device": "cuda:0"')
    do_search_params = input('search params? "yes" or "no", default("no"): ') or "no"
    dataname=input("cora/citeseer/amazon_photo/pubmed, default(cora): ") or "cora"
    runs = int(input("nums of runs, default(1): ") or "1")
    if do_search_params == "yes":
        allconfs, conf = sharedutils.read_config("./models_conf.json"), {}
        search_params(dataname, allconfs['fgcn']['log_dir'])
    else:
        me, st = model_startup(dataname=dataname, runs=runs)
        print("acc_averages %04d times: means: %04f std: %04f" % (runs, me, st))
    
