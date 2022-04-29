import torch
import sys, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import datareader, sharedutils
import graphgallery as gg
from sklearn import metrics
from sharedutils import add_log
from thop import profile
from thop import clever_format


def gcn_run(dataname="cora", random_seed=2020, train_epoch=100, lr=0.01, dropout=0.5, device="cuda:0"):
    """根据具体的参数进行实质的训练"""
    rd = datareader.ReadData("~/datasets/datafromgg")
    data = rd.read_raw_data(dataname)
    tag = data.graph.y
    mat_index = np.arange(0, len(tag))
    tr_ind, val_ind, ts_ind = rd.get_random_ind_tensor(mat_index, tag, 0.2)
    print("length of all,tr,val,ts: ", len(tag), len(tr_ind), len(val_ind), len(ts_ind))


    gg.set_backend('torch')
    gcn = gg.gallery.GCN(data.graph).process().build(lr=lr, dropout=dropout)
    gcn.train(tr_ind, val_ind, epochs=train_epoch)
    result = gcn.test(ts_ind)
    return float(result.accuracy)


def model_startup(dataname, runs, **model_params):
    """模型初始化"""
    allconfs, conf = sharedutils.read_config("./models_conf.json"), {}
    conf.update(allconfs['shared_conf'])
    conf.update(allconfs['gcn'][dataname])
    conf.update(model_params)

    scores = []
    for run in range(runs):
        score = gcn_run(dataname    = dataname, 
                        train_epoch = conf["train_epoch"], 
                        lr          = conf["learning_rate"], 
                        dropout     = conf["dropout"],
                        device      = conf["device"])
        scores.append(score)
        if score <= 0.7: break
        if score <= 0.8 and dataname != "citeseer": break
    return np.mean(scores), np.std(scores)


def search_params(dataname, log_dir):
    learning_rate = np.linspace(0.001, 0.02, 20)
    dropout = np.array([0.5])
    params_set = {"learning_rate": learning_rate, "dropout": dropout}
    best_score, std, best_params = sharedutils.grid_search(dataname, 10, params_set, model_startup)
    msg = "gcn; %s; best_score, std, best_params %s %s %s\n" % (dataname, best_score, std, best_params)
    print(msg)
    add_log(os.path.join(log_dir, "gcn_search.log"), msg)


if __name__ == '__main__':
    print('* Set parameters in models_conf.json, such as device": "cuda:0"')
    do_search_params = input('search params? "yes" or "no", default("no"): ') or "no"
    dataname=input("cora/citeseer/pubmed, default(cora): ") or "cora"
    runs = int(input("nums of runs, default(1): ") or "1")
    if do_search_params == "yes":
        allconfs, conf = sharedutils.read_config("./models_conf.json"), {}
        search_params(dataname, allconfs['gcn']['log_dir'])
    else:
        me, st = model_startup(dataname, runs)
        print("acc_averages %04d times: means: %04f std: %04f" % (runs, me, st))