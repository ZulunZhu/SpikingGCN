import torch
import sys, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datareader
import sharedutils
import graphgallery as gg
from sklearn import metrics

def sgc_run(dataname="cora", random_seed=2020, train_epoch=100, lr=0.2, dropout=0.5, weight_decay=5e-5):
    rd = datareader.ReadData()
    data = rd.read_raw_data(dataname)
    print(data.graph.A.nnz)
    tag = data.graph.y
    mat_index = np.arange(0, len(tag))
    data_split = rd.get_fixed_splited_data(dataname)
    # tr_ind, val_ind, ts_ind = data_split.split_nodes().train_nodes, data_split.split_nodes().val_nodes, data_split.split_nodes().test_nodes
    tr_ind, val_ind, ts_ind = rd.get_random_ind_tensor(mat_index, tag, 0.2)
    print("length of all,tr,val,ts: ", len(tag), len(tr_ind), len(val_ind), len(ts_ind))


    gg.set_backend('torch')
    sgc = gg.gallery.SGC(data_split.graph.to_undirected()).process(attr_transform="normalize_attr").build()
    sgc.train(tr_ind, val_ind, epochs=train_epoch)
    result = sgc.test(ts_ind)
    return float(result.accuracy)

def model_startup(dataname, runs, **model_params):
    """模型初始化"""
    allconfs, conf = sharedutils.read_config("./models_conf.json"), {}
    # public conf
    conf.update(allconfs['shared_conf'])
    # private conf
    conf.update(allconfs['sgc'][dataname])
    # other conf
    conf.update(model_params)

    scores = []
    for run in range(runs):
        score = sgc_run(dataname = dataname,
                   train_epoch   = conf["train_epoch"], 
                   lr            = conf["learning_rate"], 
                   dropout       = conf["dropout"], 
                   weight_decay  = conf["weight_decay"])
        scores.append(score)
        # # just in case
        # if score <= 0.7: break
        # if score <= 0.8 and dataname != "citeseer": break
    return np.mean(scores), np.std(scores)


def search_params(dataname, log_dir):
    learning_rate = np.linspace(0.001, 0.2, 20)
    dropout = np.array([0.5])
    # weight_decay = np.array([5e-5])
    # params_set = {"learning_rate": learning_rate, "dropout": dropout, "weight_decay": weight_decay}
    params_set = {"learning_rate": learning_rate, "dropout": dropout}

    best_score, std, best_params = sharedutils.grid_search(dataname, 10, params_set, model_startup)
    msg = "sgc; %s; best_score, std, best_params %s %s %s\n" % (dataname, best_score, std, best_params)
    print(msg)
    sharedutils.add_log(os.path.join(log_dir, "sgc_search.log"), msg)



if __name__ == '__main__':
    print('* Set parameters in models_conf.json, such as device": "cuda:0"')
    do_search_params = input('search params? "yes" or "no", default("no"): ') or "no"
    dataname=input("cora/citeseer/amazon_photo/pubmed, default(cora): ") or "cora"
    runs = int(input("nums of runs, default(1): ") or "1")
    if do_search_params == "yes":
        allconfs, conf = sharedutils.read_config("./models_conf.json"), {}
        search_params(dataname, allconfs['sgc']['log_dir'])
    else:
        me, st = model_startup(dataname=dataname, runs=runs)
        print("acc_averages %04d times: means: %04f std: %04f" % (runs, me, st))
    
