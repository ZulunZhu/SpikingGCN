import datareader, sharedutils, os
import torch
import numpy as np
import pandas as pd
import numpy as np,networkx as nx, scipy.sparse as sp

from sklearn.model_selection import train_test_split
from torch.utils import data as Data
from model_lif_fc_no_val import model_lif_fc
import pickle, random, math, sklearn
import gc
from scipy.special import softmax
def snn_run(dataname, **new_conf):
  conf, cnf = sharedutils.read_config(), {}
  cnf.update(conf['shared_conf'])
  cnf.update(conf['snn_al'][dataname])
  cnf.update(new_conf)
#1 random 2 sopt 3 predictive 4 combination

  #Set the path
  rd = datareader.ReadData("~/datasets/datafromgg")
  result_fp = os.path.join(os.getenv('PWD'), 'al_result_files')
  if not (os.path.isdir(result_fp)):
    os.mkdir(result_fp)
  
  # get fixed dataset，fixed split
  data_fixed = rd.get_fixed_splited_data(dataname)
  # data_fixed = rd.read_raw_data(dataname)
  data = data_fixed
  mat, adj_mat, features, all_x, tag = rd.conv_subgraph(data_fixed)
  


  if dataname=="citeseer" : mat = mat+0.05
  #initial the active learning
  random_seed = cnf['random_seed']
  result_fp = os.path.join(result_fp, 'SNN_AL-{0}-rs_{1}.p'.format("acm", random_seed))
  res_list = [{'vid':random_seed-1, 'test_acc':np.NaN}]
  n_sample_acquired = 1
  n_sample_budget = 50
  tr_mask = np.array([False]*features.shape[0])
  tr_mask[random_seed-1] = True
  #Get the laplacian of graph 
  laplacian = np.diag(np.sum(adj_mat, 1)) - adj_mat
  predCovCQ = np.zeros((len(laplacian), len(laplacian)))

  #Test whether it is a uncompleted process
  if os.path.isfile(result_fp):
      print ('Param. file already exists! Loading from {0}.'.format(result_fp))
      res_list = pickle.load(open(result_fp, 'rb'))
      print("res_list:", res_list)
      n_sample_acquired = len(res_list)-1
      print("Nodes loaded from file:",n_sample_acquired)
      ele = 0
      while ele < n_sample_acquired:
          tr_mask[res_list[ele]['vid']] = True
          ele+=1
      print("Train mask obtained!")


  while n_sample_acquired < n_sample_budget+1:
    tr_ind = all_x[tr_mask].flatten()
    tr_mat = mat[tr_ind]
    ts_ind = all_x[~tr_mask].flatten()
    ts_mat = mat[ts_ind]
    tr_tag = tag[tr_ind]
    ts_tag = tag[ts_ind]
    
    k = pd.DataFrame(mat)
    u = k.describe()
    print("tr_mat.shape()",u)
    
    
    # may be some new params
    cnf['log_dir'] = conf['snn']['log_dir']
    if cnf['v_reset'] == -100: cnf['v_reset'] = None
    train_data_loader, test_data_loader = rd.tr_ts_numpy2dataloader(tr_mat, ts_mat,
      tr_tag, ts_tag, batch_size=cnf["batch_size"])
      
    print("train, valiadation,test's batch num:", len(train_data_loader), len(test_data_loader))
    print("train, valiadation,test's shape:", tr_mat.shape, ts_mat.shape)
    n_nodes, n_feat, n_flat = mat.shape[0], mat.shape[1], 1
    print("data: %s, num_node_classes: %d" % (dataname, data.graph.num_classes))
    print(cnf)
    ret, spike_rate = model_lif_fc(device=cnf["device"], dataset_dir=cnf["dataset_dir"],
                      dataname=dataname, batch_size=cnf["batch_size"], 
                      learning_rate=cnf["learning_rate"], T=cnf["T"], tau=cnf["tau"], 
                      v_reset=cnf["v_reset"], v_threshold=cnf["v_threshold"],
                      train_epoch=cnf["train_epoch"], log_dir=cnf["log_dir"], n_labels=data.graph.num_classes,
                      n_dim0=n_nodes, n_dim1=n_flat, n_dim2=n_feat, train_data_loader=train_data_loader,
                      test_data_loader=test_data_loader)
    accuracy = ret
    spike_rate = sklearn.preprocessing.normalize(spike_rate, norm="l1")
    print("spike_rate:", spike_rate)
    #SOPT
    masks = np.reshape(~tr_mask, (-1,1)) & np.reshape(~tr_mask, (1,-1))
    predCovCQ[masks] = np.linalg.inv(laplacian[~tr_mask][:,~tr_mask]).flatten()
    acq_scores = np.sum(predCovCQ[~tr_mask][:, ~tr_mask], 1)/np.sqrt(np.diag(predCovCQ)[~tr_mask])
    to_label = all_x[~tr_mask][np.argmax(acq_scores)]
    print("Next node:", to_label)

    #predictive entropy
    # spike_rate = softmax(spike_rate, axis=1)
    # p_entropy = entropy(spike_rate)
    # to_label = all_x[~tr_mask][np.argmax(p_entropy)]
    # print(to_label)

    #Random
    # to_label = all_x[~tr_mask][random.randint(0,features.shape[0])]
    # print("Next node:", to_label)

    #Adding the res_list
    if n_sample_acquired == n_sample_budget:
        res_list[-1]['test_acc'] = accuracy
    else:
        res_list.append({'vid':-100, 'test_acc':np.NaN})
        res_list[-2]['test_acc'], res_list[-1]['vid'] = accuracy, to_label[0] 
        print("np.sum(tr_mask)", np.sum(tr_mask))
        print("n_sample_acquired",n_sample_acquired)
        assert np.sum(tr_mask) == n_sample_acquired, 'Num. of sample in tr_mask != n_sample_acquired'
        assert tr_mask[res_list[-1]['vid']]==False, 'Node {0} alrdy acq.'.format(res_list[-1]['vid'])
        tr_mask[res_list[-1]['vid']] = True
    print(n_sample_acquired, " nodes", "accuracy:", accuracy)

    n_sample_acquired += 1
    #Save the result list 
    print ('\nSaving result to {0}'.format(result_fp))
    pickle.dump(res_list, open(result_fp, 'wb'))
    del ret
    gc.collect()

  #calulate the active learning curve
  curve_rate = ALC_get(res_list)
  return curve_rate
  
def entropy(score):
    result = np.zeros(score.shape[0])
    ele = 0
    for P in score:
        entropy = 0
        for p in P:
            entropy += (-p)*math.log(p,2)
        result[ele] = entropy
        ele+=1
    return result

def ALC_get(res_list):
  i = 0
  accuracy_sum = 0
  while i < len(res_list):
      accuracy_sum += res_list[i]['test_acc']
      i+=1
  area = (accuracy_sum*2-res_list[0]['test_acc']-res_list[-1]['test_acc'])/2
  curve_rate = area/(49)
  print("ALC obtained!:", curve_rate)
  return curve_rate

def model_startup(dataname, runs, **new_conf):
  scores = []
  submits = []

  conc = False
  if conc:
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(10)
    for run in range(runs):
      obj = pool.submit(snn_run, dataname, **new_conf)
      submits.append(obj)
    pool.shutdown(wait=True)
    for sub in submits:
      scores.append(sub.result())
  else:
    for run in range(runs):
      score = snn_run(dataname, **new_conf)
      scores.append(score)
  # sharedutils.add_log(os.path.join("./tmpdir/snn/", "snn_search.log"), "-1.5: " + str(scores))
  return np.mean(scores), np.std(scores)

def search_params(dataname, runs, log_dir):
  params_set = {"learning_rate": np.array([0.01, 0.015, 0.02, 0.025, 0.03]),
                "T": np.array([200, 300, 400, 500]),
                # "tau": np.array([80, 100, 120]),
                # "v_reset": np.array([0.0, -1.0]),
                # "v_threshold": np.array([0.2, 0.4, 0.6, 0.8, 1.0])
                }

  best_score, std, best_params = sharedutils.grid_search(dataname, runs, params_set, model_startup)
  msg = "sgc; %s; best_score, std, best_params %s %s %s\n" % (dataname, best_score, std, best_params)
  print(msg)
  sharedutils.add_log(os.path.join(log_dir, "snn_search.log"), msg)

  
if __name__ == '__main__':
  print('* Set parameters in models_conf.json, such as device": "cuda:0"')
  # do_search_params = input('search params? "yes" or "no", default("no"): ') or "no"
  # dataname=input("cora/acm/citeseer/pubmed, default(cora): ") or "cora"
  # runs = int(input("nums of runs, default(1): ") or "1")
  do_search_params, dataname, runs = "no", "cora", 1

  if do_search_params == "yes":
    allconfs = sharedutils.read_config("./models_conf.json")
    search_params(dataname, runs, allconfs["snn"]["log_dir"])
  else:
    me, st = model_startup(dataname, runs)
    print("acc_averages %04d times: means: %04f std: %04f" % (runs, me, st))
