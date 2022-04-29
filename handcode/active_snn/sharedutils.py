import time, os
import pickle, json
import numpy as np
import matplotlib
from matplotlib.pyplot import plot,savefig

def read_config(config_pth="./models_conf.json"):
  with open(config_pth, "r") as f:
    return json.load(f)

def add_log(pth, log):
  time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
  with open(pth, "a+") as f:
    f.write(time_now + ": " + str(log) + "\n")

def grid_search(dataname, runs, params, func):
  from itertools import product
  def generate_conf(confs):
    for conf in product(*confs.values()):
      yield {k:v for k,v in zip(confs.keys(),conf)}
  
  current_count = 0
  best_score, std_score, best_params = float('-inf'), -1, {}
  score_lis = []
  for params in generate_conf(params):
    score, std = func(dataname, runs, **params)
    current_count += 1
    print("::::: current_count:", current_count)
    if score > best_score: best_score, std_score, best_params = score, std, params
  return best_score, std_score, best_params



def load_pickle(file_path):
  with open(file_path, 'rb') as f:
    return pickle.load(f)
  return 1


def dump_pickle(data, file_path):
  with open(file_path, 'wb') as f:
    pickle.dump(data, f)
    return 0
  return 1

def plot_array(data):
  x=range(len(data))
  y=data
  matplotlib.use('Agg')

  plot(x,y,'--*b')
  savefig('tmpdir/snn/accuracy.jpg') 

