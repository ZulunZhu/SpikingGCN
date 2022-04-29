import torch, sys
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import sharedutils, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils import data as Data
from img_model_lif_fc import model_lif_fc
import time
from tqdm import tqdm
# import faulthandler
# # 在import之后直接添加以下启用代码即可
# faulthandler.enable()
dataname = 'mnist'
conf, cnf = sharedutils.read_config(), {}
cnf.update(conf['shared_conf'])
cnf.update(conf['snn_img'][dataname])
cnf['log_dir'] = conf['snn_img']['log_dir']
if cnf['v_reset'] == -100: cnf['v_reset'] = None
    
train_dataset = dsets.MNIST(root = '~/datasets/mnist', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = dsets.MNIST(root = '~/datasets/mnist', train = False, transform = transforms.ToTensor())
train_data_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = cnf['batch_size'], shuffle = True)
test_data_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = cnf['batch_size'], shuffle = False)
print("train, valiadation,test's batch num:", len(train_data_loader), len(test_data_loader))
# for img, label in test_data_loader:
#     print(img)

ret = model_lif_fc(device=cnf["device"], dataset_dir=cnf["dataset_dir"],
                      dataname=dataname, batch_size=cnf["batch_size"], 
                      learning_rate=cnf["learning_rate"], T=cnf["T"], tau=cnf["tau"], 
                      v_reset=cnf["v_reset"], v_threshold=cnf["v_threshold"],
                      train_epoch=cnf["train_epoch"], log_dir=cnf["log_dir"], n_labels=10,
                      n_dim0=700, n_dim1=28, n_dim2=28, train_data_loader=train_data_loader,
                      test_data_loader=test_data_loader)
accuracy = ret