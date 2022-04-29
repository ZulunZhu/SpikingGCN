import torch
import sys, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import graphgallery as gg
from graphgallery.datasets import NPZDataset
from sklearn import metrics
from thop import profile
from thop import clever_format

def calc(dataname):
  gg.set_backend("pytorch")
  data = NPZDataset(dataname, root="~/datasets/datafromgg")
  splits = data.split_nodes()
  method = gg.gallery.SGC(data.graph).process().build(lr=0.02)
  method.train(splits.train_nodes, splits.val_nodes, epochs=100)
  acc = method.test(splits.test_nodes).accuracy

  from graphgallery.nn.layers.pytorch.conv.gcn import GraphConvolution
  def count_your_model(m, datain, dataout):
    # your rule here
    att, adj = datain
    fc = nn.Linear(m.in_channels, m.out_channels, bias=False)
    h = fc(att)
    m_mu, m_nu, m_tu = h.shape[0], h.shape[1], np.count_nonzero(h)
    n_mu, n_nu, n_tu = adj.shape[0], adj.shape[1], adj._nnz()
    ops1 = adj.shape[0] * adj.shape[1] * m.out_channels
    ops2 = int(m_mu * n_nu + m_tu * n_tu / n_mu)
    m.total_ops += ops1 + ops2

  """
  macs, params = profile(method.model, 
                        inputs=(method.cache.X, method.cache.A))
  """
  method.model.eval()
  macs, params = profile(method.model, 
                        inputs=(method.cache.X[0], ))
                        # custom_ops={GraphConvolution: count_your_model})
  # print("num of nodes: ", data.graph.num_nodes)
  # macs = macs / data.graph.num_nodes
  macs, params = clever_format([macs, params], "%.4f")
  print('dataname: macs, params: ', dataname, macs, params)
  return macs, params

answers = []
for name in ["cora", "acm", "citeseer", "pubmed"]:
  answers.append(calc(name))
for i in answers: print(i)