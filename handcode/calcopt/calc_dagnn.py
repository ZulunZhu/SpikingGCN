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
from graphgallery.nn.layers.pytorch.conv.gcn import GraphConvolution
from graphgallery.nn.layers.pytorch.conv.dagnn import PropConvolution

def count_graph_conv(m, datain, dataout):
  att, adj = datain
  fc = nn.Linear(m.in_channels, m.out_channels, bias=False)
  h = fc(att)
  m_mu, m_nu, m_tu = h.shape[0], h.shape[1], np.count_nonzero(h)
  n_mu, n_nu, n_tu = adj.shape[0], adj.shape[1], adj._nnz()
  ops1 = adj.shape[0] * adj.shape[1] * m.out_channels
  ops2 = int(m_mu * n_nu + m_tu * n_tu / n_mu)
  m.total_ops += ops1 + ops2

def count_prop_conv(m, datain, dataout):
  x, adj = datain
  propagations = [x]
  for _ in range(m.K):
    i, j, k = adj.shape[0], adj.shape[1], x.shape[1]
    x = torch.spmm(adj, x)
    propagations.append(x)
    m.total_ops += i * j * k

  h = torch.stack(propagations, axis=1)
  retrain_score = m.w(h)
  retrain_score = m.activation(retrain_score).permute(0,2,1).contiguous()
  i, j, k = retrain_score.shape[0], retrain_score.shape[1], h.shape[1]
  out = (retrain_score @ h).squeeze(1)
  m.total_ops += i * j * k
  return out

def calc(dataname):
  gg.set_backend("pytorch")
  data = NPZDataset(dataname, root="~/datasets/datafromgg")
  splits = data.split_nodes()
  method = gg.gallery.DAGNN(data.graph).process().build(lr=0.02)
  method.train(splits.train_nodes, splits.val_nodes, epochs=2)
  acc = method.test(splits.test_nodes).accuracy

  method.model.eval()
  macs, params = profile(method.model, 
                        inputs=(method.cache.X, method.cache.A),
                        custom_ops={PropConvolution: count_prop_conv})
  print("num of nodes: ", data.graph.num_nodes)
  macs = macs / data.graph.num_nodes
  macs, params = clever_format([macs, params], "%.4f")
  print('dataname: macs, params: ', dataname, macs, params)
  return macs, params


answers = []
for name in ["cora", "acm", "citeseer", "pubmed"]:
  answers.append(calc(name))
for i in answers: print(i)