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
from graphgallery.nn.layers.pytorch.conv.gat import GraphAttention
from graphgallery.nn.layers.pytorch.conv.gcn import GraphConvolution

def count_graph_conv(m, datain, dataout):
  # your rule here
  att, adj = datain
  fc = nn.Linear(m.in_channels, m.out_channels, bias=False)
  h = fc(att)
  m_mu, m_nu, m_tu = h.shape[0], h.shape[1], np.count_nonzero(h)
  n_mu, n_nu, n_tu = adj.shape[0], adj.shape[1], adj._nnz()
  ops1 = adj.shape[0] * adj.shape[1] * m.out_channels
  ops2 = int(m_mu * n_nu + m_tu * n_tu / n_mu)
  m.total_ops += ops1 + ops2

def count_graph_attention(m, datain, dataout):
  x, adj = datain
  if adj.layout != torch.strided:
    adj = adj.to_dense()
  attnum_heads = m.attnum_heads

  outputs = []
  for head in range(m.attnum_heads):
    W, a1, a2 = m.kernels[head], m.attn_kernel_self[head], m.attn_kernel_neighs[head]
    Wh =  x  @ W
    f_1 = Wh @ a1
    f_2 = Wh @ a2
    m.total_ops += x.shape[0] * x.shape[1] * W.shape[1]
    m.total_ops += Wh.shape[0] * Wh.shape[1] * a1.shape[1]
    m.total_ops += Wh.shape[0] * Wh.shape[1] * a2.shape[1]

    e = m.leakyrelu(f_1 + f_2.transpose(0, 1))
    default_alpha = 0.2
    seq = nn.Sequential(nn.LeakyReLU(default_alpha))
    tmacs, tparms = profile(seq, inputs=(f_1 + f_2.transpose(0, 1), ))
    m.total_ops += tmacs

    zero_vec = -9e15 * torch.ones_like(e)
    attention = torch.where(adj > 0, e, zero_vec)
    calc_attention = attention.clone()
    attention = F.softmax(attention, dim=1)
    attention = F.dropout(attention, m.dropout, training=False)
    h_prime = torch.matmul(attention, Wh)

    m.total_ops += attention.shape[0] * attention.shape[1] * Wh.shape[1]

    if m.use_bias: h_prime += m.biases[head]
    outputs.append(h_prime)

  if m.reduction == 'concat': output = torch.cat(outputs, dim=1)
  else: output = torch.mean(torch.stack(outputs), 0)
  ret = m.activation(output)
  return ret


def calc(dataname):
  gg.set_backend("pytorch")
  data = NPZDataset(dataname, root="~/datasets/datafromgg")
  splits = data.split_nodes()
  method = gg.gallery.GAT(data.graph, device="cpu").process().build(lr=0.02)
  method.train(splits.train_nodes, splits.val_nodes, epochs=1)
  acc = method.test(splits.test_nodes).accuracy

  method.model.eval()
  macs, params = profile(method.model, 
                        inputs=(method.cache.X, method.cache.A),
                        custom_ops={GraphConvolution: count_graph_conv,
                                    GraphAttention: count_graph_attention})
  print("num of nodes: ", data.graph.num_nodes)
  macs = macs / data.graph.num_nodes
  macs, params = clever_format([macs, params], "%.4f")
  print('dataname: macs, params: ', dataname, macs, params)
  return macs, params

answers = []
for name in ["cora", "acm", "citeseer", "pubmed"]:
# for name in ["pubmed"]:
  answers.append(calc(name))
for i in answers: print(i)