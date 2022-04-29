import graphgallery as gg 
from graphgallery.datasets import NPZDataset
import pandas as pd

"""
names = ["cora", "citeseer", "amazon_photo", "pubmed"]

for name in names:
  data = NPZDataset(name, root="~/datasets/datafromgg")
  print(data.avaliable())
  print(data.graph)
  print("num edges:", data.graph.num_edges)
  print("num class:", data.graph.num_classes)
  sp = data.graph.num_edges / (data.graph.num_nodes ** 2)
  print("sp:", sp)
  # df = pd.DataFrame(data.graph.x)
"""


names = ['amazon_photo', 'dblp', 'cora', 'citeseer_full', 'karate_club', 'blogcatalog', 'cora_ml', 'polblogs', 'flickr', 'coauthor_cs', 'amazon_cs', 'uai', 'coauthor_phy', 'citeseer', 'pubmed', 'acm', 'cora_full']

res = []

tinfo = '|name|num_nodes|num_edges|num_attrs|num_graphs|num_classes|is_directed|'
ttinfo= '|:---:|:---:|:---:|:---:|:---:|:---:|:---:|'
for name in names:
  data = NPZDataset(name, root="~/datasets/datafromgg")
  G = data.graph
  sp = G.num_edges/(G.num_nodes**2) * 100.0
  sp_str = "%.4f" % (sp) + "%"
  numnodes = format(G.num_nodes, ',')
  numedges = format(G.num_edges, ',')
  numattrs = format(G.num_attrs, ',')
  info = f'|{name}|{numnodes}|{numedges}|{numattrs}|{sp_str}|{G.num_classes}|{int(G.is_directed())}|'
  res.append([int(G.num_nodes), info])

res.sort(key = lambda x : (x[0]))
print(tinfo)
print(ttinfo)
for i in res:
    print(i[1])
