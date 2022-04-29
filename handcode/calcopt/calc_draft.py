from torchvision.models import resnet50
from thop import profile
from thop import clever_format
import torch
import torch.nn as nn


class PipeModule(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    # do some operations.
    # suppose the opt number is product of the dimensions of x
    return x

def count_pipe_module(module, datain, dataout):
  ans = 1
  x = datain[0]
  for i in x.shape: ans = ans * i
  module.total_ops += ans

class myModule(nn.Module):
  def __init__(self):
    super().__init__()
    # self.other = PipeModule()
    self.fc = nn.Linear(60, 60, bias=False)

  def forward(self, x):
    for i in range(2):
      x = self.fc(x)
    return x

x = torch.rand(50, 60)
mm = myModule()
macs, params = profile(mm, inputs=(x, ), custom_ops={PipeModule:count_pipe_module})
print("calc result: ", macs, params)

seq = nn.Sequential(nn.Linear(60,60,bias=False))
x = torch.rand(50, 60)      
macs, params = profile(seq, inputs=(x, ))
print("macs2:", macs, type(macs))
