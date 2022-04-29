import torch
import collections
import numpy as np
import torch.nn as nn
from spikingjelly.clock_driven import neuron, encoding, functional
import torch.nn.functional as F
def mem_update(ops,x,lif=None):
    x = ops(x)
    if lif!=None:
        # print("x:",x[0])
        x = lif.forward(x)
        # print("spike:",x[0])
    return x
class img_linear(torch.nn.Module):
    def __init__(self, batch_size,tau, v_threshold, v_reset, device):
        super(img_linear, self).__init__()
        # self.num_hidden_units = num_hidden_units
        # self.num_layers = len(num_hidden_units)-1
        # self.dim_input = dim_input
        # self.num_hidden_units = num_hidden_units
        self.tau = tau
        self.batch_size = batch_size
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.device = device
        # self.num_layers = len(num_hidden_units)
        self.if_bias = True
        self.cnn = ((1, 32, 3, 1, 1, 5, 2), (32, 32, 3, 1, 1, 5, 2))
        self.kernel = (28, 14, 7)
        self.fc = (128, 10)
        self.conv1 = nn.Conv2d(self.cnn[0][0], self.cnn[0][1], kernel_size = self.cnn[0][2], stride = self.cnn[0][3], padding = self.cnn[0][4], bias = self.if_bias)
        self.conv2 = nn.Conv2d(self.cnn[1][0], self.cnn[1][1], kernel_size = self.cnn[1][2], stride = self.cnn[1][3], padding = self.cnn[1][4], bias = self.if_bias)
        self.fc1 = nn.Linear(self.kernel[-1] * self.kernel[-1] * self.cnn[-1][1], self.fc[0], bias = self.if_bias)
        # self.fc1 = nn.Linear(self.kernel[0] * self.kernel[0], self.fc[0], bias = self.if_bias)
        self.fc2 = nn.Linear(self.fc[0], self.fc[1], bias = self.if_bias)
        self.lif_layer1 = neuron.IFNode(v_threshold=self.v_threshold, v_reset=self.v_reset)
        self.lif_layer2 = neuron.IFNode(v_threshold=self.v_threshold, v_reset=self.v_reset)
        self.lif_layer3 = neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, v_reset=self.v_reset)
        self.lif_layer4 = neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, v_reset=self.v_reset)
    def forward(self, x):
        
        flatten = nn.Flatten()
        out=flatten.forward(x)
        c1_mem = c1_spike = torch.zeros(self.batch_size, self.cnn[0][1], self.kernel[0], self.kernel[0]).cuda()
        c2_mem = c2_spike = torch.zeros(self.batch_size, self.cnn[1][1], self.kernel[1], self.kernel[1]).cuda()

        h1_mem = h1_spike = h1_sumspike = torch.zeros(self.batch_size, self.fc[0]).cuda()
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, self.fc[1]).cuda()
        # print("x.shape1:",x.shape)
        c1_spike = mem_update(self.conv1,x)
        x = F.avg_pool2d(c1_spike, 2)
        # print("x.shape2:",x.shape)
        c2_spike = mem_update(self.conv2,x)
        x = F.avg_pool2d(c2_spike, 2)
        # print("x.shape3:",x.shape)
        x = x.view(self.batch_size, -1)
        h1_spike = mem_update(self.fc1,x)
        # print("x.shape4:",x.shape)
        # print("h1_spike:",h1_spike)
        h2_spike = mem_update(self.fc2,h1_spike,lif=self.lif_layer4)#lif=self.lif_layer4

        return h2_spike


