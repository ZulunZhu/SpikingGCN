import torch
import collections
import numpy as np
import torch.nn as nn
from spikingjelly.clock_driven import neuron, encoding, functional
class bayesian_linear(torch.nn.Module):
    def __init__(self,dim_input, num_hidden_units, tau, v_threshold, v_reset, device):
        super(bayesian_linear, self).__init__()
        # self.num_hidden_units = num_hidden_units
        # self.num_layers = len(num_hidden_units)-1
        self.dim_input = dim_input
        self.num_hidden_units = num_hidden_units
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.device = device
        self.num_layers = len(num_hidden_units)
        

    def forward(self, x, weight):
        self.weight=weight

        flatten = nn.Flatten()
        out=flatten.forward(x)
        w = self.weight['w1'].to(self.device)
        b = self.weight['b1'].to(self.device)
        # w.requires_grad_()
        # b.requires_grad_()
        # print("out.size():",out.size(),"self.weight:", w.size(),b.size())
        # print("self.weight:",self.weight.size())
        out = torch.nn.functional.linear(input=out, weight=w, bias=b)

        self.lif_layer = neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, v_reset=self.v_reset)
        # print("out0:",out)
        out = self.lif_layer.forward(out)
        print("out1:",torch.sum(out))
        return out

    def sample_nn_weight(self,meta_params):
        w = {}
        for key in meta_params['mean'].keys():
            eps_sampled = torch.randn_like(input=meta_params['mean'][key], device=self.device)
            w[key] = meta_params['mean'][key] + eps_sampled * torch.exp(meta_params['logSigma'][key])
        return w

    def get_weight_shape(self):
        w_shape = collections.OrderedDict()
        num_hidden_units = [self.dim_input]
        num_hidden_units.extend(self.num_hidden_units)

        for i in range(self.num_layers):
            w_shape['w{0:d}'.format(i + 1)] = (num_hidden_units[i + 1], num_hidden_units[i])
            w_shape['b{0:d}'.format(i + 1)] = num_hidden_units[i + 1]
        return w_shape
