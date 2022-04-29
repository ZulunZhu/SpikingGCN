import sys
sys.path.append('/home/zlzhu/snn/spikingjelly')
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, encoding, functional

thresh, lens, decay, if_bias = (0.5, 0.5, 0.2, True)

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply

def mem_update(ops, x, mem, spike, lif):
    mem =  mem * decay * (1. - spike)+ops(x) #mem * decay * (1. - spike) +
    # print("mem:",mem[0][0][0])
    if lif!=None:
        spike = lif.forward(mem)
    else:
        spike = mem
        # spike = act_fun(mem)
    return mem, spike

class LIS_model(nn.Module):
    def __init__(self, opt):
        super(LIS_model, self).__init__()
        self.batch_size = opt.batch_size
        self.dts = opt.dts
        self.fc = (128, 10)
        self.tau = 80
        self.v_threshold = 0.2
        self.v_reset = None

        if self.dts == 'MNIST' or self.dts == 'Fashion-MNIST':
            self.cnn = ((1, 32, 3, 1, 1, 5, 2), (32, 32, 3, 1, 1, 5, 2))
            self.kernel = (28, 14, 7)
        elif self.dts == 'NMNIST':
            self.cnn = ((2, 64, 3, 1, 1, 5, 2), (64, 64, 3, 1, 1, 5, 2))
            self.kernel = (36, 18, 9)

        self.conv1 = nn.Conv2d(self.cnn[0][0], self.cnn[0][1], kernel_size = self.cnn[0][2], stride = self.cnn[0][3], padding = self.cnn[0][4], bias = if_bias)
        self.conv2 = nn.Conv2d(self.cnn[1][0], self.cnn[1][1], kernel_size = self.cnn[1][2], stride = self.cnn[1][3], padding = self.cnn[1][4], bias = if_bias)
        self.fc1 = nn.Linear(self.kernel[-1] * self.kernel[-1] * self.cnn[-1][1], self.fc[0], bias = if_bias)
        self.fc2 = nn.Linear(self.fc[0], self.fc[1], bias = if_bias)

        self.lif_layer1 = neuron.IFNode(v_threshold=self.v_threshold, v_reset=self.v_reset)
        self.lif_layer2 = neuron.IFNode(v_threshold=self.v_threshold, v_reset=self.v_reset)
        self.lif_layer3 = neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, v_reset=self.v_reset)
        self.lif_layer4 = neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, v_reset=self.v_reset)
        
        
        # self.lif_layer1 = None
        # self.lif_layer2 = None
        # self.lif_layer3 = None
        # self.lif_layer4 = None


    def forward(self, input, time_window = 30):
        c1_mem = c1_spike = torch.zeros(self.batch_size, self.cnn[0][1], self.kernel[0], self.kernel[0]).cuda()
        c2_mem = c2_spike = torch.zeros(self.batch_size, self.cnn[1][1], self.kernel[1], self.kernel[1]).cuda()
        
        
        h1_mem = h1_spike = h1_sumspike = torch.zeros(self.batch_size, self.fc[0]).cuda()
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, self.fc[1]).cuda()
        for step in range(time_window):
            if self.dts == 'MNIST' or self.dts == 'Fashion-MNIST':
                # x = input > torch.rand(input.size()).cuda()
                x = input
            elif self.dts == 'NMNIST':
                x = input[:, :, :, :, step]
            # print("x.shape1:",x.shape)
            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike, self.lif_layer1)
            # print("c1_spike:",c1_spike)
            x = F.avg_pool2d(c1_spike, 2)
            # print("x.shape2:",x.shape)
            

            c2_mem, c2_spike = mem_update(self.conv2, x, c2_mem,c2_spike, self.lif_layer2)
            conv_image = torch.zeros(c1_spike.shape)
            if step ==1: conv_image = c1_spike.detach().cpu()
            else: conv_image += c1_spike.detach().cpu()

            x = F.avg_pool2d(c2_spike, 2)
            
        
            x = x.view(self.batch_size, -1)

            # print("x.shape3:",x.shape)
            

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike, self.lif_layer3)
            h1_sumspike += h1_spike
         
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike, self.lif_layer4)
            # h2_spike = self.lif_layer4.forward(h2_mem)
            h2_sumspike += h2_spike #h2_spike
            # print("h2_spike:",h2_spike)
        outputs = h2_sumspike / time_window
        conv_image = conv_image/time_window
        return outputs, conv_image