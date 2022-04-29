import sys, torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spikingjelly.clock_driven import neuron, encoding, functional,layer
from os.path import join as pjoin
import sharedutils
import matplotlib.pyplot as plt
import gc
import time
from tqdm import tqdm
from img_linear import img_linear
# from memory_profiler import profile
# @profile




# !!!This model file only contain train and test, compare with SpikingGCN's model!!!

def model_lif_fc(dataname, dataset_dir, device, batch_size,
                 learning_rate, T, tau, v_threshold, v_reset, train_epoch, log_dir, 
                 n_labels, n_dim0, n_dim1, n_dim2, train_data_loader, test_data_loader):
    # init
    net =img_linear(batch_size=batch_size,tau=tau, v_threshold=v_threshold, v_reset=v_reset, device=device)
    net = net.to(device)
    # Adam opt
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # Encoder, actually Bernoulli sampling here
    encoder = encoding.BiPoissonEncoder()
    train_times = 0
    max_val_accuracy = 0
    model_pth = 'tmpdir/snn/best_snn.model'
    train_accs = []

    for epoch in range(train_epoch):
        net.train()
        start_time = time.time()
        for img, label in tqdm(train_data_loader):
            img = img.to(device)
            # for: RuntimeError: one_hot is only applicable to index tensor.
            label = label.long().to(device)
            label_one_hot = F.one_hot(label, n_labels).float()
            optimizer.zero_grad()

            # T，out_spikes_counter is tensor with shape=[batch_size, num_labels]
            # 记录整个仿真时长内，输出层的num_labels个神经元的脉冲发放次数
            for t in range(T):
                if t == 0: out_spikes_counter = net(img.float())
                else: out_spikes_counter += net(img.float())

            # out_spikes_counter / T 
            out_spikes_counter_frequency = out_spikes_counter / T

            # MSE
            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
            loss.backward()
            optimizer.step()
            # reset, because snn has memory
            functional.reset_net(net)

            # the max is the label predicted
            accuracy = (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().mean().item()
            
            train_accs.append(accuracy)

            train_times += 1

        net.eval()
        # sharedutils.plot_array(train_accs)
        torch.no_grad()
        print(f'Epoch {epoch}: device={device}, batch_size={batch_size}, learning_rate={learning_rate}, T={T}, max_train_accuracy={train_accs[-1]:.4f},loss={loss:.4f}, time consumed={time.time()-start_time},train_time={train_times}', end="\r")
        # 测试集：
        best_snn = net
        best_snn.eval()
        best_snn.to(device)
        max_test_accuracy = 0.0
        result_sops, result_num_spikes_1, result_num_spikes_2 = 0, 0, 0
        with torch.no_grad():
            functional.set_monitor(best_snn, True)
            spike_matrix = np.empty(shape=(0,n_labels))
            test_sum, correct_sum = 0, 0
            
            for img, label in test_data_loader:
                img = img.to(device)
                n_imgs = img.shape[0]
                out_spikes_counter = torch.zeros(n_imgs, n_labels).to(device)
                denominator = n_imgs * len(test_data_loader)
                for t in range(T):
                    enc_img = img.float()
                    out_spikes_counter += best_snn(enc_img)
                    # pre spikes
                    result_num_spikes_1 += torch.sum(enc_img) / denominator
                # post spikes
                result_num_spikes_2 += torch.sum(out_spikes_counter) / denominator
                correct_sum += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
                test_sum += label.numel()

                # voltages changes num
                # charge_v, fire_v = best_snn[-1].monitor['h'], best_snn[-1].monitor['v']
                # for i in range(1, T):
                #     charge_sops = np.sum(charge_v[i] != charge_v[i - 1])
                #     fire_sops   = np.sum(fire_v[i] != fire_v[i - 1])
                #     result_sops += (charge_sops + fire_sops) / denominator
                
                # monitor_label = (best_snn[-1].monitor, label.numpy())
                # sharedutils.dump_pickle(monitor_label, "tmpdir/snn/monitor_acm.pickle")
                counter_matrix = out_spikes_counter.cpu().numpy()
                spike_matrix = np.concatenate((spike_matrix,counter_matrix),axis=0)
                functional.reset_net(best_snn)

            test_accuracy = correct_sum / test_sum
            max_test_accuracy = max(max_test_accuracy, test_accuracy)

            result_msg = f'testset\'Current acc: {max_test_accuracy:.4f}'
            # result_msg += f", sops_per_nodes: {result_sops: .4f}"
            # result_msg += f", num_s1: {int(result_num_spikes_1)}, num_s2: {int(result_num_spikes_2)}"
            # result_msg += f", num_s_per_node: {int(result_num_spikes_1)+int(result_num_spikes_2)}"
            # sharedutils.add_log(pjoin(log_dir, "snn_search.log"), result_msg)
            print(result_msg)

    del net; del optimizer
    gc.collect()
    return max_test_accuracy, spike_matrix