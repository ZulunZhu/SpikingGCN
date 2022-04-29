import sys, torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spikingjelly.clock_driven import neuron, encoding, functional
from os.path import join as pjoin
import sharedutils
import matplotlib.pyplot as plt
from bayesian_linear import bayesian_linear

def model_lif_fc(dataname, dataset_dir, device, batch_size,
                 learning_rate, T, tau, v_threshold, v_reset, train_epoch, log_dir, 
                 n_labels, n_dim0, n_dim1, n_dim2, train_data_loader,
                 val_data_loader, test_data_loader):
    # init the model
    num_hidden_units = [n_labels]
    dim_input = n_dim1 * n_dim2

    # theta0 = torch.ones(n_labels,n_dim1 * n_dim2)
    # theta0 = theta0.to(device)
    # theta0.requires_grad = True

    net =bayesian_linear(dim_input = dim_input, num_hidden_units= num_hidden_units,\
     tau=tau, v_threshold=v_threshold, v_reset=v_reset, device=device)
    net = net.to(device)

    #init the parameters \theta = [mean, std]
    weight_shape = net.get_weight_shape()
    theta = {}
    theta['mean'] = {}
    theta['logSigma'] = {}
    for key in weight_shape.keys():
        if 'b' in key:
            theta['mean'][key] = nn.Parameter(torch.zeros(weight_shape[key], device=device, requires_grad=True))
        else:
            theta['mean'][key] = nn.Parameter(torch.empty(weight_shape[key], device=device))
            torch.nn.init.xavier_normal_(tensor=theta['mean'][key], gain=np.sqrt(2))
            theta['mean'][key].requires_grad_()
        theta['logSigma'][key] = nn.Parameter(torch.rand(weight_shape[key], device=device) - 4)
        theta['logSigma'][key].requires_grad_()
    # print("theta:", theta)

    # Adam opt
    # optimizer = torch.optim.Adam([theta0], lr=learning_rate)
    optimizer = torch.optim.Adam(
        [
            {
                'params': theta['mean'].values(),
                'weight_decay': 0
            },
            {
                'params': theta['logSigma'].values(),
                'weight_decay': 0
            }
        ],
        lr=learning_rate
    )
    # print(optimizer.param_groups)
    # Encoder, actually Bernoulli sampling here
    encoder = encoding.PoissonEncoder()
    train_times = 0
    max_val_accuracy = 0
    model_pth = 'tmpdir/snn/best_snn.model'
    val_accs, train_accs, loss_sum = [], [], []

    for epoch in range(train_epoch):
        net.train()
        for rind, (img, label) in enumerate(train_data_loader):
            img = img.to(device)
            # print("img.size():",img.size())
            # for: RuntimeError: one_hot is only applicable to index tensor.
            label = label.long().to(device)
            label_one_hot = F.one_hot(label, n_labels).float()
            # print("label:", label.size(), "label_one_hot:", label_one_hot.size())
            optimizer.zero_grad()
            
            # T，out_spikes_counter is tensor with shape=[batch_size, num_labels]
            # 记录整个仿真时长内，输出层的num_labels个神经元的脉冲发放次数
            for t in range(T):
                w = net.sample_nn_weight(meta_params=theta)
                if t == 0: out_spikes_counter = net.forward(encoder(img).float(),w)
                else: out_spikes_counter += net.forward(encoder(img).float(),w)

            # out_spikes_counter / T 
            out_spikes_counter_frequency = out_spikes_counter / T

            # MSE
            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
            loss.requires_grad_()
            # print("loss:::", loss)
            loss.backward()
            loss_sum.append(loss)
            torch.nn.utils.clip_grad_norm_(
                parameters=theta['mean'].values(),
                max_norm=10
            )
            torch.nn.utils.clip_grad_norm_(
                parameters=theta['logSigma'].values(),
                max_norm=10
            )
            optimizer.step()
            # reset, because snn has memory
            functional.reset_net(net)

            # the max is the label predicted
            accuracy = (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().mean().item()
            
            train_accs.append(accuracy)

            train_times += 1

        net.eval()#固定网络层参数
        sharedutils.plot_array(train_accs)#画图
        #去掉验证集
        with torch.no_grad():
            # test_sum = 0
            # correct_sum = 0
            # for img, label in val_data_loader:
            #     img = img.to(device)
            #     n_imgs = img.shape[0]
            #     out_spikes_counter = torch.zeros(n_imgs, n_labels).to(device)
            #     for t in range(T):
            #         w = net.sample_nn_weight(meta_params=theta)
            #         out_spikes_counter += net.forward(encoder(img).float(),w)

            #     correct_sum += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
            #     test_sum += label.numel()
            #     functional.reset_net(net)
            # val_accuracy = correct_sum / test_sum
            # val_accs.append(val_accuracy)
            # if val_accuracy > max_val_accuracy:
            #     max_val_accuracy = val_accuracy
            if  epoch ==  (train_epoch-1):
                torch.save(net, model_pth)
        print(f'Epoch {epoch}: device={device}, dataset_dir={dataset_dir}, batch_size={batch_size}, learning_rate={learning_rate}, T={T}, log_dir={log_dir}, max_train_accuracy={train_accs[-1]:.4f},max_val_accuracy={max_val_accuracy:.4f}, train_times={train_times}', end="\r")
    
    # 测试集：
    best_snn = torch.load(model_pth)
    best_snn.eval()
    best_snn.to(device)
    max_test_accuracy = 0.0
    result_sops, result_num_spikes_1, result_num_spikes_2 = 0, 0, 0
    with torch.no_grad():
        functional.set_monitor(best_snn, True)
        test_sum, correct_sum = 0, 0
        for img, label in test_data_loader:
            img = img.to(device)
            n_imgs = img.shape[0]
            out_spikes_counter = torch.zeros(n_imgs, n_labels).to(device)
            denominator = n_imgs * len(test_data_loader)
            
            for t in range(T):
                w = net.sample_nn_weight(meta_params=theta)
                enc_img = encoder(img).float()
                out_spikes_counter += best_snn(enc_img,w)
                # pre spikes
                result_num_spikes_1 += torch.sum(enc_img) / denominator
            # post spikes
            result_num_spikes_2 += torch.sum(out_spikes_counter) / denominator
            correct_sum += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
            test_sum += label.numel()

            # voltages changes num
            # charge_v, fire_v = best_snn.lif_layer.monitor['h'], best_snn.lif_layer.monitor['v']
            # for i in range(1, T):
            #     charge_sops = np.sum(charge_v[i] != charge_v[i - 1])
            #     fire_sops   = np.sum(fire_v[i] != fire_v[i - 1])
            #     result_sops += (charge_sops + fire_sops) / denominator
            
            # monitor_label = (best_snn[-1].monitor, label.numpy())
            # sharedutils.dump_pickle(monitor_label, "tmpdir/snn/monitor_acm.pickle")
            functional.reset_net(best_snn)

        test_accuracy = correct_sum / test_sum
        max_test_accuracy = max(max_test_accuracy, test_accuracy)
    result_msg = f'testset\'acc: device={device}, dataset={dataname}, batch_size={batch_size}, learning_rate={learning_rate}, T={T}, max_test_accuracy={max_test_accuracy:.4f}'
    # result_msg += f", sops_per_nodes: {result_sops: .4f}"
    result_msg += f", num_s1: {int(result_num_spikes_1)}, num_s2: {int(result_num_spikes_2)}"
    result_msg += f", num_s_per_node: {int(result_num_spikes_1)+int(result_num_spikes_2)}"
    sharedutils.add_log(pjoin(log_dir, "snn_search.log"), result_msg)
    print(result_msg)
    return max_test_accuracy