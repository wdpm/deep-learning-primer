# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

# 通过和数值微分的梯度对比，确定反向传播是否正确。
grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))

# W1:5.552703265091946e-10
# b1:3.1008145785662957e-09
# W2:6.303131577993114e-09
# b2:1.407285739135622e-07

# W1:5.105883248486079e-10
# b1:2.9970475618986727e-09
# W2:7.051627497703427e-09
# b2:1.3937010810566308e-07
