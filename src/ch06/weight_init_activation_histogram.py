# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


# 1000 x 100
input_data = np.random.randn(1000, 100)  # 1000个数据
node_num = 100  # 各隐藏层的节点（神经元）数
hidden_layer_size = 5  # 隐藏层有5层
activations = {}  # 激活值的结果保存在这里

x = input_data

for i in range(hidden_layer_size):
    # 上次激活的输出就是这次的输入x
    if i != 0:
        x = activations[i - 1]

    # 改变初始值进行实验
    #  两极分化
    # w = np.random.randn(node_num, node_num) * 1

    # 集中于0.5，表现力弱
    # w = np.random.randn(node_num, node_num) * 0.01

    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    a = np.dot(x, w)

    # 将激活函数的种类也改变，来进行实验！
    # z = sigmoid(a)
    z = ReLU(a)
    # z = tanh(a)

    activations[i] = z

# 绘制直方图
for i, a in activations.items():
    # 绘制子图形，这里是5个
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")

    # 由于i=0时已经绘制了左侧y轴刻度(locs+ticks),i>0不需要再次绘制，没必要
    if i != 0: plt.yticks([], [])
    # 比较ReLU时打开这行，否则看起来比例尺跨度太大
    plt.ylim(0, 7000)

    # 缩小x刻度标签避免重叠
    plt.tick_params(axis='x', labelsize=6)
    # 1.1是为了将1.0包含进来
    plt.xticks(np.arange(0, 1.1, step=0.2))

    # x, bins, range
    # x的值为a.flatten(),小长方形条的数量，x的取值范围为(0,1)
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()
