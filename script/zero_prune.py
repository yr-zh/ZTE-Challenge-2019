# coding:utf-8

"""
把低于阈值的卷积核置0
"""

import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np
import matplotlib.pyplot as plt
import shutil


def weight_0(prototxt, model, layer, threshold):
    caffe.set_mode_gpu()
    net = caffe.Net(prototxt, model, caffe.TEST)
    weight = net.params[layer][0].data
    bias = net.params[layer][1].data

    sum_l1 = []
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            sum_l1.append((i, j, np.sum(abs(weight[i, j, :,
                                            :]))))  # i是核的顺序,j是每个卷积核与前面某个channel的连接顺序,求出每个连接的类似于L1范数的权重和,加上i,j是为了后续判断weight的时候好直接处理到原weight

    # display(sum_l1, 32)  # 从小到大排序后打印出前128个L1范数

    # l1_plot(sum_l1)  # 画出L1范数关于out*input的坐标图,以确定多少个需要修剪.

    weight_l1 = []
    for i in sum_l1:
        weight_l1.append(i[2])  # 得到仅含有l1范数的列表
    for i, weight_sum in enumerate(weight_l1):
        if weight_sum < threshold:
            out_channel_sort = sum_l1[i][0]
            input_channel_sort = sum_l1[i][1]
            weight[out_channel_sort, input_channel_sort, :, :] = 0  # 小于阈值的,weight置0
    net.save(root + "TestModel_zero.caffemodel")


def l1_plot(weight_l1):
    weight_l1_n = []
    for i in weight_l1:
        weight_l1_n.append(i[2])
    weight_l1_n.sort()
    x = [i for i in range(len(weight_l1_n))]
    plt.plot(x, weight_l1_n)
    plt.legend()
    plt.show()


def display(weight_l1, threshold):
    weight_l1_n = []
    for i in weight_l1:
        weight_l1_n.append(i[2])
    weight_l1_n.sort()
    print weight_l1_n
    # print [weight_l1_n[i] for i in range(threshold)]


root = "../models/"
prototxt = root + "TestModel_prune.prototxt"
pro_n = root + "TestModel_zero.prototxt"
shutil.copyfile(prototxt, pro_n)
model = root + "TestModel_prune.caffemodel"
layers = ['conv1_1_1','conv1_2_1','conv1_2_2','conv1_3_1', 'conv1_3_2', 'conv1_3_3',
          'conv2_1','conv2_2','conv2_3','conv2_4','conv2_5','conv2_6','conv2_7','conv2_8',
                'conv1_1_1','conv1_2_1','conv1_2_2','conv1_3_1', 'conv1_3_2', 'conv1_3_3',
                'conv3_1_1','conv3_1_2','conv3_2_1','conv3_2_2','conv3_3_1','conv3_3_2',
                'conv3_4_1','conv3_4_2','conv3_5_1','conv3_5_2','conv3_6_1','conv3_6_2',
                'conv4_1_1','conv4_1_2','conv4_2_1','conv4_2_2','conv4_3_1','conv4_3_2',
                'conv4_4_1','conv4_4_2','conv4_5_1','conv4_5_2','conv4_6_1','conv4_6_2',
                'conv5_1_1','conv5_1_2','conv5_2_1','conv5_2_2','conv5_3_1','conv5_3_2',
                'conv5_4_1','conv5_4_2','conv5_5_1','conv5_5_2','conv5_6_1','conv5_6_2']
for layer in layers:

    weight_0(prototxt, model, layer, 1e-4)
