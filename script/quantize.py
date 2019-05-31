# -*- coding:utf-8 -*-
# 通过Kmeans聚类的方法来量化权重
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.vq as scv
import pickle
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import time

# 获得各层的量化码表
def kmeans_net(net, layers, num_c=16, initials=None):
    # net: 网络
    # layers: 需要量化的层
    # num_c: 各层的量化级别
    # initials: 初始聚类中心
    codebook = {} # 量化码表
    if type(num_c) == type(1):
        num_c = [num_c] * len(layers)
    else:
        assert len(num_c) == len(layers)

    # 对各层进行聚类分析
    print "==============Perform K-means============="
    for idx, layer in enumerate(layers):
        print "Eval layer:", layer
        W = net.params[layer][0].data.flatten()
        W = W[np.where(W != 0)] # 筛选不为0的权重
        # 默认情况下，聚类中心为线性分布中心
        if initials is None:  # Default: uniform sample
            min_W = np.min(W)
            max_W = np.max(W)
            initial_uni = np.linspace(min_W, max_W, num_c[idx] - 1)
            codebook[layer], _ = scv.kmeans(W, initial_uni)
        elif type(initials) == type(np.array([])):
            codebook[layer], _ = scv.kmeans(W, initials)
        elif initials == 'random':
            codebook[layer], _ = scv.kmeans(W, num_c[idx] - 1)
        else:
            raise Exception

        # 将0权重值附上
        codebook[layer] = np.append(0.0, codebook[layer])
        print "codebook size:", len(codebook[layer])
        print("codebook: ", codebook[layer])
    return codebook


def quantize_net_with_dict(net, layers, codebook, use_stochastic=False, timing=False):
    start_time = time.time()
    codeDict = {} # 记录各个量化中心所处的位置
    maskCode = {} # 各层量化结果
    for layer in layers:
        print "Quantize layer:", layer
        W = net.params[layer][0].data
        if use_stochastic:
            codes = stochasitc_quantize2(W.flatten(), codebook[layer])
        else:
            codes, _ = scv.vq(W.flatten(), codebook[layer])
        W_q = np.reshape(codebook[layer][codes], W.shape)
        net.params[layer][0].data[...] = W_q

        maskCode[layer] = np.reshape(codes, W.shape)
        codeBookSize = len(codebook[layer])
        a = maskCode[layer].flatten()
        b = xrange(len(a))

        codeDict[layer] = {}
        for i in xrange(len(a)):
            codeDict[layer].setdefault(a[i], []).append(b[i])

    if timing:
        print "Update codebook time:%f" % (time.time() - start_time)

    return codeDict, maskCode

caffe.set_mode_gpu()
caffe.set_device(0)

# caffe_root = '../../'
model_dir = '../my_model/'
deploy = model_dir + 'TestModel.prototxt'
# solver_file = model_dir + 'TestModel.prototxt'
# model_name = 'LeNet5_Mnist_shapshot_iter_10000'
model_name = 'TestModel'
caffemodel = model_dir + model_name + '.caffemodel'

# dir_t = './weight_quantize/'

# 运行测试命令

start_time = time.time()

# solver = caffe.SGDSolver(solver_file)
# solver.net.copy_from(caffemodel)
test_net = caffe.Net(deploy, caffemodel, caffe.TEST)
# 需要量化的权重
total_layers = ['conv2_1','conv2_2','conv2_3','conv2_4','conv2_5','conv2_6','conv2_7','conv2_8',
                'conv1_1_1','conv1_2_1','conv1_2_2','conv1_3_1', 'conv1_3_2', 'conv1_3_3',
                'conv3_1_1','conv3_1_2','conv3_2_1','conv3_2_2','conv3_3_1','conv3_3_2',
                'conv3_4_1','conv3_4_2','conv3_5_1','conv3_5_2','conv3_6_1','conv3_6_2',
                'conv4_1_1','conv4_1_2','conv4_2_1','conv4_2_2','conv4_3_1','conv4_3_2',
                'conv4_4_1','conv4_4_2','conv4_5_1','conv4_5_2','conv4_6_1','conv4_6_2',
                'conv5_1_1','conv5_1_2','conv5_2_1','conv5_2_2','conv5_3_1','conv5_3_2',
                'conv5_4_1','conv5_4_2','conv5_5_1','conv5_5_2','conv5_6_1','conv5_6_2',
                'conv3_1_1b','conv4_1_1b','conv5_1_1b',]

num_c = 2 ** 8 # 量化级别，由8位整数表示
codebook = kmeans_net(test_net, total_layers, num_c)

codeDict, maskCode = quantize_net_with_dict(test_net, total_layers, codebook)
quantize_net_caffemodel = model_dir + model_name + '_quantize.caffemodel'
test_net.save(quantize_net_caffemodel)