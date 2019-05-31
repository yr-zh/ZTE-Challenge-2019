# coding=utf-8
import numpy as np
import os
import caffe
os.environ['GLOG_minloglevel'] = '2'


class fc2conv(object):
    def __init__(self, net):
        self._net = net

    def del_inputs(self, del_kernels):
        params = ['fc_svd_v']
        fc_params = {pr: (self._net.params[pr][0].data, self._net.params[pr][1].data) for pr in params}
        for fc in params:
            print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape,
                                                                                       fc_params[fc][1].shape)
        root = "../my_model/"
        # 导入全卷积网去移植的参数
        net_full_conv = caffe.Net(root + 'TestModel_fc2conv.prototxt',
                                  root + 'TestModel.caffemodel',
                                  caffe.TEST)
        params_full_conv = ['fc_conv']
        # conv_params = {name: (weights, biases)}
        conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

        for conv in params_full_conv:
            print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape,
                                                                                       conv_params[conv][1].shape)

        for pr, pr_conv in zip(params, params_full_conv):
            conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
            conv_params[pr_conv][1][...] = fc_params[pr][1]
            # conv_data = np.delete(conv_params[pr_conv][0], del_kernels, axis=1)
            # del_kernels = [i for i in range(768)]
            conv_data = np.delete(conv_params[pr_conv][0], del_kernels, axis=1)
            print(conv_data.shape)
            conv_data = conv_data.reshape(180, -1)
            print(conv_data.shape)
        return conv_data
