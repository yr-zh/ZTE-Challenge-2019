# coding=utf-8
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np
import shutil
import matplotlib.pyplot as plt
from script.fc2conv import fc2conv
from channel_vis import get_del_filter


class Prune(object):

    def __init__(self, net):
        self._net = net
        self.conv_data = {}
        self.ratio = 100                 # 裁剪掉多少卷积核
        self.base_layer = 'conv3_1_1b'    # 以哪一层为基础进行裁剪 一般是带b的层
        # self.del_filter = image_test(net, "../models/image/")


    def init_layer(self, name):
        conv_param = self._net.params[name]
        weight, bias = conv_param
        weight = weight.data
        bias = bias.data
        origin_channels = weight.shape[0]
        del_kernels = []

        self.conv_data[name] = weight, bias, del_kernels, origin_channels


    def _prune(self, name, conv_param, del_kernels=None, not_del_filters=False, del_filters=None):
        """

        :param name: 这一层的名字
        :param conv_param: 这一层的weight和bias
        :param del_kernels: 这一层要裁剪的通道数,由上一次得到  input减少
        :param not_del_filters:
        :param del_filters: 这一层要裁剪的卷积核,由这一层卷积核的数值得到, output减少
        :return:
        """
        weight, bias = conv_param
        weight = weight.data
        bias = bias.data
        origin_channels = weight.shape[0]

        if del_filters is not None:
            weight = np.delete(weight, del_filters, axis=0)
            bias = np.delete(bias, del_filters, axis=0)
        else:
            # delete filters 减少output维度
            if not not_del_filters:
                abs_mean = np.abs(weight).mean(axis=(1, 2, 3))
                del_filters = np.where(abs_mean < 1e-5)[0]
                weight = np.delete(weight, del_filters, axis=0)
                bias = np.delete(bias, del_filters, axis=0)
            else:
                del_filters = np.array([])
        print('\n')
        print(name)
        print(name + " filter nums need to delete is " + str(len(del_filters)))
        print(name + " filter nums need to preserve is " + str(origin_channels - len(del_filters)))

        # delete kernels  减少input维度
        if del_kernels is not None:
            weight = np.delete(weight, del_kernels, axis=1)
        print(weight.shape)
        return weight, bias, del_filters, origin_channels

    def _prune_rude(self, name, conv_param, del_kernels=None, del_filters=None):
        weight, bias = conv_param
        weight = weight.data
        bias = bias.data
        origin_channels = weight.shape[0]
        if name in [self.base_layer,]:         # 以shortcut层为基础来剪枝
            # del_filters = get_del_filter(net, "../models/image/")
            del_filters = np.loadtxt('del_filter.txt', dtype=np.float32)
            kernel_sum = np.sum(np.abs(weight), axis=(1,2,3))
            # print (kernel_sum)
            # del_num = self.ratio
            # kernel_axis = np.argsort(kernel_sum)
            # del_filters = kernel_axis[0:del_num]
            # print(del_filters)
        if del_filters is not None:
            # 裁剪通道数,output减小
            weight = np.delete(weight, del_filters, axis=0)
            bias = np.delete(bias, del_filters, axis=0)

        # print('\n')
        # print(name)
        # print(name + " filter nums need to delete is " + str(len(del_filters)))
        # print(name + " filter nums need to preserve is " + str(origin_channels - len(del_filters)))

        if del_kernels is not None:
            # 计算出要裁剪的kernel input减小
            weight = np.delete(weight, del_kernels, axis=1)

        print("{}层裁剪后的输出维度是{}".format(name,weight.shape))

        return weight, bias, del_filters, origin_channels

    # 暴力裁剪
    def prune_conv_rude(self, name, bottom=None , not_del_filters=False):
        if bottom is None:
            self.conv_data[name] = self._prune_rude(name, self._net.params[name])
        else:
            if not_del_filters is True:  # filters不需要裁剪(output), 但是kernels需要裁剪(input)
                self.conv_data[name] = self._prune_rude(name, self._net.params[name],
                                                        del_kernels=self.conv_data[self.base_layer][2],
                                                        del_filters=None)
            else:   # filters需要裁剪(output),但是kernels不需要裁剪(input)
                self.conv_data[name] = self._prune_rude(name, self._net.params[name],
                                                        del_kernels=None,
                                                        del_filters=self.conv_data[self.base_layer][2],)


    def fc_prune(self,conv_param, del_kernels):

        bias = conv_param[1]
        bias = bias.data
        f2c = fc2conv(self._net)
        weight = f2c.del_inputs(del_kernels)
        return weight, bias

    def prune_conv(self, name, bottom=None):
        if bottom is None:
            self.conv_data[name] = self._prune(name, self._net.params[name])
        else:
            self.conv_data[name] = self._prune(name, self._net.params[name], del_kernels=self.conv_data[bottom][2])

    def prune_concat(self, name, bottoms=None):
        if bottoms is not None:
            offsets = [0] + [self.conv_data[b][3] for b in bottoms]
            for i in range(1, len(offsets)):
                offsets[i] += offsets[i - 1]
            del_filters = [self.conv_data[b][2] + offsets[i] for i, b in enumerate(bottoms)]
            del_filters_new = np.concatenate(del_filters)
        else:
            del_filters_new = []
        if name[0:2] == 'fc':
            self.conv_data[name] = self.fc_prune(self._net.params[name], del_filters_new)
        else:
            self.conv_data[name] = self._prune_rude(name, self._net.params[name],
                                               del_kernels=del_filters_new, del_filters=None)

    def prune_sum(self, name, bottoms):
        del_filters = [self.conv_data[b][2] for b in bottoms]
        del_filter = np.union1d(del_filters[0], del_filters[1])
        print(del_filter)
        weight = []
        bias = []
        origin_channels = self.conv_data[bottoms[0]][3] - len(del_filter)
        for b in bottoms:
            if b[0:3] != 'res':
                self.conv_data[b] = self._prune(b, self._net.params[b], del_filters=del_filter)
        self.conv_data[name] = weight, bias, del_filter, origin_channels
        print("\n {} preserve num : {}".format(name, origin_channels))

    def save(self, new_model, output_weights):
        net2 = caffe.Net(new_model, caffe.TEST)
        for key in net2.params.keys():
            if key in self.conv_data:
                net2.params[key][0].data[...] = self.conv_data[key][0]
                net2.params[key][1].data[...] = self.conv_data[key][1]
            else:
                net2.params[key][0].data[...] = self._net.params[key][0].data
                net2.params[key][1].data[...] = self._net.params[key][1].data
        net2.save(output_weights)


root = "../my_model/"
prototxt = root + "TestModel_prune.prototxt"
caffemodel = root + "TestModel_prune.caffemodel"
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

pruner = Prune(net)

# block1,2

# pruner.prune_conv("conv1_1_1")
# pruner.prune_conv("conv1_2_1")
# pruner.prune_conv("conv1_2_2", "conv1_2_1")
# pruner.prune_conv("conv1_3_1")
# pruner.prune_conv("conv1_3_2", "conv1_3_1")
# pruner.prune_conv("conv1_3_3", "conv1_3_2")
#
# pruner.prune_concat("conv2_1", ("conv1_1_1", "conv1_2_2", "conv1_3_3"))
# pruner.prune_conv("conv2_2", "conv2_1")
# pruner.prune_conv("conv2_3", "conv2_2")
# pruner.prune_conv("conv2_4", "conv2_3")
# pruner.prune_conv("conv2_5", "conv2_4")
# pruner.prune_conv("conv2_6", "conv2_5")
# pruner.prune_conv("conv2_7", "conv2_6")
# pruner.prune_conv("conv2_8", "conv2_7")
#
# pruner.prune_concat("conv3_1_1", ("conv2_2", "conv2_4", "conv2_6", "conv2_8"))
# pruner.prune_concat("conv3_1_1b", ("conv2_2", "conv2_4", "conv2_6", "conv2_8"))


# block3 剪枝过程

pruner.init_layer('conv3_1_1')
pruner.init_layer('conv3_2_1')
pruner.init_layer('conv3_3_1')
pruner.init_layer('conv3_4_1')
pruner.init_layer('conv3_5_1')
pruner.init_layer('conv3_6_1')

pruner.prune_conv_rude('conv3_1_1b')
pruner.prune_conv_rude("conv3_1_2", "conv3_1_1", )
pruner.prune_conv_rude("conv3_2_1", "conv3_1_2", not_del_filters=True)
pruner.prune_conv_rude("conv3_2_2", "conv3_2_1", )
pruner.prune_conv_rude("conv3_3_1", "conv3_2_2",  not_del_filters=True)
pruner.prune_conv_rude("conv3_3_2", "conv3_3_1", )
pruner.prune_conv_rude("conv3_4_1", "conv3_3_2",  not_del_filters=True)
pruner.prune_conv_rude("conv3_4_2", "conv3_4_1", )
pruner.prune_conv_rude("conv3_5_1", "conv3_4_2",  not_del_filters=True)
pruner.prune_conv_rude("conv3_5_2", "conv3_5_1", )
pruner.prune_conv_rude("conv3_6_1", "conv3_5_2",  not_del_filters=True)
pruner.prune_conv_rude("conv3_6_2", "conv3_6_1", )

pruner.prune_concat("conv4_1_1", ("conv3_2_2", "conv3_4_2", "conv3_6_2", ))
pruner.prune_concat("conv4_1_1b", ("conv3_2_2", "conv3_4_2", "conv3_6_2",))

# # block4 剪枝过程
#
# pruner.init_layer('conv4_1_1')
# pruner.init_layer('conv4_2_1')
# pruner.init_layer('conv4_3_1')
# pruner.init_layer('conv4_4_1')
# pruner.init_layer('conv4_5_1')
# pruner.init_layer('conv4_6_1')
#
# pruner.prune_conv_rude('conv4_1_1b')
# pruner.prune_conv_rude("conv4_1_2", "conv4_1_1", )
# pruner.prune_conv_rude("conv4_2_1", "conv4_1_2", not_del_filters=True)
# pruner.prune_conv_rude("conv4_2_2", "conv4_2_1", )
# pruner.prune_conv_rude("conv4_3_1", "conv4_2_2",  not_del_filters=True)
# pruner.prune_conv_rude("conv4_3_2", "conv4_3_1", )
# pruner.prune_conv_rude("conv4_4_1", "conv4_3_2",  not_del_filters=True)
# pruner.prune_conv_rude("conv4_4_2", "conv4_4_1", )
# pruner.prune_conv_rude("conv4_5_1", "conv4_4_2",  not_del_filters=True)
# pruner.prune_conv_rude("conv4_5_2", "conv4_5_1", )
# pruner.prune_conv_rude("conv4_6_1", "conv4_5_2",  not_del_filters=True)
# pruner.prune_conv_rude("conv4_6_2", "conv4_6_1", )
#
# pruner.prune_concat("conv5_1_1", ("conv4_2_2", "conv4_4_2", "conv4_6_2", ))
# pruner.prune_concat("conv5_1_1b", ("conv4_2_2", "conv4_4_2", "conv4_6_2",))

#
# # block5 剪枝过程
# pruner.init_layer('conv5_1_1')
# pruner.init_layer('conv5_2_1')
# pruner.init_layer('conv5_3_1')
# pruner.init_layer('conv5_4_1')
# pruner.init_layer('conv5_5_1')
# pruner.init_layer('conv5_6_1')
#
# pruner.prune_conv_rude('conv5_1_1b')
# pruner.prune_conv_rude("conv5_1_2", "conv5_1_1", )
# pruner.prune_conv_rude("conv5_2_1", "conv5_1_2", not_del_filters=True)
# pruner.prune_conv_rude("conv5_2_2", "conv5_2_1", )
# pruner.prune_conv_rude("conv5_3_1", "conv5_2_2",  not_del_filters=True)
# pruner.prune_conv_rude("conv5_3_2", "conv5_3_1", )
# pruner.prune_conv_rude("conv5_4_1", "conv5_3_2",  not_del_filters=True)
# pruner.prune_conv_rude("conv5_4_2", "conv5_4_1", )
# pruner.prune_conv_rude("conv5_5_1", "conv5_4_2",  not_del_filters=True)
# pruner.prune_conv_rude("conv5_5_2", "conv5_5_1", )
# pruner.prune_conv_rude("conv5_6_1", "conv5_5_2",  not_del_filters=True)
# pruner.prune_conv_rude("conv5_6_2", "conv5_6_1", )
# pruner.prune_concat('fc_svd_v', ('conv5_2_2', 'conv5_4_2', 'conv5_6_2'))

pro_new = root + "TestModel_prune_1.prototxt"
pruner.save(pro_new, root + 'TestModel_prune_1.caffemodel')
