# coding=utf-8
import os
import numpy as np
os.environ['GLOG_minloglevel'] = '2'
import caffe

orig_net = caffe.Net('../my_model/TestModel.prototxt', '../my_model/TestModel.caffemodel', caffe.TEST)
resize_net = caffe.Net('../my_model/TestModel_resize.prototxt', caffe.TEST)

print resize_net.params.keys()
for key in resize_net.params.keys():

    if key == 'fc_svd_v':
        weight = orig_net.params[key][0].data
        bias = orig_net.params[key][1].data
        channel = weight[1].shape[0] / 3
        w = weight[:, 0: channel]
        b = bias[0: channel]
        resize_net.params[key][0].data[...] = w
        resize_net.params[key][1].data[...] = b

    elif key in ['conv5_1_1', 'conv5_1_1b']:

        weight = orig_net.params[key][0].data
        bias = orig_net.params[key][1].data
        del_filters = np.arange(384)[256:384]
        w = np.delete(weight, del_filters, axis=1)
        b = np.delete(bias, del_filters, axis=0)
        resize_net.params[key][0].data[...] = w
        # resize_net.params[key][1].data[...] = b

    else:
        # other layers
        resize_net.params[key][0].data[...] = orig_net.params[key][0].data
        resize_net.params[key][1].data[...] = orig_net.params[key][1].data

# 保存模型
resize_net.save('../my_model/TestModel_resize.caffemodel')          # save new caffemodel
