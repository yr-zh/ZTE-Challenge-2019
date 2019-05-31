# coding:utf-8

"""
计算两个模型之间的余弦距离
"""

import numpy as np
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe


def image_test(net, img_dir):
    print(net.blobs['data'].data.shape)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})    # transformer读进来是RGB格式和0~1(float)
    transformer.set_transpose('data', (2, 0, 1))                                  # 通道优先,从(227,227,3)转换成(3,227,227)
    # transformer.set_raw_scale('data', 255.0)                                      # 图像到0-255
    transformer.set_channel_swap('data', (2, 1, 0))                               # 通道转换成BGR
    x_list = []

    for f in os.listdir(img_dir):
        file_name = img_dir + f
        image = caffe.io.load_image(file_name)
        net.blobs['data'].data[...] = transformer.preprocess('data', (image * 255 - 127.5) * 0.0078125) # 图像归一化到-1~1
        # net.blobs['data'].data[...] = transformer.preprocess('data', image)
        predict = net.forward()
        x = predict['fc5_'][0]
        x_list.append(x)
    return x_list


def cal_dist(x_list, y_list):
    dists = []
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        dist = np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
        dists.append(dist)

    print(np.mean(dists))


if __name__ == "__main__":

    prototxt = '../my_model/TestModel.prototxt'
    caffemodel = '../my_model/TestModel.caffemodel'
    my_prototxt = '../my_model/TestModel_prune.prototxt'
    my_model = '../my_model/TestModel_prune.caffemodel'
    img_dir = '../models/image_test/'
    img_file = img_dir + '000003.jpg'

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    x_list = image_test(net, img_dir)
    net1 = caffe.Net(my_prototxt, my_model, caffe.TEST)
    y_list = image_test(net1, img_dir)
    cal_dist(x_list, y_list)
