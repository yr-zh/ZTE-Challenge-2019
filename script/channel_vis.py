# coding=utf-8
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import cv2
import matplotlib.pyplot as plt
import numpy as np


def image_test(net, img_dir, layer1='conv3_1_1b', layer2='res_conv3_1_2'):

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})    # transformer读进来是RGB格式和0~1(float)
    transformer.set_transpose('data', (2, 0, 1))                                  # 通道优先,从(227,227,3)转换成(3,227,227)
    # transformer.set_raw_scale('data', 255.0)                                      # 图像到0-255
    transformer.set_channel_swap('data', (2, 1, 0))                               # 通道转换成BGR

    weight = net.blobs[layer2].data
    channel = weight.shape[1]
    files = os.listdir(img_dir)
    img_num = len(files)
    print("{} has {} channels".format(layer1, channel))
    data = np.zeros((img_num, channel))

    for i, f in enumerate(files):
        file_name = img_dir + f
        image = caffe.io.load_image(file_name)
        net.blobs['data'].data[...] = transformer.preprocess('data', (image*255 - 127.5) * 0.0078125) # 图像归一化到-1~1
        net.blobs['data'].data[...] = transformer.preprocess('data', image)

        net.forward()

        for j in range(channel):
            # data = net.blobs[layer_show].data[0, j]
            data[i][j] = np.sum(np.abs(net.blobs[layer2].data[0, j]))
        avg = np.mean(data, axis=0)

    del_filter = []
    # for i, x in enumerate(avg):
    #     if x < 0.1:
    #         del_filter.append(i)
    # print(channel - len(del_filter))
    # # del_filter = [x.index() for x in avg if x < 1]
    # del_filter = np.array(del_filter)
    return del_filter, avg, channel


def image_vis(net, img_dir, layer1='conv5_6_2', layer2='feature5'):

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})    # transformer读进来是RGB格式和0~1(float)
    transformer.set_transpose('data', (2, 0, 1))                                  # 通道优先,从(227,227,3)转换成(3,227,227)
    transformer.set_raw_scale('data', 255.0)                                      # 图像到0-255
    transformer.set_channel_swap('data', (2, 1, 0))                               # 通道转换成BGR

    weight = net.blobs[layer2].data
    channel = weight.shape[1]
    files = os.listdir(img_dir)
    img_num = len(files)
    print("{} has {} channels".format(layer1, channel))
    data = np.zeros((img_num, channel))

    for i, f in enumerate(files):
        file_name = img_dir + f
        image = caffe.io.load_image(file_name)
        # net.blobs['data'].data[...] = transformer.preprocess('data', (image * 255 - 127.5) * 0.0078125) # 图像归一化到-1~1
        net.blobs['data'].data[...] = transformer.preprocess('data', image)
        predict = net.forward()
        filt_min, filt_max = net.blobs[layer2].data.min(), net.blobs[layer2].data.max()

        for batch in range(10):
        # 通道数
            channels = range(batch*25 , (batch+1)*25)
            width = 5
            height = 5

            # 显示图片
            for i in range(25):
                data = net.blobs[layer2].data[0, 25*batch+i]
                print("channel {} sum {}:".format(i, np.sum(data)))
                plt.subplot(width, height, i+1)
                plt.title("filter #{} output".format(i))
                plt.imshow(data, vmin=filt_min, vmax=filt_max)
                plt.axis('off')

                # plt.tight_layout()
            plt.show()



def get_del_filter(net, image_dir):
    # layer1获得图片尺寸  layer2是特征图来筛选卷积核
    # layers1 = ["conv5_1_2", "conv5_2_2", "conv5_3_2", "conv5_4_2", "conv5_5_2", "conv5_6_2", ]
    # layers2 = ["res_conv5_1_2", "res_conv5_2_2", "res_conv5_3_2", "res_conv5_4_2", "res_conv5_5_2", "res_conv5_6_2",]
    # layers1 = ["conv5_1_2","conv5_2_2","conv5_4_2","conv5_6_2"]
    # layers2 = ["res_conv5_1_2","res_conv5_2_2","res_conv5_4_2","res_conv5_6_2"]
    layers1 = ["conv3_1_1b",]
    layers2 = ["res_conv3_1_2",]
    # print(net.blobs.keys)
    del_filters = []
    average = []
    for i in range(len(layers1)):
        layer1 = layers1[i]
        layer2 = layers2[i]
        del_filter, avg, channel = image_test(net, image_dir, layer1, layer2)
        # del_filters = np.union1d(del_filter, del_filters)
        average.append(avg)
    average = np.array(average)
    a = np.mean(average, axis=0)
    del_filter = []
    # del_filter = np.argsort(a)[0:56]
    for i, x in enumerate(a):
        if x < 1e-2:
            del_filter.append(i)
    print(channel - len(del_filter))
    del_filter = np.array(del_filter)
    return del_filter
    # print(weight.shape)
    # show_feature(weight.transpose(0, 2, 3, 1))


if __name__ == '__main__':

    prototxt = '../my_model/TestModel.prototxt'
    caffemodel = '../my_model/TestModel.caffemodel'
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    image_dir = "../models/image2/"
    del_filter = get_del_filter(net, image_dir)
    np.savetxt("del_filter.txt", del_filter)
    # b = np.loadtxt('del_filter.txt', dtype=np.float32)
    # print(b)
    # image_vis(net, image_dir)
