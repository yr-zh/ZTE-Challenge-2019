# coding=utf-8
"""
全连接层的svd分解
"""
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np


def SVD(k):
    orig_net = caffe.Net('../model/TestModel.prototxt', '../model/TestModel.caffemodel', caffe.TEST)
    svd_net = caffe.Net('../models/svd/TestModel.prototxt',  caffe.TEST)

    # 提取orig_net weights
    W = orig_net.params['fc5_'][0].data                      # shape = (1536, 512)
    b = orig_net.params['fc5_'][1].data
    print("W.shape;{}".format(W.shape))

    # k为需要取的奇异值数量，大小应该和prototxt中fc_svd_v的num_output一致
    # k = 160

    # SVD 对权重W进行奇异值分解
    U, s, V = np.linalg.svd(W, full_matrices=False)
    print("U.shape:{}".format(U.shape),
          "s.shape:{}".format(s.shape),
          "V.shape:{}".format(V.shape))
    # print(s)

    print(s[:k].sum()/s.sum())

    U_ = np.matrix(U[:, :k])                                       # shape = (10, k)
    s_ = np.matrix(np.diag(s[:k]))                                 # shape = (k, )
    V_ = np.matrix(V[:k, :])                                       # shape = (k, 64)
    # L = np.dot(np.diag(s_), V_)                                  # np.diag(s_): (k, ) to (k, k)     L_shape = (k, 64)
    L = s_ * V_
    print("U.shape:{}".format(U_.shape),
          "s.shape:{}".format(s_.shape),
          "V.shape:{}".format(V_.shape),
          "L.shape:{}".format(L.shape))

    # 更新模型参数
    for key in svd_net.params.keys():
        if key == 'fc_svd_v':
            svd_net.params[key][0].data[...] = L
        elif key =='fc5_':
            svd_net.params[key][0].data[...] = U_
            svd_net.params[key][1].data[...] = b
        else:
            svd_net.params[key][0].data[...] = orig_net.params[key][0].data
            svd_net.params[key][1].data[...] = orig_net.params[key][1].data

    # 保存模型
    svd_net.save('../models/svd/TestModel.caffemodel')          # save new caffemodel


if __name__ == "__main__":
    k = 160
    SVD(k)