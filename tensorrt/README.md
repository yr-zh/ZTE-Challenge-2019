#### 量化工具：Tensorrt 5.1.2.2

#### 代码

- caffe2int8.py ：把prototxt和caffemodel模型转为tensorrt的.engine文件，使用不同数量的数据集进行校准，效果相差不大，并且不是校准数据集数量越多越好。6k张校准图片时TPR为0.971,模型大小94.8M
- zte_accuracy ：测试量化效果文件。
- calibrator.py ：包含了ImagenetEntropyCalibrator和ImageBatchStream两个类。ImagenetEntropyCalibrator用于校准，ImageBatchStream用于读取batch数据。
- read_tensorrt.py ：读取tensorrt模型文件

同时也花了一半时间尝试了ncnn，但是效果始终不好，代码在/ncnn中（classify.cpp)