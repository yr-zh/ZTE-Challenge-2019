# ZTE_Challenge_2019



**ZTE_Challenge_2019 Alpha group , the titile is Model Compression and Acceleration**

Given a trained caffemodel, compress and accelerate the caffemodel without retraining.

#### Preliminary

**Here are  some ideas:**

1. SVD (full connect layer)
2. Fuse the conv and bn layers
3. Prune
4. Quantize
5. Decomposed convolution kernel(Seperate a kxk-convolution into deepwise or kx1, 1xk)

code in the folder script 

#### Finals

main method is int8 quantize,there are two tools we can use:

**[ncnn](https://github.com/Tencent/ncnn)**

**[tensorrt](https://developer.nvidia.com/tensorrt)**

prototxt in the models folder

