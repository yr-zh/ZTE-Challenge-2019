import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
import pycuda.driver as cuda
import numpy as np

with open("zte.v7.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    while True:
        pass

