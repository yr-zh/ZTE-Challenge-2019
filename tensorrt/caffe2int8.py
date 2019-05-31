import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import random

# For our custom calibrator
import calibrator

# For ../common.py
import sys, os
import common
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelData(object):
    DEPLOY_PATH = "/home/ubuntu/MyFiles/ZTE/R18/resnet18_deploy_inference.prototxt"
    MODEL_PATH = "/home/ubuntu/MyFiles/ZTE/R18/Resnet18_train_iter_270000_inference.caffemodel"
    OUTPUT_NAME = "fc5"
    # The original model is a float32 one.
    DTYPE = trt.float32
    MEAN = [127.5, 127.5, 127.5]
#     MEAN = [100.23, 115.42, 148.79]
#     STD = [31.8, 29.9, 33.2]
#     STD = [0.0315, 0.0334, 0.0301]
    STD = [0.0078, 0.0078, 0.0078]

# This function builds an engine from a Caffe model.
def build_int8_engine(deploy_file, model_file, calib):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = calib.get_batch_size()
        builder.max_workspace_size = common.GiB(1)
        builder.int8_mode = True
        builder.int8_calibrator = calib
        # Parse Caffe model
        model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)
        network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        # Build engine and do int8 calibration.
        return builder.build_cuda_engine(network)

# Loads a random batch from the supplied calibrator.
def load_random_batch(calib):
    # Load a random batch.
    batch = random.choice(calib.batch_files)
    _, data, labels = calib.read_batch_file(batch)
    data = np.fromstring(data, dtype=np.float32)
    # labels = np.fromstring(labels, dtype=np.float32)
    return data, labels


# Note that we don't expect the accuracy to be 100%, but it should
# be close to the fp32 model (which is in the 98-99% range).
def validate_output(output, labels):
    preds = np.argmax(output, axis=1)
    print("Expected Predictons:\n" + str(labels))
    print("Actual Predictons:\n" + str(preds))
    check = np.equal(preds, labels)
    accuracy = np.sum(check) / float(check.size) * 100
    print("Accuracy: " + str(accuracy) + "%")
    # If a prediction was incorrect print out an array of booleans indicating accuracy.
    if accuracy != 100:
        print("One or more predictions was incorrect:\n" + str(check))


def find_sample_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(data_path + " does not exist. Please check the correct data path .")
    # list all files in filedir
    files = os.listdir(path)
    file_paths = []
    # get path for each file, return a list
    for file in files:
        file_path = path + file
        file_paths.append(file_path)
    return file_paths

def sub_mean_chw(data):
    # (C, W, H) BGR
#     print(data.shape)
#     exit()
    mean = np.array(ModelData.MEAN)
#     data[0] = (data[0] - mean[0]) 
#     data[1] = (data[1] - mean[1]) 
#     data[2] = (data[2] - mean[2]) 

    std = np.array(ModelData.STD)
    data[0] = (data[0] - mean[0]) * std[0]
    data[1] = (data[1] - mean[1]) * std[1]
    data[2] = (data[2] - mean[2]) * std[2]
#     print(data)
#     print(data.shape)
#     exit()
#     data = data.transpose((2,0,1)) # 
    return data
             
def main():

    # label_path = "label/val.txt"
    data_path = "/home/ubuntu/MyFiles/ZTE/sample_6k/"
    deploy_file = "/home/ubuntu/MyFiles/ZTE/R18/resnet18_deploy_inference.prototxt"
    model_file = "/home/ubuntu/MyFiles/ZTE/R18/Resnet18_train_iter_270000_inference.caffemodel"

    data_files = find_sample_data(data_path)
    calibration_files = data_files
   
    # Process batch_size images at a time for calibration
    batch_size = 16

    # This batch size can be different from MaxBatchSize (1 in this example)
    batchstream = calibrator.ImageBatchStream(batch_size, calibration_files, sub_mean_chw)
    print(123)
    # Now we create a calibrator and give it the location of our calibration data.
    # We also allow it to cache calibration data for faster engine building.
    calibration_cache = "calibration.cache"
    calib = calibrator.ImagenetEntropyCalibrator(data_files, calibration_cache, batchstream)

    # We will use the calibrator batch size across the board.
    # This is not a requirement, but in this case it is convenient.
    # batch_size = calib.get_batch_size()

    # with build_int8_engine(deploy_file, model_file, calib) as engine, engine.create_execution_context() as context:
    with build_int8_engine(deploy_file, model_file, calib) as engine:
        # save the model
        with open("zte.v7.engine", "wb") as f:
            f.write(engine.serialize())
        
        # Allocate engine buffers.
        # inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        # # Do inference for the whole batch. We have to specify batch size here, as the common.do_inference uses a default
        # inputs[0].host, labels = load_random_batch(calib)
        # [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=batch_size)
        # output = output.reshape(batch_size, 10)
        # get_labels(labels_path)
        # validate_output(output, labels)

if __name__ == "__main__":
    main()
