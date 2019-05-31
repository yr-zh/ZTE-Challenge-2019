import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np
import sys
sys.path.append('..')
# caffe.set_mode_cpu()
import time
SEED = 0
np.random.seed(SEED)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
BATCH_SIZE = 50
TEST_NUM = 1000

class ModelData(object):

    OUTPUT_NAME = "fc5"
    # The original model is a float32 one.
    DTYPE = trt.float32
    INPUT_SHAPE = (3, 128, 128)
    MEAN = [127.5, 127.5, 127.5]
    
def cpu_cos_dis(pre_ft,new_ft):
    sum2_pre=np.sqrt(np.square(pre_ft).sum())
    sum2_new=np.sqrt(np.square(new_ft).sum())
    # print sum2_pre,sum2_new
    return np.sum(pre_ft*new_ft)/(sum2_new*sum2_pre)#+1e-5)


def GetTPR(features1, features2): 
    posSims=[];
    negSims=[];

    # print(len(features1))
    # print(len(features2))
    for i in range(len(features1)):     
        for j in range(len(features2)):  
#             print(features1[i][:10])
#             print(features2[j][:10])
#             exit()
            sim = cpu_cos_dis(np.array(features1[i]),np.array(features2[j]))
            # print()
            # print(sim)
            if(i == j):           
                posSims.append(sim)            
            else:                
                negSims.append(sim)             
    
    print(len(negSims))

    negSims.sort(reverse=True)
    print(negSims[:20])
    print(negSims[-20:-1])
    posSims.sort(reverse=False)
    print(posSims[-20:-1])
    print(posSims[:20])
    # exit()
    
    print("negSims/10000: ")
    print(int(len(negSims)/10000))
    
    threshold_00001=negSims[int(len(negSims)/10000)]     

    posErrorNum_00001=0         

    for i in range(len(posSims)):          
        if posSims[i]<threshold_00001:         
               posErrorNum_00001+=1;         
        else:
            break   
            
    print("threshold:", str(threshold_00001))
    print("posError:",str(posErrorNum_00001))
    
    TPR_00001=1.0*(len(posSims)-posErrorNum_00001)/(1.0*len(posSims)); 
    print("TPR: ", TPR_00001)
    # cout << "TPR is " << TPR_00001 << " @FPR=0.0001."<<endl; 
    return TPR_00001


# read val.txt and get test data and label
def get_test_data(data_path, labels_path):
    f = open(labels_path)
    lines = f.readlines()[0:TEST_NUM]
    datas = []
    labels = []

    # read the top n lines , get data path and labels
    for line in lines:
        data = data_path + line.split()[0]
        label = data_path + line.split()[1]
        datas.append(data)
        labels.append(label)

    return datas, labels


def do_inference(context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()


def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream


def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        # return np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()

        im = cv2.imread(image).astype(np.float32)
        im = cv2.resize(im, (w, h))
#         print( im.shape)
#         im = np.asarray(im)
#         im = im.transpose((1,2,0)) # CHW -> HWC
        im -= 127.5 # Broadcast subtract
        im = im * 0.007815
#         im = im.transpose((2,0,1)) # HWC -> CHW
        
        return np.asarray(im).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()
    
    # Normalize the image and copy to pagelocked memory.
    # pagelocked_buffer = np.copy(normalize_image(test_image))
    np.copyto(pagelocked_buffer, normalize_image(test_image))
    return test_image


def main():
#     while True:
#         pass
#     data_test_path = "/home/ubuntu/MyFiles/ZTE/FACE-ALL-5-POINTS_CROP/"
    data_test_path = "/home/ubuntu/MyFiles/ZTE/1000pairs/"
    data_txt = "/home/ubuntu/MyFiles/ZTE/test_2000_images_list.txt"

    with open("zte.v7.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
#         while True:
#             pass
        batch_size = BATCH_SIZE

        # Allocate buffers and create a CUDA stream.
        features1 = []
        features2 = []
        with engine.create_execution_context() as context:

            data_test1, data_test2 = get_test_data(data_test_path, data_txt)
            for i in range(TEST_NUM):
                h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
                test_case = load_normalized_test_case(data_test1[i], h_input)
                do_inference(context, h_input, d_input, h_output, d_output, stream)
                features1.append(h_output)
#             print(data_test1[0:10])

            for i in range(TEST_NUM):
                h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
                test_case = load_normalized_test_case(data_test2[i], h_input)
                do_inference(context, h_input, d_input, h_output, d_output, stream)
                features2.append(h_output)
#             print(data_test2[0:10])
#         print(features1[0][0:10])
#         print(features2[0][0:10])
        
        GetTPR(features1, features2)


if __name__ == '__main__':
    main()

