#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.

import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np
import cv2

# For reading size information from batches
import struct

class ImagenetEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batch_data_dir, cache_file, stream):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        # Get a list of all the batch files in the batch folder.
        # self.batch_files = [os.path.join(batch_data_dir, f) for f in os.listdir(batch_data_dir)]
        self.channel = 3
        self.width = 128
        self.height = 128
        self.batchsize = 512

        # Find out the shape of a batch and then allocate a device buffer of that size.
        # NCHW dimensions
        self.shape = (self.batchsize, self.channel, self.height, self.width)

        # define a stream to batch data
        self.stream = stream

        # Each element of the calibration data is a float32.
        self.device_input = cuda.mem_alloc(trt.volume(self.shape) * trt.float32.itemsize)
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names, stream):
#         data = next(self.batches)
        data = self.stream.next_batch()
        if not data.size:   
            return None
        cuda.memcpy_htod(self.device_input, data)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class ImageBatchStream():
  def __init__(self, batch_size, calibration_files, preprocessor):
    self.batch_size = batch_size
    self.channel = 3
    self.width = 128
    self.height = 128
    self.max_batches = (len(calibration_files) // batch_size) + \
                    (1 if (len(calibration_files) % batch_size) \
                        else 0)
    self.files = calibration_files
    self.calibration_data = np.zeros((batch_size, self.channel, self.height, self.width), dtype=np.float32)
    self.batch = 0
    self.preprocessor = preprocessor

  @staticmethod
  def read_image_chw(path):
    w = 128
    h = 128

    # read image as BGR
    im = cv2.imread(path).astype(np.float32)
    # img shape(224, 224, 3)
    im = cv2.resize(im, (w, h))

    # (224,224,3) -> (3,244,244)
    im = im.transpose((2,0,1))
    # return （3，128,128）C，H，W
    return im
         
  def reset(self):
    self.batch = 0
     
  def next_batch(self):
    if self.batch < self.max_batches:
      imgs = []
      files_for_batch = self.files[self.batch_size * self.batch : \
                        self.batch_size * (self.batch + 1)]
      for i, f in enumerate(files_for_batch):
        num = self.batch * self.batch_size + i
        print(str(num), " [ImageBatchStream] Processing ", f)
        img = ImageBatchStream.read_image_chw(f)
        img = self.preprocessor(img)
        imgs.append(img)
      for i in range(len(imgs)):
        self.calibration_data[i] = imgs[i]
      self.batch += 1
      return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
    else:
      return np.array([])