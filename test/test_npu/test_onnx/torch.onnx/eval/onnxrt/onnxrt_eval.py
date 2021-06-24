# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf8

import argparse
import os
import random
import shutil
import time
import warnings
#from __future__ import division

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from apex import amp
import numpy as np
import onnx
import onnxruntime as ort
from export.onnx_parser import *
from export.cp_parser import *


from onnx import numpy_helper
import urllib.request
import json
import time

# display images in notebook
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

#%matplotlib inline


#G_ONNX_FILEPATH = './densenet121.onnx'



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("[gpu id:",os.environ['KERNEL_NAME_ID'],"]",'\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_labels(path):
    with open(path) as f:
        # content = f.read()
        # if content.startswith(u'\ufeff'):
        #     content = content.encode('utf8')[3:].decode('utf8')
        data = json.load(f)

    return np.asarray(data)

def preprocess(input_data,dimx=1):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        
    #add batch channel
    print(norm_img_data.shape)
    #norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    norm_img_data.resize(dimx, 3, 224, 224)
    norm_img_data = norm_img_data.astype('float32')
    return norm_img_data

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()



test_data_dir = 'resnet50v2/test_data_set'
test_data_num = 3
import glob
import os

def load_IOPuts():
    # Load inputs
    inputs = []
    for i in range(test_data_num):
        input_file = os.path.join(test_data_dir + '_{}'.format(i), 'input_0.pb')
        tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            tensor.ParseFromString(f.read())
            inputs.append(numpy_helper.to_array(tensor))

    print('Loaded {} inputs successfully.'.format(test_data_num))
            
    # Load reference outputs

    ref_outputs = []
    for i in range(test_data_num):
        output_file = os.path.join(test_data_dir + '_{}'.format(i), 'output_0.pb')
        tensor = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            tensor.ParseFromString(f.read())    
            ref_outputs.append(numpy_helper.to_array(tensor))
            
    print('Loaded {} reference outputs successfully.'.format(test_data_num))
    return inputs,ref_outputs

def onnxrt_run_resnet50(onnx_path,inputs,ref_outputs):

    sess = ort.InferenceSession(onnx_path, None)
    input_name = sess.get_inputs()[0].name  

    print('Input Name:', input_name)
    # img_height = 224
    # img_width = 224
    #sess.run(None, {'input_1': np.random.rand(2, 3, img_height, img_width).astype('float32')})
    # rtn = sess.run(None, {'actual_input_1': np.random.rand(10, 3, img_height, img_width).astype('float32')})
    # return rtn

    #%%time
    print(onnx_path)
    print(type(inputs[0]),np.size(inputs[0]))
    outputs = [sess.run([], {input_name: inputs[i]})[0] for i in range(test_data_num)]

    print('Predicted {} results.'.format(len(outputs)))

    # Compare the results with reference outputs up to 4 decimal places
    for ref_o, o in zip(ref_outputs, outputs):
        np.testing.assert_almost_equal(ref_o, o, 4)
        
    print('ONNX Runtime outputs are similar to reference outputs!')
    return outputs

def onnxrt_runeval_resnet50(cp_path,onnx_path):

    # Load the ONNX model
    #onnx_model = onnx.load(onnx_path)
    
    #cp_model = cp_load(torchvision.models.resnet50(pretrained=False),cp_path)
    print(onnx_path)
    inputs,ref_outputs = load_IOPuts()
    onnxrt_result = onnxrt_run(onnx_path,inputs,ref_outputs)

    # compare ONNX Runtime and PyTorch results
    #np.testing.assert_allclose(to_numpy(torch_out), onnxrt_result[0], rtol=1e-03, atol=1e-05)
    #print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def onnxrt_runeval_resnet50_pic(cp_path,onnx_path,dimx=1):
    session = ort.InferenceSession(onnx_path, None)
    labels = load_labels('resnet50v2/imagenet-simple-labels.json')
    image = Image.open('resnet50v2/dog224.224.jpg')
    #image = Image.open('resnet50v2/dog.jpg')
    #image = Image.open('resnet50v2/dog.png')
    # image = Image.open('images/plane.jpg')

    input_name = session.get_inputs()[0].name  
    output_name = session.get_outputs()[0].name
    print("input_name: ", input_name,  "output_name: ", output_name)
    print("Image size: ", image.size)
    plt.axis('off')
    display_image = plt.imshow(image)
    image_data = np.array(image).transpose(2, 0, 1)
    input_data = preprocess(image_data,dimx=dimx)

    start = time.time()
    raw_result = session.run([output_name], {input_name: input_data})
    end = time.time()
    res = postprocess(raw_result)

    inference_time = np.round((end - start) * 1000, 2)
    idx = np.argmax(res)

    print('========================================')
    print('Final top prediction is: ' + labels[idx])
    print('========================================')

    print('========================================')
    print('Inference time: ' + str(inference_time) + " ms")
    print('========================================')

    sort_idx = np.flip(np.squeeze(np.argsort(res)))
    print('============ Top 5 labels are: ============================')
    print(labels[sort_idx[:5]])
    print('===========================================================')

    plt.axis('off')
    display_image = plt.imshow(image)
    
def resnet50_eval(cp_path,dimx=1):
    resnet50 = torchvision.models.resnet50(pretrained=False)
    model = cp_load(resnet50,cp_path)
    model.eval()
    image = Image.open('resnet50v2/dog224.224.jpg')
    image_data = np.array(image).transpose(2, 0, 1)
    input_data = preprocess(image_data, dimx=dimx)
    image_tensor = torch.from_numpy(input_data)
    out = model(image_tensor)
    _, indices = torch.sort(out, descending=True)   
    return indices[0]

def onnxrt_resnet50_eval(onnx_path,dimx=1):
    #print('-------------cp_path-------------',cp_path)
    session = ort.InferenceSession(onnx_path, None)
    image = Image.open('resnet50v2/dog224.224.jpg')
    #image = Image.open('resnet50v2/dog.jpg')
    #image = Image.open('resnet50v2/dog.png')
    # image = Image.open('images/plane.jpg')

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("input_name: ", input_name,  "output_name: ", output_name)  
    print("Image size: ", image.size)
    plt.axis('off')
    image_data = np.array(image).transpose(2, 0, 1)
    input_data = preprocess(image_data,dimx=dimx)

    start = time.time()
    raw_result = session.run([output_name], {input_name: input_data})
    end = time.time()
    res = postprocess(raw_result)
    inference_time = np.round((end - start) * 1000, 2)
    print('========================================')
    print('Inference time: ' + str(inference_time) + " ms")
    print('========================================')
    return res

def onnxrt_runeval_resnet50_cmp(cp_path,onnx_path,dimx=1):
    labels = load_labels('resnet50v2/imagenet-simple-labels.json')
    res_onnx = onnxrt_resnet50_eval(onnx_path,dimx=dimx)
    idx_onnx = np.argmax(res_onnx)
    print('========================================')
    print('=============onnx inference=============')
    print('Final top prediction is: ' + labels[idx_onnx])
    print('========================================')

    sort_idx_onnx = np.flip(np.squeeze(np.argsort(res_onnx)))
    print('============ Top 5 labels are: ============================')
    print(labels[sort_idx_onnx[:5]])
    print('===========================================================')

    res_cp = resnet50_eval(cp_path, dimx=dimx)
    print('========================================')
    print('=============cp inference=============')
    print('Final top prediction is: ' + labels[res_cp[0]])
    print('========================================')
    print('============ Top 5 labels are: ============================')
    print(labels[res_cp[:5]])
    print('===========================================================')


def onnxrt_densnet121_eval(onnx_path,inputdata,dimx=1):
    session = ort.InferenceSession(onnx_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    #input_data = torch.randn(10, 3, 224, 224)
    raw_result = session.run([output_name], {input_name: inputdata.numpy()})
    print(raw_result[0])
    return raw_result[0]


def densenet121_eval(densenet_model, cp_path,inputdata, ispth=False):
    model = cp_load(densenet_model, cp_path, ispth=ispth)
    model.eval()
    #image_tensor = torch.randn(10, 3, 224, 224)
    out = model(inputdata)
    print(out.detach().numpy())
    return out


def efficientnet_eval(efficientnet_model, cp_path,inputdata):
    model = cp_load(efficientnet_model, cp_path)
    model.eval()
    #image_tensor = torch.randn(10, 3, 224, 224)
    out = model(inputdata)
    print(out)
    return out

def onnxrt_efficientnet_eval(onnx_path,inputdata,dimx=1):
    session = ort.InferenceSession(onnx_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    #input_data = torch.randn(10, 3, 224, 224)
    raw_result = session.run([output_name], {input_name: inputdata.numpy()})
    print(raw_result)
    return raw_result

def shufflenet_eval(shufflenet_model, cp_path, inputdata, ispth=False):
    model = cp_load(shufflenet_model, cp_path,ispth=ispth)
    model.eval()
    #image_tensor = torch.randn(10, 3, 224, 224)
    out = model(inputdata)
    print(out)
    return out

def onnxrt_shufflenet_eval(onnx_path,inputdata):
    session = ort.InferenceSession(onnx_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    #input_data = torch.randn(10, 3, 224, 224)
    raw_result = session.run([output_name], {input_name: inputdata.numpy()})
    print(raw_result)
    return raw_result

def deepmar_resnet50_eval(deepmar_model, cp_path, inputdata, ispth=False):
    model = cp_load(deepmar_model, cp_path,ispth=ispth)
    model.eval()
    #image_tensor = torch.randn(10, 3, 224, 224)
    out = model(inputdata)
    print(out)
    return out

def onnxrt_deepmar_resnet50_eval(onnx_path, inputdata):
    session = ort.InferenceSession(onnx_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    #input_data = torch.randn(10, 3, 224, 224)
    raw_result = session.run([output_name], {input_name: inputdata.numpy()})
    print(raw_result)
    return raw_result


# RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
def mobilenetv2_eval(mobilenetv2_model, cp_path, inputdata, ispth=False):
    model = cp_load(mobilenetv2_model, cp_path,ispth=ispth)
    model.eval()
    #image_tensor = torch.randn(10, 3, 224, 224).to('cuda:0')
    out = model(inputdata)
    print(out)
    return out

def onnxrt_mobilenetv2_eval(onnx_path, inputdata):
    session = ort.InferenceSession(onnx_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    #input_data = torch.randn(10, 3, 224, 224)
    raw_result = session.run([output_name], {input_name: inputdata.numpy()})
    print(raw_result)
    return raw_result




