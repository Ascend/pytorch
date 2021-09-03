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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from apex import amp
import numpy as np
import onnx
import onnxruntime as ort

def onnx_export():
    pred_mask = torch.randn((1, 3, 512, 512)).cuda()
    model = UNet(num_classes=6).eval().cuda()
    state = {'net': model.state_dict()}
    torch.save(state, 'unet.pth')

    checkpoint = torch.load('unet.pth')
    model.load_state_dict(checkpoint['net'])

    for i in range(10):
        with torch.no_grad():
            start = time.time()
            out = model(pred_mask)
            print("torchtime", time.time() - start)

    input_names = ["input0"]
    output_names = ["output0"]
    onnx_path = 'unet.onnx'
    # pth??onnx
    torch.onnx.export(model, pred_mask, onnx_path, verbose=False, input_names=input_names,
                       output_names=output_names, opset_version=11)
def onnx_getNodeNum(model):
    return len(model.graph.node)

def onnx_getNodetype(model):
    op_name = []
    for i in range(len(model.graph.node)):
        if model.graph.node[i].op_type not in op_name:
            op_name.append(model.graph.node[i].op_type)
    return op_name

def onnx_getNodeNameList(model):
    NodeNameList = []
    for i in range(len(model.graph.node)):
        #print(model.graph.node[i].name)
        NodeNameList.append(model.graph.node[i].name)
    return NodeNameList

def onnx_printNodeList(model):
    NodeNameList = []
    for i in range(len(model.graph.node)):
        print("-------------------------------------------------")
        print(model.graph.node[i])
        print("-------------------------------------------------")
        #NodeNameList.append(model.graph.node[i].name)
    return model.graph.node



def onnx_getModelInputInfo(model,i):
    #return model.graph.input[0]
    return model.graph.input[i]

def onnx_getModelOutputInfo(model):
    return model.graph.output[0:4]

def onnx_getNodeAndIOname(nodename,model):
    for i in range(len(model.graph.node)):
        if model.graph.node[i].name == nodename:
            Node = model.graph.node[i]
            input_name = model.graph.node[i].input
            output_name = model.graph.node[i].output
    return Node,input_name,output_name

def onnx_getInputTensorValueInfo(input_name,model):
    in_tvi = []
    for name in input_name:
        for params_input in model.graph.input:
            if params_input.name == name:
               in_tvi.append(params_input)
        for inner_output in model.graph.value_info:
            if inner_output.name == name:
                in_tvi.append(inner_output)
    return in_tvi

def onnx_printInputTensorValueInfo(model):
    in_tvi = []

    print("*************************************************")
    print(model.graph.input)
    print("-------------------------------------------------")
    for params_input in model.graph.input:
        print(params_input)
        print("-------------------------------------------------")
    print(model.graph.value_info)
    print("-------------------------------------------------")
    for inner_output in model.graph.value_info:
        print(inner_output)
    print("*************************************************")
    return in_tvi

def onnx_getOutputTensorValueInfo(output_name,model):
    out_tvi = []
    for name in output_name:
        out_tvi = [inner_output for inner_output in model.graph.value_info if inner_output.name == name]
        if name == model.graph.output[0].name:
            out_tvi.append(model.graph.output[0])
    return out_tvi

def onnx_getInitTensorValueList(input_nameList,model):
    init_t = []
    for name in input_nameList:
        init_t = [init for init in model.graph.initializer if init.name == name]
    return init_t



def onnx_getParameterNameList(model):
    NameList = []
    for init in model.graph.initializer:
        NameList.append(init.name)
    return NameList

def onnx_getInitTensorValue(input_name,model):
    for init in model.graph.initializer:
        if init.name == input_name :
            return init.raw_data
    return None

def onnx_printInitTensorValue(model):
    init_t = []
    print("-------------------------------------------------")
    print(len(model.graph.initializer))
    print("-------------------------------------------------")
    #print(model.graph.initializer)
    #print("-------------------------------------------------")
    for init in model.graph.initializer :
        print(init.name,init.dims,init.data_type)  #init.raw_data
    print("-------------------------------------------------")
    return init_t


def printModelInfo(model):
    print("=====================================")
    print(onnx_getModelInputInfo(model,0))
    print("=====================================")
    print(onnx_getModelOutputInfo(model))
    print("=====================================")
    print(onnx_getNodeNameList(model))
    print("=====================================")
    print(onnx_getNodetype(model))
    print("=====================================")
    print(onnx_getInputTensorValueInfo("fc.bias",model))
    print("=====================================")
    #onnx_printNodeList(model)
    #onnx_printInputTensorValueInfo(model)
    print("=====================================")
    onnx_printInitTensorValue(model)
    print("=====================================")
    
def onnxrt_eval(onnx_path, onnx_model):
    output_dir = os.getcwd()
    # onnx spend time
    print(ort.get_device())
    
    ort_session = ort.InferenceSession(os.path.join(output_dir, onnx_path))

    img_height = 224
    img_width = 224
    #x_input = pred_mask.data.cpu().numpy()
    x_input = np.random.rand( 10, 3, img_height, img_width).astype('float32')
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    """
    #outputs = [session.run([], {input_name: inputs[i]})[0] for i in range(test_data_num)]
    outputs = [ort_session.run([], {input_name: x_input})[0] for i in range(10)]
    print('Predicted {} results.'.format(len(outputs)))
    print(outputs)
    """
    for i in range(10):
        print("=================",i)
        start = time.time()
        outputs = ort_session.run([output_name], {input_name: x_input})
        print("orttime", time.time() - start)
        print(type(outputs),len(outputs),"===",type(outputs[0]),len(outputs[0]))
        print(outputs)
        res = postprocess(outputs)
        #acc1, acc5 = accuracy(outputs, x_input, topk=(1, 5))
        #losses.update(loss.item(), images.size(0))
        #top1.update(acc1[0], images.size(0))
        #top5.update(acc5[0], images.size(0))
        #progress.display(i)
    
    print("=====================================")


def checkOnnx(onnx_path, onnx_model):
    # Check that the IR is well formed
    print("=====================================")
    print("onnx.checker.check_model(onnx_model)")
    print("=====================================")
    rtn = onnx.checker.check_model(onnx_model)
    print(type(rtn))
    print(rtn)
    print("=====================================")
    print("onnx.helper.printable_graph(onnx_model.graph)")
    print("=====================================")
    rtn = onnx.helper.printable_graph(onnx_model.graph)
    print(type(rtn))
    print(rtn)
    print("=====================================")
    

