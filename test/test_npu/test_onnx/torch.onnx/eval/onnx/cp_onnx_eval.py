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

from collections import OrderedDict

from export.onnx_parser import *
from export.cp_parser import *
def b_equal(a,b,length):
    for i in range(length):
        if a[i] != b[i]:
            return False
    return True

def onnx_cp_compare_rawdata(nameList,cp_model,onnx_model) :
    SuccessTList = []
    FailTList = []
    cpTNotFoundList = []
    onnxTNotFoundList = []
    i = 0
    print("-------------------------------------------------")
    for name in nameList:
        cp_t = cp_getParamTensorValue(name,cp_model)
        onnx_t_b = onnx_getInitTensorValue(name,onnx_model)
        if(cp_t is not None) and (onnx_t_b is not None):
            cp_t = cp_t.to("cpu")
            cp_t_n = cp_t.data.numpy()
            #cp_t_n_b = cp_t_n.astype(bytes)
            cp_t_n_b = bytes(cp_t_n)
            
            #cp_t_n_b_n = np.array(cp_t_n_b)
            #onnx_t_n = np.array(onnx_t_b,dtype=np.float32,ndmin=4)
            #onnx_t_n = np.array(onnx_t_b)
            
            # print("cp_t type:",type(cp_t),cp_t.size)
            # print("cp_t_n type:",type(cp_t_n),cp_t_n.size,":",cp_t_n.dtype)
            # #print("cp_t_n astype(bytes) type:",type(cp_t_n_b),cp_t_n_b.size,":",cp_t_n_b.dtype)
            # print("cp_t_n_b type:",type(cp_t_n_b),len(cp_t_n_b))

            # print("onnx_t_b type:",type(onnx_t_b),len(onnx_t_b))
            # print("onnx_t_n type:",type(onnx_t_n),onnx_t_n.size)
            

            # print("-------------------------------------------------")
            # print(name)
            # print("-------------------------------------------------")
            # #print(cp_t_n)
            # #print(cp_t_n_b)
            # print("-------------------------------------------------")
            # #print(onnx_t_b)
            # #print(onnx_t_n)
            # #print(onnx_t_b.decode('UTF-8','strict'))
            # print("-------------------------------------------------")

            #result = np.equal(cp_t_n,onnx_t_n)
            result = b_equal(cp_t_n_b,onnx_t_b,len(onnx_t_b))
            #if result.all() == False:
            if result == False:
                print("[Fail]",str(i+1),":",name)
                FailTList.append(name)
            else :
                #print("[success]",str(i+1),":",name)
                SuccessTList.append(name)
                
        else :
            if (cp_t is None) : 
                cpTNotFoundList.append(name)
            if (onnx_t_b is None) :
                onnxTNotFoundList.append(name)
        i += 1
    print("-------------------------------------------------")
    print("[Count  ]",i)
    print("[success]",len(SuccessTList))
    #print(SuccessTList)
    print("[Fail   ]",len(FailTList))
    print(FailTList)
    print("[CP   NF]",len(cpTNotFoundList))
    print(cpTNotFoundList)
    print("[ONNX NF]",len(onnxTNotFoundList))
    print(onnxTNotFoundList)
    print("-------------------------------------------------")
    return SuccessTList,FailTList,cpTNotFoundList,onnxTNotFoundList

def onnx_runeval(model,cp_path,onnx_path,ispth=False):
    # Load the ONNX model
    onnx_model = onnx.load(onnx_path)
    
    cp_model = cp_load(model,cp_path,ispth=ispth)
    #onnx_cp_compare_rawdata(cp_getParameterNameList(cp_model),cp_model,onnx_model)
    onnx_cp_compare_rawdata(onnx_getParameterNameList(onnx_model),cp_model,onnx_model)
