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

import torch
import torchvision
from export.cp_parser import *

def getDeviceStr(deviceStr, DeviceNo):
    #print("cp_getDeviceId test device : ","(", deviceStr,"  ", DeviceNo, ")")
    if DeviceNo == None:
        return deviceStr
    if deviceStr == 'cpu':
        return deviceStr
    elif deviceStr == 'npu' or deviceStr == 'cuda':
        loc = '{}:{}'.format(deviceStr, DeviceNo)
        return loc
    else: 
        return deviceStr


def cp2onnx(model,cpfile,onnxfile, input_data, ispth=False,device="cpu",dno=None):
    if os.path.isfile(cpfile):
        #model = torchvision.models.resnet50(pretrained=False)
        model = cp_load(model,cpfile,ispth=ispth,device=device,dno=dno)
    else :
        print("warning : \"",cpfile,"\"not exist!")
        model.state_dict()
    deviceStr = getDeviceStr(device,dno)
    print("cp2onnx device: ",deviceStr,"(",device," ",dno,")")
    #torch.npu.set_device("npu:0")
    #dummy_input = torch.randn(10, 3, 224, 224, device='npu:0')
    dummy_input = input_data.to(deviceStr)

    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.
    input_names = [ "actual_input_1" ] #+ [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]
    model = model.to(deviceStr)
    torch.onnx.export(model, dummy_input, onnxfile, verbose=True, input_names=input_names, output_names=output_names,opset_version=11)


def cp2onnx_dynamic_axes(model,cpfile,onnxfile,device="cuda",dno=None):
    if os.path.isfile(cpfile):
        #model = torchvision.models.resnet50(pretrained=False)
        model = cp_load(model,cpfile)
    else :
        print("warning : \"",cpfile,"\"not exist!")
        model.state_dict()
    deviceStr = getDeviceStr(device,dno)
    #torch.npu.set_device("npu:0")
    #dummy_input = torch.randn(10, 3, 224, 224, device='npu:0')
    dummy_input = torch.randn(10, 3, 224, 224)
    dummy_input = dummy_input.to(deviceStr)

    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.
    input_names = [ "actual_input_1" ] #+ [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]
    model = model.to(deviceStr)
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {0: '-1'}}
    torch.onnx.export(model, dummy_input, onnxfile, verbose=True, input_names=input_names, output_names=output_names,dynamic_axes=dynamic_axes,opset_version=11)


