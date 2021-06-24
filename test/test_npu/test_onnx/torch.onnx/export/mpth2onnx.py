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


import os
import torch

import logging

logger_level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

loggerScreanHander = logging.StreamHandler()
if 0:
    loggerScreanHander.setFormatter(logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s') )
else:
    loggerScreanHander.setFormatter(logging.Formatter('%(message)s') )

logger = logging.getLogger('torch.onnx.export.test.mpth2onnx')
logger.addHandler(loggerScreanHander)
logger.setLevel(logger_level_relations.get('info'))
logger.debug('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))


def mpth_getDeviceId(deviceStr, DeviceNo):
    logger.debug("cp_getDeviceId test device : ( {0}  {1} )".format(deviceStr, DeviceNo))
    if DeviceNo == None:
        return deviceStr
    if deviceStr == 'cpu':
        return deviceStr
    elif deviceStr == 'npu' or deviceStr == 'cuda':
        loc = '{}:{}'.format(deviceStr, DeviceNo)
        return loc
    else: 
        return deviceStr



def model2onnx(model,onnxfile, input_data, device="cpu",dno=None):
    
    deviceStr = mpth_getDeviceId(device,dno)
    logger.debug("cp2onnx device: {0}( {1}  {2} )".format(deviceStr,device,dno))
    
    #data init as : dummy_input = torch.randn(10, 3, 224, 224, device='npu:0')
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
    input_names = [ "actual_input_1" ] 
    output_names = [ "output1" ]
    model = model.to(deviceStr)
    torch.onnx.export(model, dummy_input, onnxfile, verbose=False, input_names=input_names, output_names=output_names,opset_version=11)
    return 0

def mpth2onnx(mpthfile,onnxfile, input_data, device="cpu",dno=None):

    if os.path.isfile(mpthfile):
        model = torch.load(mpthfile)
    else :
        logger.error( 'file(\"{0}\") not exist! '.format(mpthfile))
        return -1

    return model2onnx(model,onnxfile, input_data, device=device,dno=dno)





