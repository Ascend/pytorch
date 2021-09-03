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
torch.npu.set_device("npu:0")
#dummy_input = torch.randn(10, 3, 224, 224, device='npu:0')
dummy_input = torch.randn(10, 3, 224, 224)
dummy_input = dummy_input.to("npu")
model = torchvision.models.resnet50(pretrained=False)

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]
model = model.to("npu")
torch.onnx.export(model, dummy_input, "resnet50.onnx", verbose=True, input_names=input_names, output_names=output_names)



# 有坑 会提示下载不下来 修改下resnet.py，手动下载下来，然后放到 D:/Pytorch/models 目录下。
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], model_dir="D:/Pytorch/models",
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
