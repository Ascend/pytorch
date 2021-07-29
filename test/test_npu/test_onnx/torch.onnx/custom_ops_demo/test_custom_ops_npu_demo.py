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
import torch.utils.cpp_extension
import cProfile
from cProfile import Profile
import numpy as np
from torch.onnx.symbolic_helper import parse_args, _parse_arg, _unimplemented
from torch.nn.parameter import Parameter
import math
from torch.nn  import init


import sys
from functools import wraps
from torch.autograd import Function

ONNX_NPU_OP_FILENAME = "onnx/custom_npu_op_demo.onnx"
ONNX_NPU_OP_MODULE_FILENAME = "onnx/custom_npu_op_module_demo.onnx"
def do_export(model, inputs, f= ONNX_NPU_OP_FILENAME, *args, **kwargs):
    out = torch.onnx._export(model, inputs, f, verbose=True, export_params=True, do_constant_folding=True,*args, **kwargs)
    #other method:
    other_method_b = False
    if (other_method_b) :
      out = torch.onnx.export_to_pretty_string(model, inputs, f, verbose=True, *args, **kwargs)
################################################################################
############################################################
#define & register npu_roi_align onnx symbolic 
############################################################

@parse_args('v', 'v', 'f', 'i', 'i', 'i', 'i')
def symbolic_npu_roi_align(g, input, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode):
    args = [input, rois]
    kwargs = {"spatial_scale_f": spatial_scale,
            "pooled_height_i": pooled_height,
            "pooled_width_i": pooled_width,
            "sample_num_i": sample_num,
            "roi_end_mode_i": roi_end_mode}

    return g.op('torch::npu_roi_align',*args, **kwargs)

def symbolic_npu_roi_align1(g, input,rois,spatial_scale, pooled_height, pooled_width , sample_num, roi_end_mode):
    #graph info:
    #%8 : Double() = onnx::Constant[value={-0.239442}]()
    #%9 : Long() = onnx::Constant[value={2}]()
    #%10 : Long() = onnx::Constant[value={0}]()
    #%11 : Long() = onnx::Constant[value={1}]()
    #%12 : Long() = onnx::Constant[value={1}]()
    #%13 : Float(8, 10, 2, 0) = torch::npu_roi_align(%intput, %npu_roi_align.weight, %8, %9, %10, %11, %12) # test_custom_ops_npu_demo.py:89:0
    return g.op('torch::npu_roi_align',input, rois, spatial_scale, pooled_height, pooled_width , sample_num, roi_end_mode)

import torch.onnx.symbolic_registry as sym_registry
def register_onnx_sym_npu_roi_align():
    if not sym_registry.is_registered_op('npu_roi_align', '',11):
      if 0:
          sym_registry.register_op('npu_roi_align', symbolic_npu_roi_align, '', 11)
      else:
          sym_registry.register_op('npu_roi_align', symbolic_npu_roi_align1, '', 11)

      print('=======================================================================================')
      print("'npu_roi_align' export onnx op registered state : ", sym_registry.is_registered_op('npu_roi_align', '',11))
      print('=======================================================================================')
    else:
      print('=======================================================================================')
      print("'npu_roi_align' export onnx op was registered : ", sym_registry.is_registered_op('npu_roi_align', '',11))
      print('=======================================================================================')
#################################
#custom Model: Npu OP demo
#################################
class CustomModel_npu_op(torch.nn.Module):
    def __init__(self,a,b):
        super(CustomModel_npu_op, self).__init__()

        self.weight = Parameter(torch.Tensor(8,10,1024))
        #other method: 
        #self.register_parameter('weight', a)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        

    #@staticmethod
    def forward(self, a, b, d):
        print("forward======================")
        print(type(a),a.dtype)
        print(type(b),b.dtype)
        print(type(d[0]),d[0].dtype)
        print(type(d[1]),d[1].dtype)
        print(type(d[2]),d[2].dtype)
        print(type(d[3]),d[3].dtype)
        print(type(d[4]),d[4].dtype)
        print("forward======================")
        
        spatial_scale=d[0].item()
        pooled_height=d[1].item()
        pooled_width=d[2].item()
        sample_num=d[3].item()
        roi_end_mode=d[4].item()
        print("spatial_scale:{},\n pooled_height:{},\n pooled_width:{},\n sample_num:{},\n roi_end_mode:{}\n".format(spatial_scale, pooled_height, pooled_width, sample_num,roi_end_mode))
        rtn = torch.npu_roi_align(a, self.weight, spatial_scale, pooled_height, pooled_width, sample_num,roi_end_mode)

        return rtn


#################################
#export onnx Demo: Npu OP demo
#################################
def export_custom_onnx_NO():
    register_onnx_sym_npu_roi_align()
    
    rnddata_device = 'cpu'
    #rnddata_device = 'npu'
    #rnddata_device = target_device

    a = torch.randn(5, 10, 1024, dtype=torch.float32, requires_grad=True,device=rnddata_device)
    b = torch.randn(10, 10, 1024, dtype=torch.float32, requires_grad=True,device=rnddata_device)
    c = torch.randn(10, dtype=torch.float32, requires_grad=False,device=rnddata_device)
    d = torch.randn(10, dtype=torch.float32, requires_grad=False,device=rnddata_device).int()
    e = torch.randn(10, dtype=torch.float32, requires_grad=False,device=rnddata_device).int()
    f = torch.randn(10, dtype=torch.float32, requires_grad=False,device=rnddata_device).int()
    g = torch.randn(10, dtype=torch.float32, requires_grad=False,device=rnddata_device).int()
    cx = c[0]
    if 1:
        d[0] = 2
        e[0] = 0
        f[0] = 1
        g[0] = 1
    if 1 :
        dx = d[0]
        ex = e[0]
        fx = f[0]
        gx = g[0]
    else :
        dx = 2
        ex = 0
        fx = 1
        gx = 1
    #print('{}-{}-{}-{}-{}-{}-{}'.format(a,b,c,d,e,f,g))
    print('type(cx[0])({}):type(dx[0])({}):{}-{}-{}-{}-{}'.format(type(cx.item()),type(dx),cx,dx,ex,fx,gx))
    #print('type(cx[0])({}):type(dx[0])({}):{}-{}-{}-{}-{}'.format(type(cx.item()),type(dx.item()),cx,dx,ex,fx,gx))
    model = CustomModel_npu_op(a,b)
    model = model.npu()
    model.eval()
    h = [dx, ex, fx, gx]
    h1 = [cx, dx, ex, fx, gx]
    #do_export(model, (a ,b, c, d, e, f, g), opset_version=11)
    #do_export(model, (a ,b, c,h), opset_version=11)
    a = a.to(target_device)
    b = b.to(target_device)
    #c = c.to("npu:0")
    #d = d.to("npu:0")
    #e = e.to("npu:0")
    #f = f.to("npu:0")
    #g = g.to("npu:0")
    print("======================")
    print(type(a),a.dtype)
    print(type(b),b.dtype)
    print(type(h1[0]),h1[0].dtype)
    print(type(h1[1]),h1[1].dtype)
    print(type(h1[2]),h1[2].dtype)
    print(type(h1[3]),h1[3].dtype)
    print(type(h1[4]),h1[4].dtype)
    print("======================")
    #do_export(model, (a ,b, h1), f=ONNX_NPU_OP_FILENAME, opset_version=11)
    #do_export(model, (a ,b, h1), f=ONNX_NPU_OP_FILENAME, input_names=["intput"]+["1","2","3","4","5","6","npu_roi_align.weight"],opset_version=11)
    do_export(model, (a ,b, h1), f=ONNX_NPU_OP_FILENAME, input_names=["intput"]+["","","","","","","npu_roi_align.weight"],opset_version=11)
    #do_export(model, (a ,b, h1), f=ONNX_NPU_OP_FILENAME, input_names=["intput"]+[None,None,None,None,None,None,"npu_roi_align.weight"],opset_version=11)

################################################################################

############################################################
#class op Function
############################################################
class CustomClassOp_Func_npu_roi_align(Function):
    @staticmethod
    def forward(ctx,input, rois, spatial_scale, pooled_height, pooled_width , sample_num, roi_end_mode):
        rtn = torch.npu_roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode)
        return rtn
        
    @staticmethod
    def symbolic(g, input, rois, spatial_scale, pooled_height, pooled_width , sample_num, roi_end_mode):
        if 1:
            args = [input, rois]
            kwargs = {"spatial_scale_f": spatial_scale,
                    "pooled_height_i": pooled_height,
                    "pooled_width_i": pooled_width,
                    "sample_num_i": sample_num,
                    "roi_end_mode_i": roi_end_mode}
            return g.op('torch::npu_roi_align',*args, **kwargs)
        elif 0:
            return g.op('torch::npu_roi_align', input, rois, spatial_scale_f=spatial_scale, pooled_height_i=pooled_height, pooled_width_i=pooled_width, sample_num_i=sample_num, roi_end_mode_i=roi_end_mode)
        elif 0:
            #error branch:
            #Invoked with: %8 : Tensor = onnx::Constant(), 'value', -0.4300241768360138 (occurred when translating CustomClassOp_Func_npu_roi_align)
            return g.op('torch::npu_roi_align', input, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode)
        elif 0:
            #error branch:
            #Invoked with: %8 : Tensor = onnx::Constant(), 'value', -1.2140488624572754 (occurred when translating CustomClassOp_Func_npu_roi_align)
            args = [input, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode]
            return g.op('torch::npu_roi_align',*args)
        elif 0:
            #error branch:
            #IndexError: Invalid attribute specifier 'pooled_height' names  must be suffixed with type, e.g. 'dim_i' or 'dims_i'
            return g.op('torch::npu_roi_align', input, rois, spatial_scale=spatial_scale, pooled_height=pooled_height, pooled_width=pooled_width, sample_num=sample_num, roi_end_mode=roi_end_mode)
        else:
            print("no proc for CustomClassOp_Func_npu_roi_align")

############################################################
#custom npu_roi_align OP Module 
############################################################
class NpuOp_npu_roi_align_Module(torch.nn.Module):
    def __init__(self):
        super(NpuOp_npu_roi_align_Module, self).__init__()

        self.spatial_scale = torch.randn(10, dtype=torch.float32, requires_grad=False,device="cpu")[0].item()
        self.pooled_height = 2
        self.pooled_width = 0
        self.sample_num = 1
        self.roi_end_mode = 1

        self.weight = Parameter(torch.Tensor(8,10,1024))
        #other method: 
        #self.register_parameter('weight', a)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.func = CustomClassOp_Func_npu_roi_align.apply
        self.test_npu_op=1
        if(self.test_npu_op):
          register_onnx_sym_npu_roi_align()
    def forward(self, input):
        if(self.test_npu_op):
          rtn = torch.npu_roi_align(input, self.weight, self.spatial_scale, self.pooled_height, self.pooled_width, self.sample_num, self.roi_end_mode)
          input = rtn
        rtn = self.func(input, self.weight, self.spatial_scale, self.pooled_height, self.pooled_width, self.sample_num, self.roi_end_mode)
        return rtn

#################################
#custom Model demo
#################################
class CustomModel_Module_op(torch.nn.Module):
    def __init__(self,a,b):
        super(CustomModel_Module_op, self).__init__()
        self.npu_roi_align = NpuOp_npu_roi_align_Module()
    #@staticmethod
    def forward(self, a):
        rtn = self.npu_roi_align(a) 
        return rtn

def export_custom_onnx_MO():
    rnddata_device = 'cpu'
    #rnddata_device = 'npu'
    #rnddata_device = target_device

    a = torch.randn(5, 10, 1024, dtype=torch.float32, requires_grad=True,device=rnddata_device)
    b = torch.randn(10, 10, 1024, dtype=torch.float32, requires_grad=True,device=rnddata_device)
    model = CustomModel_Module_op(a,b)
    model = model.npu()
    model.eval()
    a = a.to(target_device)
    b = b.to(target_device)
    print("======================")
    print(type(a),a.dtype)
    print(type(b),b.dtype)
    print("======================")
    #do_export(model, a, f=ONNX_NPU_OP_MODULE_FILENAME, opset_version=11)
    do_export(model, a, f=ONNX_NPU_OP_MODULE_FILENAME, input_names=["intput"]+["npu_roi_align.weight"],opset_version=11)

################################################################################
#torch.npu.device(0)
target_device = 'npu:6'
torch.npu.set_device(target_device)

export_custom_onnx_NO()
export_custom_onnx_MO()

def profile_test():
    prof = Profile()
    prof.enable()
    export_custom_onnx()
    prof.create_stats()
    prof.print_stats()
#profile_test()
#cProfile.run('export_custom_onnx()')

