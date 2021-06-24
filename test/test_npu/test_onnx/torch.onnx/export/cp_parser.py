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

from collections import OrderedDict

import logging

logger_level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

loggerScreanHander = logging.StreamHandler()
if 0 :
    loggerScreanHander.setFormatter(logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s') )
else:
    loggerScreanHander.setFormatter(logging.Formatter('%(message)s') )

logger = logging.getLogger('torch.onnx.export.test.cp_parser')
logger.addHandler(loggerScreanHander)
logger.setLevel(logger_level_relations.get('debug'))
logger.debug('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))


def cp_getDeviceId(deviceStr, DeviceNo):
    logger.debug("device: ( {0}  {1} )".format(deviceStr,DeviceNo))
    if DeviceNo == None:
        return deviceStr
    if deviceStr == 'cpu':
        return deviceStr
    elif deviceStr == 'npu' or deviceStr == 'cuda':
        loc = '{}:{}'.format(deviceStr, DeviceNo)
        return loc
    else: 
        return deviceStr

def cp_printNamedparameters(model):
    i=0
    parm = {}
    logger.debug("+++++++++++++++++++++++++++++++++++++printParm model begin")
    for name,parameters in model.named_parameters():
        # parameters.register_hook(lambda grad: print(grad.to("cpu")))
        logger.debug(parameters.grad_fn)
        if 1:
            logger.info(name,':',parameters.size(),'+++{0}+++{1}'.format(parameters.dtype,torch.sum(parameters)))
        else:
            logger.debug(name,':',parameters.size(),'+++{0}+++{1}'.format(parameters.dtype,parameters.sum()))
            logger.debug(parameters.to("cpu"))
            logger.debug(name,':',parameters.size(),'+++{0}+++{1}'.format(parameters.dtype,torch.sum(parameters).to("cpu")))
        
            if i==0:
                print("==========================main test begin")
                print(parameters)
                print("==========================main test end")
        #parm[name]=parameters.detach().to("cpu").numpy()
        logger.debug(parm[name])
        i+=1


    if 0:
        print('conv1.weight:',parm['conv1.weight'].dtype)
        print("-------------------------------start","conv1.weight")
        print(parm['conv1.weight'])
        print("-------------------------------end","conv1.weight")
    logger.info("-------------------------------------------------")
    logger.info(i)
    logger.info("-------------------------------------------------")
    para = list(model.parameters())
    logger.info(len(para))
    logger.info("-------------------------------------------------")
    if 0:
        logger.info(para)
    logger.debug("+++++++++++++++++++++++++++++++++++++printParm model end")



def cp_getParameterNameList(model):
    NameList = []
    for name,parameters in model.named_parameters():
        NameList.append(name)
    for name,parameters in model.named_buffers():
        NameList.append(name)
    return NameList


def cp_getNamedparametersNameList(model):
    NameList = []
    for name,parameters in model.named_parameters():
        NameList.append(name)
    return NameList


def cp_getNamedbufferNameList(model):
    NameList = []
    for name,parameters in model.named_buffers():
        NameList.append(name)
    return NameList

def cp_getParamTensorValue(input_name,model):
    for name,parameters in model.named_parameters():
        if input_name == name :
            return parameters.detach()
    for name,buf in model.named_buffers():
        if input_name == name :
            return buf
    return None

G_CHECKPOINT_INPUTFILE = "./checkpoint.pth.tar"

#G_ONNX_OUTFILE =  "resnet50.onnx"
#G_PTH_OUTFILE = 'resnet50-19c8e357.pth'


G_DEVICE_STRINT = "cuda"
G_ONNX_OUTFILE =  "densenet121.onnx"
G_PTH_OUTFILE = 'densenet121-a639ec97.pth'

#G_ONNX_OUTFILE =  "densenet121.onnx"
#G_PTH_OUTFILE = 'densenet121-a639ec97.pth'

def save_pthfile(state, filename=G_PTH_OUTFILE):
    torch.save(state, filename)

def proc_nodes_module(checkpoint,AttrName):
    new_state_dict = OrderedDict()
    for k,v in checkpoint[AttrName].items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name]=v
    # for k,v in checkpoint[AttrName].items():
    #     if(k[0:9] == "features."):
    #         name = k[9:]
    #     else:
    #         name = k[0:]
    #     new_state_dict[name]=v
    return new_state_dict

def load_tar(tarfilepath,model=None):
    if os.path.isfile(tarfilepath):
        logger.debug("=> loading checkpoint '{}'".format(tarfilepath))
        checkpoint = torch.load(tarfilepath,map_location=torch.device(G_DEVICE_STRINT))
        if 0:
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
               #best_acc1 may be from a checkpoint from a different GPU
               best_acc1 = best_acc1.to(args.gpu)
        if  model is not None :
            model.load_state_dict(checkpoint['state_dict'])
        if 0:
            optimizer.load_state_dict(checkpoint['optimizer'])
        logger.debug("=> loaded checkpoint '{}' (epoch {})"
              .format(tarfilepath, checkpoint['epoch']))
    else:
        logger.debug("=> no checkpoint found at '{}'".format(tarfilepath))
    checkpoint['state_dict'] = proc_nodes_module(checkpoint,'state_dict')
    save_pthfile(checkpoint['state_dict'])
    return model

def cp_load_return_check(model,cpFile,ispth=False,device=None,dno=None):

    if os.path.isfile(cpFile):
        logger.debug("[device id:{0}:{1}]=> loading checkpoint '{2}'".format(device,dno,cpFile))
        if 0:
            if dno is None:
                checkpoint = torch.load(cpFile, map_location="cpu")
            else:
                # Map model to be loaded to specified single gpu.
                if device == 'npu':
                    loc = 'npu:{}'.format(dno)
                    checkpoint = torch.load(cpFile)
                else:
                    loc = 'cuda:{}'.format(dno)
                    checkpoint = torch.load(cpFile, map_location=loc)
        if device is None:
            if 0:
                checkpoint = torch.load(cpFile)
            else:
                checkpoint = torch.load(cpFile,map_location='cpu')
        else :
            map_location = cp_getDeviceId(device,dno) 
            checkpoint = torch.load(cpFile, map_location=map_location)
        
        if ispth :
            model.load_state_dict(checkpoint)
            logger.debug("[device id:{0}:{1}]=> loading checkpoint '{2}'".format(device,dno,cpFile))
        else :
            if 0:
                start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                if device == 'npu':
                   best_acc1 = best_acc1.to(loc)
                else:
                   if dno is not None:
                       # best_acc1 may be from a checkpoint from a different GPU
                       best_acc1 = best_acc1.to(dno)
            checkpoint['state_dict'] = proc_nodes_module(checkpoint,'state_dict')
            save_pthfile(checkpoint['state_dict'],filename=cpFile+".pth")
            model.load_state_dict(checkpoint['state_dict'])
            if 0:
                optimizer.load_state_dict(checkpoint['optimizer'])
                if device == 'npu':
                   model=model.to(loc)
                logger.debug("[device id:{0}:{1}] => loading checkpoint '{2}' (epoch {3})".format(device,dno,cpFile, checkpoint['epoch']))
        return model,checkpoint
    else:
        logger.info("[device id:{0}:{1}] => no checkpoint found at '{2}'".format(device,dno,cpFile))
        return None,None

def cp_load(model,cpFile,ispth=False,device=None,dno=None) :
    model,checkpoint = cp_load_return_check(model,cpFile,ispth=ispth,device=device,dno=dno)
    return model
