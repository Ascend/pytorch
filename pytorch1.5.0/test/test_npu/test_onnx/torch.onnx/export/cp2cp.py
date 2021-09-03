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

from collections import OrderedDict

import logging

logger_level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

loggerScreanHander = logging.StreamHandler()
if 0 :
    loggerScreanHander.setFormatter(logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s') )
else:
    loggerScreanHander.setFormatter(logging.Formatter('%(message)s') )

logger = logging.getLogger('torch.onnx.export.test.cp2cp')
logger.addHandler(loggerScreanHander)
logger.setLevel(logger_level_relations.get('info'))
logger.debug('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))




from export.mpth2onnx import *

#from .. import export
from export.cp_parser import *
# from export.export_onnx import *
# from eval.onnxrt.onnxrt_eval import *
# from eval.onnx.cp_onnx_eval import *



def densnet_convert():
    model = models.densenet121(pretrained = False)
    pthfile = G_CP_DENSENET121_NPU_FILEPATH
    state_dict =torch.load(str(pthfile))
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if 'denseblock' in k:
            param = k.split(".")
            k = ".".join(param[:-3] + [param[-3]+param[-2]] + [param[-1]])
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)


    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(16, 3, 224, 224)
    torch.onnx.export(model, dummy_input,G_ONNX_DENSENET121_NPU_FILEPATH, input_names = input_names, output_names = output_names, opset_version=11)

def print_checkpointParamName(checkpoint):
    for key, value in checkpoint.items():
        print(key)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)







#################################################################################################
#set configure for test
#################################################################################################
G_DEVICE_STRINT = "cpu"
G_DEVICE_NO_STRINT     = 0

G_CP2CP_CP2CP_FLAG     = 0
G_CP2CP_CP2MPTH_FLAG   = 1
G_CP2CP_MPTH2ONNX_FLAG = 1

G_FLAG_MODLE_Resnet50        = 1
G_FLAG_MODLE_Densenet121     = 1
G_FLAG_MODLE_Efficientnet_b0 = 1
G_FLAG_MODLE_shufflenetv2    = 0
G_FLAG_MODLE_mobilenetv2     = 1
G_FLAG_MODLE_DeepMAR         = 1

#################################################################################################

logger.info('===================================================================')
logger.info('resnet50')
logger.info('===================================================================')


#################################################################################################
#test resnet50
import torchvision


def resnet50_model_eval(cp_path, input_data, dimx=1):
    if os.path.isfile(cp_path):
        resnet50 = torchvision.models.resnet50(pretrained=False)
        model = cp_load(resnet50,cp_path)
        model.eval()
        out = model(input_data)
        return out
    else :
        logger.error( 'file(\"{0}\") not exist! '.format(cp_path))
        return None

def resnet50_loadModelfile_eval(modelFile, input_data):
    if os.path.isfile(modelFile):
        model_test = torch.load(modelFile)
        model_test.eval()
        out = model_test(input_data)
        return out
    else :
        logger.error( 'file(\"{0}\") not exist! '.format(modelFile))
        return None

def test_resnet5_npu_cp2cpu(cpFile, outCpFile):
    if os.path.isfile(cpFile):
        model = torchvision.models.resnet50(pretrained=False)
        model,check = cp_load_return_check(model,cpFile,ispth=False,device="cpu",dno=None)
        save_checkpoint({
            'epoch': check['epoch'],
            'arch': check['arch'],
            'state_dict': model.state_dict(),
            'best_acc1': check['best_acc1'],
            #'optimizer' : optimizer.state_dict(),
        }, outCpFile)
        logger.info("Save checkpoint \"tar file\": \"{0}\" success!".format(outCpFile))
    else :
        logger.error( 'file(\"{0}\") not exist! '.format(cpFile))
        return None


def test_resnet5_npu_cp2mpth(outCpFile, outModelMpthFile):
    if os.path.isfile(outCpFile):
        model = torchvision.models.resnet50(pretrained=False)
        model,check = cp_load_return_check(model,outCpFile,ispth=False,device=None,dno=None)

        torch.save(model, outModelMpthFile)
        logger.info("Save model & parameter \"mpth file\": \"{0}\" success!".format(outModelMpthFile))
    else :
        logger.error( 'file(\"{0}\") not exist! '.format(outCpFile))
        return None


#################################################################################################

#test resnet50v2
#G_ONNX_resnet50v2_FILEPATH = 'resnet50v2/resnet50v2.onnx'
#onnxrt_runeval_resnet50_pic(G_CP_RESNET50_FILEPATH,G_ONNX_resnet50v2_FILEPATH,dimx=1)


G_CP_RESNET50_NPU_FILEPATH = 'cpfile/pulishCP/resnet50/checkpoint.pth.tar'
G_CP_RESNET50_NPU_OUT_FILEPATH = 'cpfile/outCP/resnet_cpu20200928_1800_out.pth.tar'
G_MPTH_RESNET50_NPU_OUT_MODEL_FILEPATH = 'cpfile/outModel/resnet_cpu20200928_1800_out_mp.mpth'
if G_FLAG_MODLE_Resnet50:
    if G_CP2CP_CP2CP_FLAG:
        test_resnet5_npu_cp2cpu(G_CP_RESNET50_NPU_FILEPATH, G_CP_RESNET50_NPU_OUT_FILEPATH)

    if G_CP2CP_CP2MPTH_FLAG:
        test_resnet5_npu_cp2mpth(G_CP_RESNET50_NPU_OUT_FILEPATH, G_MPTH_RESNET50_NPU_OUT_MODEL_FILEPATH)
        
    if G_CP2CP_MPTH2ONNX_FLAG:
        G_MPTH2ONNX_RESNET50_NPU_OUT_MODEL_FILEPATH = 'mpth2onnxfile/resnet_cpu20200928_1800_out_mp.onnx'
        input_data =  torch.randn(10, 3, 224, 224)
        rtn = mpth2onnx(G_MPTH_RESNET50_NPU_OUT_MODEL_FILEPATH,G_MPTH2ONNX_RESNET50_NPU_OUT_MODEL_FILEPATH, input_data, device=G_DEVICE_STRINT,dno=G_DEVICE_NO_STRINT)
        if not rtn:
            logger.info("model mpth file export \"onnx file\": \"{0}\" success!".format(G_MPTH2ONNX_RESNET50_NPU_OUT_MODEL_FILEPATH))
            
        


logger.info('===================================================================')
logger.info('densenet121')
logger.info('===================================================================')

#################################################################################################
#test densenet121
from models.densnet121.densenet_0_2_2 import *

def test_densenet121_npu__cp2cpu(cpFile, outCpFile):
    if os.path.isfile(cpFile):
        model = densenet121(pretrained=False)
        model,check = cp_load_return_check(model,cpFile,ispth=False,device="cpu",dno=None)
        save_checkpoint({
            'epoch': check['epoch'],
            'arch': check['arch'],
            'state_dict': model.state_dict(),
            'best_acc1': check['best_acc1'],
            #'optimizer' : optimizer.state_dict(),
        }, outCpFile)
        logger.info("Save checkpoint \"checkpoint file\": {0}  success!".format(outCpFile))
    else :
        logger.error( 'file(\"{0}\") not exist! '.format(cpFile))
        return None

def test_densenet121_npu_cp2mpth(outCpFile, outModelMpthFile):
    if os.path.isfile(outCpFile):
        model = densenet121(pretrained=False)
        model,check = cp_load_return_check(model,outCpFile,ispth=False,device="cpu",dno=None)

        torch.save(model, outModelMpthFile)
        logger.info("Save model & parameter \"mpth file\": {0}  success!".format(outModelMpthFile))
    else :
        logger.error( 'file(\"{0}\") not exist! '.format(outCpFile))
        return None

#################################################################################################
G_CP_DENSENET121_NPU_FILEPATH = 'cpfile/pulishCP/densenet121/densenet121_cpu.pth.tar'
G_CP_DENSENET121_NPU_OUT_FILEPATH = 'cpfile/outCP/densenet121_cpu_out.pth.tar'
G_MPTH_DENSENET121_NPU_OUT_MODEL_FILEPATH = 'cpfile/outModel/densenet121_cpu_out_mp.mpth'

if G_FLAG_MODLE_Densenet121:
    if G_CP2CP_CP2CP_FLAG:
        test_densenet121_npu__cp2cpu(G_CP_DENSENET121_NPU_FILEPATH, G_CP_DENSENET121_NPU_OUT_FILEPATH)

    if G_CP2CP_CP2MPTH_FLAG:
        test_densenet121_npu_cp2mpth(G_CP_DENSENET121_NPU_OUT_FILEPATH, G_MPTH_DENSENET121_NPU_OUT_MODEL_FILEPATH)

    if G_CP2CP_MPTH2ONNX_FLAG:
        G_MPTH2ONNX_DENSENET121_NPU_OUT_MODEL_FILEPATH = 'mpth2onnxfile/densenet121_cpu_out_mp.onnx'
        input_data =  torch.randn(10, 3, 224, 224)
        rtn = mpth2onnx(G_MPTH_DENSENET121_NPU_OUT_MODEL_FILEPATH,G_MPTH2ONNX_DENSENET121_NPU_OUT_MODEL_FILEPATH, input_data, device=G_DEVICE_STRINT,dno=G_DEVICE_NO_STRINT)
        if not rtn:
            logger.info("model mpth file export \"onnx file\": \"{0}\" success!".format(G_MPTH2ONNX_DENSENET121_NPU_OUT_MODEL_FILEPATH))

#################################################################################################

logger.info('===================================================================')
logger.info('efficientnet')
logger.info('===================================================================')
#################################################################################################
#test efficientnet-b0
from models.efficientnet_pytorch.EfficientNet_PyTorch_npu.efficientnet_pytorch import *

def test_efficientnet_b0_npu_cp2cpu(cpFile, outCpFile):
    if os.path.isfile(cpFile):
        model = EfficientNet.from_name("efficientnet-b0")
        model,check = cp_load_return_check(model,cpFile,ispth=False,device=None,dno=None)
        save_checkpoint({
            'epoch': check['epoch'],
            'arch': check['arch'],
            'state_dict': model.state_dict(),
            #'best_acc1': check['best_acc1'],
            #'optimizer' : optimizer.state_dict(),
        }, outCpFile)
        logger.info("Save checkpoint \"tar file\":{0}} success!".format(outCpFile))
    else :
        logger.error( 'file(\"{0}\") not exist! '.format(cpFile))
        return None

def test_efficientnet_b0_npu_cp2mpth(outCpFile, outModelMpthFile):
    if os.path.isfile(outCpFile):
        model = EfficientNet.from_name("efficientnet-b0")
        model,check = cp_load_return_check(model,outCpFile,ispth=False,device="cpu",dno=None)

        torch.save(model, outModelMpthFile)
        logger.info("Save model & parameter \"mpth file\":{0} success!".format(outModelMpthFile))
    else :
        logger.error( 'file(\"{0}\") not exist! '.format(outCpFile))
        return None


#################################################################################################
G_CP_efficientnet_NPU_FILEPATH = 'cpfile/pulishCP/efficientnet/efficientnet_cpu.pth.tar'
G_CP_efficientnet_b0_NPU_OUT_FILEPATH = 'cpfile/outCP/efficientnet_cpu_out.pth.tar'
G_MPTH_efficientnet_b0_NPU_OUT_MODEL_FILEPATH = 'cpfile/outModel/efficientnet_cpu_out_mp.mpth'


if G_FLAG_MODLE_Efficientnet_b0:
    if G_CP2CP_CP2CP_FLAG:
        test_efficientnet_b0_npu_cp2cpu(G_CP_efficientnet_NPU_FILEPATH, G_CP_efficientnet_b0_NPU_OUT_FILEPATH)

    if G_CP2CP_CP2MPTH_FLAG:
        test_efficientnet_b0_npu_cp2mpth(G_CP_efficientnet_b0_NPU_OUT_FILEPATH, G_MPTH_efficientnet_b0_NPU_OUT_MODEL_FILEPATH)

    if G_CP2CP_MPTH2ONNX_FLAG:
        G_MPTH2ONNX_efficientnet_b0_NPU_OUT_MODEL_FILEPATH = 'mpth2onnxfile/efficientnet_cpu_out_mp.onnx'

        if os.path.isfile(G_MPTH2ONNX_efficientnet_b0_NPU_OUT_MODEL_FILEPATH):
            input_data =  torch.randn(10, 3, 224, 224)
            model = torch.load(G_MPTH_efficientnet_b0_NPU_OUT_MODEL_FILEPATH)
            model.set_swish(memory_efficient=False)
            model2onnx(model, G_MPTH2ONNX_efficientnet_b0_NPU_OUT_MODEL_FILEPATH, input_data, device=G_DEVICE_STRINT,dno=G_DEVICE_NO_STRINT)
            logger.info("model mpth file export \"onnx file\": \"{0}\" success!".format(G_MPTH2ONNX_efficientnet_b0_NPU_OUT_MODEL_FILEPATH))
        else :
            logger.error( 'file(\"{0}\") not exist! '.format(G_MPTH2ONNX_efficientnet_b0_NPU_OUT_MODEL_FILEPATH))
            


logger.info('===================================================================')
logger.info('shufflenetv2')
logger.info('===================================================================')

#################################################################################################
#test shufflenetv2
from models.shufflenetv2_wock_op_woct import *

def test_shufflenetv2_npu_cp2cpu(cpFile, outCpFile):
    if os.path.isfile(cpFile):
        model = shufflenet_v2_x1_0()
        model,check = cp_load_return_check(model,cpFile,ispth=False,device=None,dno=None)
        save_checkpoint({
            'epoch': check['epoch'],
            'arch': check['arch'],
            'state_dict': model.state_dict(),
            'best_acc1': check['best_acc1'],
            #'optimizer' : optimizer.state_dict(),
        }, outCpFile)
        logger.info("Save checkpoint \"tar file\": {0} success!".format(outCpFile))
    else :
        logger.error( 'file(\"{0}\") not exist! '.format(cpFile))
        return None

def test_shufflenetv2_npu_cp2mpth(outCpFile, outModelMpthFile):
    if os.path.isfile(outCpFile):
        model = shufflenet_v2_x1_0()
        model,check = cp_load_return_check(model,outCpFile,ispth=False,device="cpu",dno=None)

        torch.save(model, outModelMpthFile)
        logger.info("Save model & parameter \"mpth file\": {0} success!".format(outModelMpthFile))
    else :
        logger.error( 'file(\"{0}\") not exist! '.format(outCpFile))
        return None


#################################################################################################
# G_CP_shufflenetv2_NPU_FILEPATH = 'cpfile/pulishCP/shufflenet/model_best_acc69.0409_epoch239.pth.tar'
# G_CP_shufflenetv2_NPU_FILEPATH = 'cpfile/pulishCP/shufflenet/model_best_acc66.5059_epoch240.pth.tar'
G_CP_shufflenetv2_NPU_FILEPATH = 'cpfile/pulishCP/shufflenet/model_best_acc66.5059_epoch240_cpu.pth.tar'
G_CP_shufflenetv2_NPU_OUT_FILEPATH = 'cpfile/outCP/shufflenetv2_model_best_acc66.5059_epoch240_cpu_out.pth.tar'
G_MPTH_shufflenetv2_NPU_OUT_MODEL_FILEPATH = 'cpfile/outModel/shufflenetv2_model_best_acc66.5059_epoch240_cpu_out_mp.mpth'


if G_FLAG_MODLE_shufflenetv2:
    if G_CP2CP_CP2CP_FLAG:
        test_shufflenetv2_npu_cp2cpu(G_CP_shufflenetv2_NPU_FILEPATH, G_CP_shufflenetv2_NPU_OUT_FILEPATH)

    if G_CP2CP_CP2MPTH_FLAG:
        test_shufflenetv2_npu_cp2mpth(G_CP_shufflenetv2_NPU_OUT_FILEPATH, G_MPTH_shufflenetv2_NPU_OUT_MODEL_FILEPATH)

    if G_CP2CP_MPTH2ONNX_FLAG:
        G_MPTH2ONNX_shufflenetv2_NPU_OUT_MODEL_FILEPATH = 'mpth2onnxfile/shufflenetv2_model_best_acc66.5059_epoch240_cpu_out_mp.onnx'
        input_data =  torch.randn(10, 3, 224, 224)
        rtn = mpth2onnx(G_MPTH_shufflenetv2_NPU_OUT_MODEL_FILEPATH,G_MPTH2ONNX_shufflenetv2_NPU_OUT_MODEL_FILEPATH, input_data, device=G_DEVICE_STRINT,dno=G_DEVICE_NO_STRINT)
        if not rtn:
            logger.info("model mpth file export \"onnx file\": \"{0}\" success!".format(G_MPTH2ONNX_shufflenetv2_NPU_OUT_MODEL_FILEPATH))


logger.info('===================================================================')
logger.info('mobilenetv2')
logger.info('===================================================================')
#################################################################################################
#test mobilenetv2
from models.modelzoo_master.MobileNetV2.NPU.p1.mobilenet import *
#from models.modelzoo_master.MobileNetV2.NPU.8p import *

def test_mobilenetv2_npu_cp2cpu(cpFile, outCpFile):
    if os.path.isfile(cpFile):
        model = mobilenet_v2()
        model,check = cp_load_return_check(model,cpFile,ispth=False,device=None,dno=None)
        save_checkpoint({
            'epoch': check['epoch'],
            #'arch': check['arch'],
            'state_dict': model.state_dict(),
            #'best_acc1': check['best_acc1'],
            #'optimizer' : optimizer.state_dict(),
        }, outCpFile)
        logger.info("Save checkpoint \"tar file\": {0} success!".format(outCpFile))
    else :
        logger.error( 'file(\"{0}\") not exist! '.format(cpFile))
        return None


def test_mobilenetv2_npu_cp2mpth(outCpFile, outModelMpthFile):
    if os.path.isfile(outCpFile):
        model = mobilenet_v2()
        model,check = cp_load_return_check(model,outCpFile,ispth=False,device=None,dno=None)

        torch.save(model, outModelMpthFile)
        logger.info("Save model & parameter \"mpth file\": {0} success!".format(outModelMpthFile))
    else :
        logger.error( 'file(\"{0}\") not exist! '.format(outCpFile))
        return None



#G_CP_mobilenetv2_NPU_FILEPATH = 'cpfile/pulishCP/mobilenet/checkpoint.pth.tar
G_CP_mobilenetv2_NPU_FILEPATH = 'cpfile/pulishCP/mobilenet/mobilenet_cpu.pth.tar'
G_CP_mobilenetv2_NPU_OUT_FILEPATH = 'cpfile/outCP/mobilenet_cpu_out.pth.tar'
G_MPTH_mobilenetv2_NPU_OUT_MODEL_FILEPATH = 'cpfile/outModel/mobilenet_cpu_out_mp.mpth'


if G_FLAG_MODLE_mobilenetv2:
    if G_CP2CP_CP2CP_FLAG:
        test_mobilenetv2_npu_cp2cpu(G_CP_mobilenetv2_NPU_FILEPATH, G_CP_mobilenetv2_NPU_OUT_FILEPATH)


    if G_CP2CP_CP2MPTH_FLAG:
        test_mobilenetv2_npu_cp2mpth(G_CP_mobilenetv2_NPU_OUT_FILEPATH,G_MPTH_mobilenetv2_NPU_OUT_MODEL_FILEPATH)

    if G_CP2CP_MPTH2ONNX_FLAG:
        G_MPTH2ONNX_mobilenetv2_NPU_OUT_MODEL_FILEPATH = 'mpth2onnxfile/mobilenet_cpu_out_mp.onnx'
        input_data =  torch.randn(10, 3, 224, 224)
        rtn = mpth2onnx(G_MPTH_mobilenetv2_NPU_OUT_MODEL_FILEPATH,G_MPTH2ONNX_mobilenetv2_NPU_OUT_MODEL_FILEPATH, input_data, device=G_DEVICE_STRINT,dno=G_DEVICE_NO_STRINT)
        if not rtn:
            logger.info("model mpth file export \"onnx file\": \"{0}\" success!".format(G_MPTH2ONNX_mobilenetv2_NPU_OUT_MODEL_FILEPATH))


logger.info('===================================================================')
logger.info('DeepMAR')
logger.info('===================================================================')


#################################################################################################
#test DeepMAR
from models.deepmar.pedestrian_attribute_recognition_pytorch.baseline.model import DeepMAR

def test_DeepMAR_gpu_cp2cpu(cpFile, outCpFile):
    if os.path.isfile(cpFile):
        model =  DeepMAR.DeepMAR_ResNet50()
        model,check = cp_load_return_check(model,cpFile,ispth=False,device=None,dno=None)
        save_checkpoint({
            'epoch': check['epoch'],
            #'arch': check['arch'],
            'state_dict': model.state_dict(),
            #'best_acc1': check['best_acc1'],
            #'optimizer' : optimizer.state_dict(),
        }, outCpFile)
        logger.info("Save checkpoint \"tar file\": {0} success!".format(outCpFile))
    else :
        logger.error( 'file(\"{0}\") not exist! '.format(cpFile))
        return None


def test_DeepMAR_gpu_cp2mpth(outCpFile, outModelMpthFile):
    if os.path.isfile(outCpFile):
        model =  DeepMAR.DeepMAR_ResNet50()
        model,check = cp_load_return_check(model,outCpFile,ispth=False,device=None,dno=None)

        torch.save(model, outModelMpthFile)
        logger.info("Save model & parameter \"mpth file\": {0} success!".format(outModelMpthFile))
    else :
        logger.error( 'file(\"{0}\") not exist! '.format(outCpFile))
        return None
#################################################################################################

#G_CP_DeepMAR_GPU_FILEPATH = 'cpfile/pulishCP/deepmar/checkpoint.pth.tar'
G_CP_DeepMAR_GPU_FILEPATH = 'cpfile/pulishCP/deepmar/deepmar_cpu.pth.tar'
G_CP_DeepMAR_GPU_OUT_FILEPATH = 'cpfile/outCP/deepmar_cpu_out.pth.tar'
G_MPTH_DeepMAR_GPU_OUT_MODEL_FILEPATH = 'cpfile/outModel/deepmar_cpu_out_mp.mpth'



if G_FLAG_MODLE_DeepMAR:
    if G_CP2CP_CP2CP_FLAG:
        test_DeepMAR_gpu_cp2cpu(G_CP_DeepMAR_GPU_FILEPATH, G_CP_DeepMAR_GPU_OUT_FILEPATH)

    if G_CP2CP_CP2MPTH_FLAG:
        test_DeepMAR_gpu_cp2mpth(G_CP_DeepMAR_GPU_OUT_FILEPATH, G_MPTH_DeepMAR_GPU_OUT_MODEL_FILEPATH)

    if G_CP2CP_MPTH2ONNX_FLAG:
        G_MPTH2ONNX_DeepMAR_GPU_OUT_MODEL_FILEPATH = 'mpth2onnxfile/deepmar_cpu_out_mp.onnx'
        input_data =  torch.randn(10, 3, 224, 224)
        rtn = mpth2onnx(G_MPTH_DeepMAR_GPU_OUT_MODEL_FILEPATH,G_MPTH2ONNX_DeepMAR_GPU_OUT_MODEL_FILEPATH, input_data, device=G_DEVICE_STRINT,dno=G_DEVICE_NO_STRINT)
        if not rtn:
            logger.info("model mpth file export \"onnx file\": \"{0}\" success!".format(G_MPTH2ONNX_DeepMAR_GPU_OUT_MODEL_FILEPATH))


