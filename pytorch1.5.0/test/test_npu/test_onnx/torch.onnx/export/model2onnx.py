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

from export.onnx_parser import *
from export.cp_parser import *
from export.export_onnx import *
from eval.onnxrt.onnxrt_eval import *
from eval.onnx.cp_onnx_eval import *
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

logger = logging.getLogger('torch.onnx.export.test.model2onnx')
logger.addHandler(loggerScreanHander)
logger.setLevel(logger_level_relations.get('info'))
logger.debug('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))

#################################################################################################
#test resnet50
G_CP_RESNET50_NPU_FILEPATH = 'cpfile/checkpoint-resnet50-npu-1p-benchmark0-bs512-valshuffle-false-epoch90-120-epoch119.pth.tar'
#G_CP_RESNET50_NPU_FILEPATH = 'cpfile/outCP/resnet_cpu20200928_1800_out.pth.tar'

#G_ONNX_RESNET50_NPU_FILEPATH = 'onnxfile/resnet50_npu.onnx'
G_ONNX_RESNET50_NPU_FILEPATH = 'onnxfile/resnet_cpu20200928_1800_out_npu.onnx'

def test_resnet50_npu(deviceStr, devNo):
    deviceId = cp_getDeviceId(deviceStr, devNo)
    logger.debug("device: {0}( {1}  {2} )".format(deviceId,deviceStr,devNo))
    input_data = torch.randn(10, 3, 224, 224)
    model = torchvision.models.resnet50(pretrained=False)
    cp2onnx(model,G_CP_RESNET50_NPU_FILEPATH,G_ONNX_RESNET50_NPU_FILEPATH, input_data.to(deviceId),device=deviceStr,dno=devNo)
    onnx_runeval(model,G_CP_RESNET50_NPU_FILEPATH,G_ONNX_RESNET50_NPU_FILEPATH)
    #onnxrt_runeval_resnet50_pic(G_CP_RESNET50_NPU_FILEPATH,G_ONNX_RESNET50_NPU_FILEPATH,dimx=10)
    onnxrt_runeval_resnet50_cmp(G_CP_RESNET50_NPU_FILEPATH,G_ONNX_RESNET50_NPU_FILEPATH, dimx=10)



#################################################################################################

#test resnet50v2
#G_ONNX_resnet50v2_FILEPATH = 'resnet50v2/resnet50v2.onnx'
#onnxrt_runeval_resnet50_pic(G_CP_RESNET50_FILEPATH,G_ONNX_resnet50v2_FILEPATH,dimx=1)

#################################################################################################
#test densenet121

from models.densnet121.densenet_0_2_2 import *

G_CP_DENSENET121_NPU_FILEPATH = 'cpfile/outCP/densenet121_cpu_out.pth.tar'
G_ONNX_DENSENET121_NPU_FILEPATH = 'onnxfile/densenet121_cpu_out.onnx'

# G_CP_DENSENET121_FILEPATH = 'cpfile/densenet121_gpu.pth.tar'
# G_ONNX_DENSENET121_FILEPATH = 'onnxfile/densenet121_gpu.onnx'

def test_densenet121_npu(deviceStr, devNo):
    deviceId = cp_getDeviceId(deviceStr, devNo)
    logger.debug("device: {0}( {1}  {2} )".format(deviceId,deviceStr,devNo))
    input_data = torch.randn(10, 3, 224, 224)
    model = densenet121(pretrained=False)
    cp2onnx(model,G_CP_DENSENET121_NPU_FILEPATH,G_ONNX_DENSENET121_NPU_FILEPATH,  input_data.to(deviceId), device=deviceStr,dno=devNo)
    if 0:
        cp2onnx(model,G_CP_DENSENET121_NPU_FILEPATH,G_ONNX_DENSENET121_NPU_FILEPATH, input_data, device="cpu",dno=None)
        cp2onnx(torchvision.models.densenet121(pretrained=False),G_CP_DENSENET121_NPU_FILEPATH,G_ONNX_DENSENET121_NPU_FILEPATH,device="cuda",dno=None)
    
    onnx_runeval(model,G_CP_DENSENET121_NPU_FILEPATH,G_ONNX_DENSENET121_NPU_FILEPATH)
    
    #model = densenet121(pretrained=False)
    densenet121_eval(model.to(deviceId),G_CP_DENSENET121_NPU_FILEPATH,input_data.to(deviceId))
    onnxrt_densnet121_eval(G_ONNX_DENSENET121_NPU_FILEPATH,input_data.to('cpu'),dimx=1)
    
    
#################################################################################################
#test efficientnet-b0
from models.efficientnet_pytorch.EfficientNet_PyTorch_npu.efficientnet_pytorch import *
G_CP_efficientnet_NPU_FILEPATH = 'cpfile/outCP/efficientnet_cpu_out.pth.tar'
G_ONNX_efficientnet_NPU_FILEPATH = 'onnxfile/efficientnet_cpu_out.onnx'
G_ONNX_efficientnet_NPU_dynamic_axex_FILEPATH = 'onnxfile/efficientnet_cpu_out_npu_dynamic_axex.onnx'

# G_CP_efficientnet_FILEPATH = 'cpfile/efficientnet_npu_20200908_1746.pth.tar'
# G_ONNX_efficientnet_FILEPATH = 'onnxfile/efficientnet_npu_20200908_1746.onnx'

# from models.efficientnet_pytorch.EfficientNet_PyTorch_gpu.efficientnet_pytorch import *
# G_CP_efficientnet_FILEPATH = 'cpfile/efficientnet_gpu_20200908.pth.tar'
# G_ONNX_efficientnet_FILEPATH = 'onnxfile/efficientnet_gpu_20200908.onnx'

def test_efficientnet_npu(deviceStr, devNo):
    deviceId = cp_getDeviceId(deviceStr, devNo)
    logger.debug("device: {0}( {1}  {2} )".format(deviceId,deviceStr,devNo))
    model = EfficientNet.from_name("efficientnet-b0")
    input_data = torch.randn(10, 3, 224, 224)
    model.set_swish(memory_efficient=False)
    cp2onnx(model,G_CP_efficientnet_NPU_FILEPATH,G_ONNX_efficientnet_NPU_FILEPATH, input_data.to(deviceId), device=deviceStr,dno=devNo)
    #cp2onnx(model,G_CP_efficientnet_NPU_FILEPATH,G_ONNX_efficientnet_NPU_FILEPATH,device="cpu",dno=None)
    #cp2onnx_dynamic_axes(model,G_CP_efficientnet_NPU_FILEPATH,G_ONNX_efficientnet_NPU_dynamic_axex_FILEPATH,device="cpu",dno=None)
    onnx_runeval(model,G_CP_efficientnet_NPU_FILEPATH,G_ONNX_efficientnet_NPU_FILEPATH)
    logger.debug(deviceId)
    efficientnet_eval(model.to(deviceId),G_CP_efficientnet_NPU_FILEPATH,input_data.to(deviceId))
    onnxrt_efficientnet_eval(G_ONNX_efficientnet_NPU_FILEPATH,input_data.to('cpu'))

from models.efficientnet_pytorch.EfficientNet_PyTorch_gpu.efficientnet_pytorch import *
G_CP_efficientnet_GPU_FILEPATH = 'models/efficientnet_pytorch/EfficientNet_PyTorch_gpu/checkpoint.pth.tar'
G_ONNX_efficientnet_GPU_FILEPATH = 'onnxfile/efficientnet_gpu.onnx'
G_ONNX_efficientnet_GPU_dynamic_axex_FILEPATH = 'onnxfile/efficientnet_gpu_dynamic_axex.onnx'

# from models.efficientnet_pytorch.EfficientNet_PyTorch_gpu.efficientnet_pytorch import *
# G_CP_efficientnet_FILEPATH = 'cpfile/efficientnet_gpu_20200908.pth.tar'
# G_ONNX_efficientnet_FILEPATH = 'onnxfile/efficientnet_gpu_20200908.onnx'

def test_efficientnet_gpu(deviceStr, devNo):
    deviceId = cp_getDeviceId(deviceStr, devNo)
    logger.debug("device: {0}( {1}  {2} )".format(deviceId,deviceStr,devNo))
    input_data = torch.randn(10, 3, 224, 224)
    model = EfficientNet.from_name("efficientnet-b0")
    model.set_swish(memory_efficient=False)
    cp2onnx(model,G_CP_efficientnet_GPU_FILEPATH,G_ONNX_efficientnet_GPU_FILEPATH,input_data.to(deviceId),device=deviceStr,dno=devNo)
    #cp2onnx_dynamic_axes(model,G_CP_efficientnet_GPU_FILEPATH,G_ONNX_efficientnet_NPU_dynamic_axex_FILEPATH,device="cpu",dno=None)
    onnx_runeval(model,G_CP_efficientnet_GPU_FILEPATH,G_ONNX_efficientnet_GPU_FILEPATH)
    model = EfficientNet.from_name("efficientnet-b0")
    efficientnet_eval(model.to(deviceId),G_CP_efficientnet_GPU_FILEPATH,input_data.to(deviceId))
    onnxrt_efficientnet_eval(G_ONNX_efficientnet_GPU_FILEPATH,input_data.to('cpu'))


#################################################################################################
#test shufflenetv2


#import sys
#sys.path.append('/home/qbin/torch.onnx/models/modelzoo_master/ShuffleNetV2/GPU/megvii/')
#import torchvision.models as tvmodels



from models.shufflenetv2_wock_op_woct import *



G_CP_shufflenetv2_NPU_FILEPATH = 'cpfile/outCP/shufflenetv2_model_best_acc66.5059_epoch240_cpu_out.pth.tar'
G_ONNX_shufflenetv2_NPU_FILEPATH = 'onnxfile/shufflenetv2_model_best_acc66.5059_epoch240_cpu_out_npu.onnx'

def test_shufflenetv2_npu(deviceStr, devNo):
    deviceId = cp_getDeviceId(deviceStr, devNo)
    logger.debug("device: {0}( {1}  {2} )".format(deviceId,deviceStr,devNo))
    model = shufflenet_v2_x1_0()
    #model = tvmodels.__dict__['shufflenet_v2_x1_0']()
    input_data = torch.randn(10, 3, 224, 224)
    cp2onnx(model,G_CP_shufflenetv2_NPU_FILEPATH,G_ONNX_shufflenetv2_NPU_FILEPATH, input_data, device=deviceStr,dno=devNo)
    
    onnx_runeval(model,G_CP_shufflenetv2_NPU_FILEPATH,G_ONNX_shufflenetv2_NPU_FILEPATH)
    
    shufflenet_eval(model.to(deviceId),G_CP_shufflenetv2_NPU_FILEPATH,input_data.to(deviceId),ispth=True)
    onnxrt_shufflenet_eval(G_ONNX_shufflenetv2_NPU_FILEPATH,input_data.to('cpu'))

#from models.modelzoo_master.ShuffleNetV2.GPU.megvii.network import *
#from models.modelzoo_master.ShuffleNetV2.GPU.pytorch.network import *
import sys
#sys.path.append('/home/qbin/torch.onnx/models/modelzoo_master/ShuffleNetV2/GPU/megvii/')
import torchvision.models as tvmodels

G_CP_shufflenetv2_GPU_FILEPATH = 'cpfile/shufflenetv2_x1-5666bf0f80-gpu.pth'
G_ONNX_shufflenetv2_GPU_FILEPATH = 'onnxfile/shufflenetv2_x1-5666bf0f80-gpu.pth.onnx'
def test_shufflenetv2_gpu(deviceStr, devNo):
    deviceId = cp_getDeviceId(deviceStr, devNo)
    logger.debug("device: {0}( {1}  {2} )".format(deviceId,deviceStr,devNo))
    model = tvmodels.__dict__['shufflenet_v2_x1_0']()
    input_data = torch.randn(10, 3, 224, 224)
    cp2onnx(model,G_CP_shufflenetv2_GPU_FILEPATH,G_ONNX_shufflenetv2_GPU_FILEPATH, input_data,ispth=True,device="cpu",dno=None)
    
    onnx_runeval(model,G_CP_shufflenetv2_GPU_FILEPATH,G_ONNX_shufflenetv2_GPU_FILEPATH,ispth=True)

    input_data = torch.randn(10, 3, 224, 224)
    shufflenet_eval(model,G_CP_shufflenetv2_GPU_FILEPATH,input_data,ispth=True)
    onnxrt_shufflenet_eval(G_ONNX_shufflenetv2_GPU_FILEPATH,input_data)


#################################################################################################
#test mobilenetv2

G_CP_mobilenetv2_NPU_FILEPATH = 'cpfile/outCP/mobilenet_cpu_out.pth.tar'
G_ONNX_mobilenetv2_NPU_FILEPATH = 'onnxfile/mobilenet_cpu_out.npu.onnx'


from models.modelzoo_master.MobileNetV2.NPU.p1.mobilenet import *
#from models.modelzoo_master.MobileNetV2.NPU.8p import *
def test_mobilenetv2_npu(deviceStr, devNo):
    deviceId = cp_getDeviceId(deviceStr, devNo)
    logger.debug("device: {0}( {1}  {2} )".format(deviceId,deviceStr,devNo))
    model = mobilenet_v2()
    input_data = torch.randn(1, 3, 224, 224)
    cp2onnx(model,G_CP_mobilenetv2_NPU_FILEPATH,G_ONNX_mobilenetv2_NPU_FILEPATH, input_data, device=deviceStr, dno=devNo)
    #cp2onnx(model,G_CP_mobilenetv2_NPU_FILEPATH,G_ONNX_mobilenetv2_NPU_FILEPATH, input_data, device="cpu", dno=None)
    
    onnx_runeval(model,G_CP_mobilenetv2_NPU_FILEPATH,G_ONNX_mobilenetv2_NPU_FILEPATH)

    mobilenetv2_eval(model.to(deviceId),G_CP_mobilenetv2_NPU_FILEPATH,input_data.to(deviceId))
    onnxrt_mobilenetv2_eval(G_ONNX_mobilenetv2_NPU_FILEPATH,input_data.to('cpu'))


G_CP_mobilenetv2_GPU_FILEPATH = 'cpfile/mobilenetv2_gpu_20200909.pth.tar'
G_ONNX_mobilenetv2_GPU_FILEPATH = 'onnxfile/mobilenetv2_gpu_20200909.onnx'

from models.modelzoo_master.MobileNetV2.GPU.mobilenet import *
def test_mobilenetv2_gpu(deviceStr, devNo):
    deviceId = cp_getDeviceId(deviceStr, devNo)
    logger.debug("device: {0}( {1}  {2} )".format(deviceId,deviceStr,devNo))
    model = mobilenet_v2()
    
    cp2onnx(model,G_CP_mobilenetv2_GPU_FILEPATH,G_ONNX_mobilenetv2_GPU_FILEPATH, input_data, device="cpu",dno=None)
    
    #onnx_runeval(model,G_CP_mobilenetv2_GPU_FILEPATH,G_ONNX_mobilenetv2_GPU_FILEPATH)

    mobilenetv2_eval(model,G_CP_mobilenetv2_GPU_FILEPATH,input_data)
    onnxrt_mobilenetv2_eval(G_ONNX_mobilenetv2_GPU_FILEPATH,input_data)

#################################################################################################
#test DeepMAR
from models.deepmar.pedestrian_attribute_recognition_pytorch.baseline.model import DeepMAR

G_CP_DeepMAR_NPU_FILEPATH = 'cpfile/outCP/deepmar_cpu_out.pth.tar'
G_ONNX_DeepMAR_NPU_FILEPATH = 'onnxfile/deepmar_cpu_out_npu.onnx'

def test_DeepMAR_npu(deviceStr, devNo):
    deviceId = cp_getDeviceId(deviceStr, devNo)
    logger.debug("device: {0}( {1}  {2} )".format(deviceId,deviceStr,devNo))
    model = DeepMAR.DeepMAR_ResNet50()
    input_data = torch.randn(10, 3, 224, 224)
    cp2onnx(model,G_CP_DeepMAR_NPU_FILEPATH,G_ONNX_DeepMAR_NPU_FILEPATH,input_data.to(deviceId),device=deviceStr,dno=devNo)
    onnx_runeval(model,G_CP_DeepMAR_NPU_FILEPATH,G_ONNX_DeepMAR_NPU_FILEPATH)
    
    #model = DeepMAR.DeepMAR_ResNet50()
    deepmar_resnet50_eval(model.to(deviceId),G_CP_DeepMAR_NPU_FILEPATH,input_data.to(deviceId))
    onnxrt_deepmar_resnet50_eval(G_ONNX_DeepMAR_NPU_FILEPATH,input_data.to('cpu'))

# G_CP_DeepMAR_GPU_FILEPATH = 'cpfile/Null.pth.tar'
# G_ONNX_DeepMAR_GPU_FILEPATH = 'onnxfile/DeepMAR_gpu.onnx'
G_CP_DeepMAR_GPU_FILEPATH = 'cpfile/outCP/deepmar_cpu_out.pth.tar'
G_ONNX_DeepMAR_GPU_FILEPATH = 'onnxfile/deepmar_cpu_out_npu.onnx'

def test_DeepMAR_gpu(deviceStr, devNo):
    deviceId = cp_getDeviceId(deviceStr, devNo)
    logger.debug("device: {0}( {1}  {2} )".format(deviceId,deviceStr,devNo))
    model = DeepMAR.DeepMAR_ResNet50()
    input_data = torch.randn(10, 3, 224, 224)
    cp2onnx(model,G_CP_DeepMAR_GPU_FILEPATH,G_ONNX_DeepMAR_GPU_FILEPATH,input_data.to(deviceId),device="cuda",dno=None)
    
    model = DeepMAR.DeepMAR_ResNet50()
    onnx_runeval(model,G_CP_DeepMAR_GPU_FILEPATH,G_ONNX_DeepMAR_GPU_FILEPATH)
    model = DeepMAR.DeepMAR_ResNet50()
    deepmar_resnet50_eval(model,G_CP_DeepMAR_GPU_FILEPATH,input_data.to(deviceId))
    onnxrt_deepmar_resnet50_eval(G_ONNX_DeepMAR_GPU_FILEPATH,input_data.to('cpu'))
#################################################################################################
#test transformer 
'''
import sys
sys.path.append('/home/qbin/torch.onnx/models/transformer/transformer_gpu/')

#from models.transformer.transformer_gpu.train import build_model,load_checkpoint
#from models.transformer.transformer_gpu.utils.ddp_trainer import DDPTrainer
from models.transformer.transformer_gpu.transformer_init import *
G_CP_transformer_GPU_FILEPATH = 'cpfile/transformer_gpu.pth.tar'
G_ONNX_transformer_GPU_FILEPATH = 'onnxfile/transformer_gpu.onnx'

logger.info("===================================================================")
parser = options.get_training_parser()
logger.info("===================================================================")
DATA_DIR='../wmt14_en_de_joined_dict'
MODELDIR="../../../cpfile/"
STAT_FILE="log.json"

input_args=  '"' + DATA_DIR + '"' \
  ' --arch transformer_wmt_en_de \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas \'(0.9, 0.997)\' \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000 \
  --lr 0.0006 \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-sentences 128\
  --max-tokens 102400\
  --seed 1 \
  --fuse-layer-norm \
  --save-dir ' + MODELDIR + \
  ' --save-interval 1\
  --online-eval \
  --stat-file ' + STAT_FILE + \
  ' --train-subset train\
  --update-freq 8 \
  --do-sanity-check'

input_args= [DATA_DIR , \
  '--arch' , 'transformer_wmt_en_de' , \
  '--share-all-embeddings' , \
  '--optimizer' , 'adam' , \
  '--adam-betas' , '(0.9, 0.997)' , \
  '--adam-eps' , '1e-9' , \
  '--clip-norm' , '0.0 ' , \
  '--lr-scheduler' , 'inverse_sqrt' , \
  '--warmup-init-lr' , '0.0' , \
  '--warmup-updates' , '4000' , \
  '--lr' , '0.0006' , \
  '--min-lr' , '0.0' , \
  '--dropout' , '0.1' , \
  '--weight-decay' , '0.0' , \
  '--criterion' , 'label_smoothed_cross_entropy' , \
  '--label-smoothing' , '0.1' , \
  '--max-sentences' , '128' , \
  '--max-tokens' , '102400' , \
  '--seed' , '1' , \
  '--fuse-layer-norm' , \
  '--save-dir' , MODELDIR , \
  '--save-interval' , '1' , \
  '--online-eval' , \
  '--stat-file' , STAT_FILE , \
  '--train-subset' , 'train' , \
  '--update-freq' , '8' , \
  '--do-sanity-check']
logger.info(input_args)
logger.info("===================================================================")
ARGS = options.parse_args_and_arch(parser,input_args=input_args)
logger.info("===================================================================")
logger.info(ARGS)

def test_transformer_gpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    model = transformer_init(ARGS)
    trainer = DDPTrainer(args, model)
    load_checkpoint(args, trainer, epoch_itr)
    
    cp2onnx(model,G_CP_transformer_GPU_FILEPATH,G_ONNX_transformer_GPU_FILEPATH,device="cuda",dno=None)
    
    onnx_runeval(model,G_CP_transformer_GPU_FILEPATH,G_ONNX_transformer_GPU_FILEPATH)
'''
#################################################################################################

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



G_DEVICE_STRINT = "cpu"
G_DEVICE_NO_STRINT = None

TEST_RESNET50_NPU_FLAG = 1
TEST_densenet121_NPU_FLAG = 1
TEST_efficientnetb0_NPU_FLAG = 1
TEST_efficientnetb0_GPU_FLAG = 0
TEST_shufflenetv2_NPU_FLAG = 0
TEST_shufflenetv2_GPU_FLAG = 1
TEST_mobilenetv2_NPU_FLAG = 1
TEST_mobilenetv2_GPU_FLAG = 0
TEST_DeepMAR_NPU_FLAG = 1
TEST_DeepMAR_GPU_FLAG = 0
TEST_transformer_GPU_FLAG = 0

logger.info('==============================================================================================')
logger.info('resnet50')
logger.info('==============================================================================================')
if TEST_RESNET50_NPU_FLAG :
    test_resnet50_npu(G_DEVICE_STRINT,G_DEVICE_NO_STRINT)

logger.info('==============================================================================================')
logger.info('densenet121')
logger.info('==============================================================================================')
if TEST_densenet121_NPU_FLAG :
    test_densenet121_npu(G_DEVICE_STRINT,G_DEVICE_NO_STRINT)

logger.info('==============================================================================================')
logger.info('efficientnet-b0')
logger.info('==============================================================================================')
if TEST_efficientnetb0_NPU_FLAG :
    test_efficientnet_npu(G_DEVICE_STRINT,G_DEVICE_NO_STRINT)
if TEST_efficientnetb0_GPU_FLAG :
 test_efficientnet_gpu()

logger.info('==============================================================================================')
logger.info('shufflenetv2')
logger.info('==============================================================================================')
if TEST_shufflenetv2_NPU_FLAG:
    test_shufflenetv2_npu("cpu",None)
    test_shufflenetv2_npu(G_DEVICE_STRINT,G_DEVICE_NO_STRINT)
if TEST_shufflenetv2_GPU_FLAG :
    test_shufflenetv2_gpu(G_DEVICE_STRINT,G_DEVICE_NO_STRINT)


logger.info('==============================================================================================')
logger.info('mobilenetv2')
logger.info('==============================================================================================')
if TEST_mobilenetv2_NPU_FLAG:
    test_mobilenetv2_npu(G_DEVICE_STRINT,G_DEVICE_NO_STRINT)
if TEST_mobilenetv2_GPU_FLAG:
    test_mobilenetv2_gpu()

logger.info('==============================================================================================')
logger.info('DeepMAR')
logger.info('==============================================================================================')
if TEST_DeepMAR_NPU_FLAG:
    test_DeepMAR_npu(G_DEVICE_STRINT,G_DEVICE_NO_STRINT)
if TEST_DeepMAR_GPU_FLAG:
    test_DeepMAR_gpu(G_DEVICE_STRINT,G_DEVICE_NO_STRINT)


logger.info('==============================================================================================')
logger.info('transformer')
logger.info('==============================================================================================')
if TEST_transformer_GPU_FLAG:
    test_transformer_gpu()

