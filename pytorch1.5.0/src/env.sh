# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
############## nnae situation ################

# cann env path
cpu_type=$(echo $HOSTTYPE)
if [ x"${cpu_type}" == x"x86_64" ];then
  cpu_type=x86_64-linux
else
  cpu_type=arm64-linux
fi

# 根据系统下的路径判断走nnae 还是 ascend-toolkit
if [ -d /usr/local/Ascend/ascend-toolkit/latest ];then
  export ASCEND_BASE=/usr/local/Ascend/ascend-toolkit/latest
  export ASCEND_AICPU_PATH=${ASCEND_BASE}/${cpu_type}
else
  export ASCEND_BASE=/usr/local/Ascend/nnae/latest
  export ASCEND_AICPU_PATH=${ASCEND_BASE}
fi

# 定义各个子包的路径
export FWK_HOME=${ASCEND_BASE}/fwkacllib
export PLUGIN_PATH=${ASCEND_BASE}/fwkacllib/lib64/plugin
export OP_PATH=${ASCEND_BASE}/opp/
export TOOLKIT_PATH=${ASCEND_BASE}/toolkit
export DRIVER_PATH=/usr/local/Ascend/driver
export ADD_ONS_PATH=/usr/local/Ascend/add-ons

# 对于下面的路径， 如果子包内的相对路径不变则不要修改
# 定义导入的依赖库
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/python3.7.5/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/openblas/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib64/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${FWK_HOME}/lib64/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${DRIVER_PATH}/lib64/common/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${DRIVER_PATH}/lib64/driver/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ADD_ONS_PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/aarch64_64-linux-gnu

# 定义python的依赖库
export PYTHONPATH=$PYTHONPATH:${FWK_HOME}/python/site-packages/
export PYTHONPATH=$PYTHONPATH:${FWK_HOME}/python/site-packages/auto_tune.egg/auto_tune
export PYTHONPATH=$PYTHONPATH:${FWK_HOME}/python/site-packages/schedule_search.egg

# 定义plugin依赖库
export OPTION_EXEC_EXTERN_PLUGIN_PATH=$OPTION_EXEC_EXTERN_PLUGIN_PATH:${PLUGIN_PATH}/opskernel/libfe.so
export OPTION_EXEC_EXTERN_PLUGIN_PATH=$OPTION_EXEC_EXTERN_PLUGIN_PATH:${PLUGIN_PATH}/opskernel/libaicpu_engine.so
export OPTION_EXEC_EXTERN_PLUGIN_PATH=$OPTION_EXEC_EXTERN_PLUGIN_PATH:${PLUGIN_PATH}/opskernel/libge_local_engine.so

# 训练需要在线编译算子故而需要确认cce编译器以及算子的路径
export ASCEND_OPP_PATH=${OP_PATH}
export PATH=$PATH:${FWK_HOME}/ccec_compiler/bin/
export PATH=$PATH:${TOOLKIT_PATH}/tools/ide_daemon/bin/
export HCCL_WHITELIST_DISABLE=1

# 搜索当前执行python库路径及三方库路径, 导入到LD_LIBRARY_PATH
path_lib=$(python3.7 -c """
import sys
import re
result=''
for index in range(len(sys.path)):
    match_sit = re.search('-packages', sys.path[index])
    if match_sit is not None:
        match_lib = re.search('lib', sys.path[index])
        match_run = re.search('Ascend', sys.path[index])
        if match_run is not None:
            continue

        if match_lib is not None:
            end=match_lib.span()[1]
            result += sys.path[index][0:end] + ':'
        
        result += sys.path[index] + '/torch/:'
        result += sys.path[index] + '/torch/lib:'
print(result)"""
)
echo ${path_lib}
export LD_LIBRARY_PATH=${path_lib}:$LD_LIBRARY_PATH

# pytorch 自定义环境变量
export TASK_QUEUE_ENABLE=0 # 使用异步任务下发，异步调用acl接口，建议默认开启，开启设置为1
export PTCOPY_ENABLE=1 # 使用PTCopy算子模式，加速转连续及copy等过程，建议默认开启，开启设置为1
#export DYNAMIC_COMPILE_ENABLE=1  # 动态shape特性功能，针对shape变化场景，可选 开启设置为1

# log 
export ASCEND_SLOG_PRINT_TO_STDOUT=0   #日志打屏, 可选
export ASCEND_GLOBAL_LOG_LEVEL=3       #日志级别常用 1 INFO级别; 3 ERROR级别
export ASCEND_GLOBAL_EVENT_ENABLE=0    #默认不使能event日志信息