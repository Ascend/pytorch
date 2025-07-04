#!/bin/bash
# 配置CANN相关环境变量
CANN_INSTALL_PATH_CONF='/etc/Ascend/ascend_cann_install.info'

if [ -f $CANN_INSTALL_PATH_CONF ]; then
  DEFAULT_CANN_INSTALL_PATH=$(cat $CANN_INSTALL_PATH_CONF | grep Install_Path | cut -d "=" -f 2)
else
  DEFAULT_CANN_INSTALL_PATH="/usr/local/Ascend/"
fi

CANN_INSTALL_PATH=${1:-${DEFAULT_CANN_INSTALL_PATH}}

if [ -d ${CANN_INSTALL_PATH}/ascend-toolkit/latest ];then
  source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
else
  source ${CANN_INSTALL_PATH}/nnae/set_env.sh
fi

# 导入依赖库
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/openblas/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib64/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/aarch64_64-linux-gnu

# 配置自定义环境变量
export HCCL_WHITELIST_DISABLE=1

#export COMBINED_ENABLE=1 # 非连续两个算子组合类场景优化，可选，开启设置为1
#export ACL_DUMP_DATA=1 # 算子数据dump功能，调试时使用，可选，开启设置为1

# log
export ASCEND_SLOG_PRINT_TO_STDOUT=0   #日志打屏, 可选
export ASCEND_GLOBAL_LOG_LEVEL=3       #日志级别常用 1 INFO级别; 3 ERROR级别
export ASCEND_GLOBAL_EVENT_ENABLE=0    #默认不使能event日志信息