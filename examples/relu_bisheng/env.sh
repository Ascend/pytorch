CANN_INSTALL_PATH_CONF='/etc/Ascend/ascend_cann_install.info'
if [ -f $CANN_INSTALL_PATH_CONF ]; then
  DEFAULT_CANN_INSTALL_PATH=$(cat $CANN_INSTALL_PATH_CONF | grep Install_Path | cut -d "=" -f 2)
else
  DEFAULT_CANN_INSTALL_PATH="/usr/local/Ascend/"
fi
CANN_INSTALL_PATH=${1:-${DEFAULT_CANN_INSTALL_PATH}}
echo "CANN_INSTALL_PATH=$CANN_INSTALL_PATH"
if [ -d ${CANN_INSTALL_PATH}/ascend-toolkit/latest ];then
  source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
else
  source ${CANN_INSTALL_PATH}/nnae/set_env.sh
fi

export TASK_QUEUE_ENABLE=0

if [ -z $BSCPP_ROOT ];then
  BSCPP_ROOT=/opt/BiShengCPP
fi
export LD_LIBRARY_PATH=${BSCPP_ROOT}/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
