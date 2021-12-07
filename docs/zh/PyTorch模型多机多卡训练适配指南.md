# 概述

该文档帮助用户快速实现Pytorch单机多卡模型在集群上使用DDP模式训练。

# 适配流程

1. 准备环境。

   准备模型多机多卡训练的软件、硬件、网络环境，包括开发和运行环境搭建、集群组网链接、芯片IP设置、防火墙设置等。

2. 准备单机多卡模型。

   该文档帮助用户实现单机多卡训练模型到多机多卡训练的适配，故用户需提前准备单机多卡模型，可以从开源社区获取，也可以自行实现。

3. 修改模型脚本。

   修改单机多卡模型启动脚本和环境变量，使模型支持在多机多卡训练。

4. 启动训练。

   实现在多机多卡环境启动模型训练并查看训练日志。

   

# 快速上手

## 概述

使用resnet50模型实现两台计算机8卡训练。两台计算机分别命名为AI Server0、AI Server1，每台计算机上的8个Ascend 910处理器命名为device 0~7。

## 环境准备

首先您需要具有至少两台装有Ascend 910处理器的计算机，并保证每台计算机都正确的安装NPU固件和驱动。

1. 准备开发和运行环境。

   在每台计算机上完成下面开发和运行环境准备。

   - 完成CANN开发和运行环境的安装，请参见《CANN 软件安装指南》。

   - 安装适配NPU的PyTorch，安装方法请参见《PyTorch安装指南》。

2. 准备组网。

   通过交换机或光口直连的方式完成计算设备组网搭建，搭建方法请参见《[数据中心训练场景组网](https://support.huawei.com/enterprise/zh/doc/EDOC1100221993/229cc0e4)》。

   该示例中采用两台计算机8卡进行训练，故可以采用光口直连的方式准备组网。

3. 配置device IP

   在AI Server0上配置device IP。

   ```shell
   hccn_tool -i 0 -ip -s address 192.168.100.101 netmask 255.255.255.0
   hccn_tool -i 1 -ip -s address 192.168.101.101 netmask 255.255.255.0
   hccn_tool -i 2 -ip -s address 192.168.102.101 netmask 255.255.255.0
   hccn_tool -i 3 -ip -s address 192.168.103.101 netmask 255.255.255.0
   hccn_tool -i 4 -ip -s address 192.168.100.100 netmask 255.255.255.0
   hccn_tool -i 5 -ip -s address 192.168.101.100 netmask 255.255.255.0
   hccn_tool -i 6 -ip -s address 192.168.102.100 netmask 255.255.255.0
   hccn_tool -i 7 -ip -s address 192.168.103.100 netmask 255.255.255.0
   ```

   在AI Server1上配置device IP。

   ```shell
   hccn_tool -i 0 -ip -s address 192.168.100.111 netmask 255.255.255.0
   hccn_tool -i 1 -ip -s address 192.168.101.111 netmask 255.255.255.0
   hccn_tool -i 2 -ip -s address 192.168.102.111 netmask 255.255.255.0
   hccn_tool -i 3 -ip -s address 192.168.103.111 netmask 255.255.255.0
   hccn_tool -i 4 -ip -s address 192.168.100.110 netmask 255.255.255.0
   hccn_tool -i 5 -ip -s address 192.168.101.110 netmask 255.255.255.0
   hccn_tool -i 6 -ip -s address 192.168.102.110 netmask 255.255.255.0
   hccn_tool -i 7 -ip -s address 192.168.103.110 netmask 255.255.255.0
   ```

4. 配置防火墙

   - Ubuntu系统防火墙关闭命令

     ```shell
     ufw disable

   - Redhat或CentOS 7系统防火墙关闭命令

     ```shell
     systemctl stop firewalld

## 准备单机多卡模型

1. 获取resnet50模型。

   ```shell
   git clone https://gitee.com/ascend/modelzoo.git
   cd modelzoo
   git reset --hard ca699fcc4022fbea5a0bcf892223abbfaed07a85
   cd built-in/PyTorch/Official/cv/image_classification/ResNet50_for_PyTorch
   ```

2. 在单机多卡上运行训练，确保模型正确。

   运行操作参见[README.md](https://gitee.com/ascend/modelzoo/blob/master/built-in/PyTorch/Official/cv/image_classification/ResNet50_for_PyTorch/README.md)。



## 修改模型脚本

1. 声明的**MASTER_ADDR**变量的值从脚本外部接收**

os.environ['MASTER_ADDR'] = addr

![img](file:///C:\Users\ZoeJ\AppData\Local\Temp\ksohtml\wps39A.tmp.jpg) 

其中addr变量是master节点的host ip，该点体现在启动命令中， 举例模型脚本中已经有该声明，不用再添加；

2. **声明HCCL环境变量**

在启动2*8集群训练之前，不同于单机多卡的是， 要声明下HCCL_IF_IP变量

和HCCL_WHITELIST_DISABLE这2个环境变量。

（1）HCCL_WHITELIST_DISABLE，HCCL通道白名单，一般设置为1，表示关闭白名单。在2台服务器上都声明export HCCL_WHITELIST_DISABLE=1

（2）HCCL_IF_IP, HCCL初始化通信网卡IP，设置为当前服务器的host网卡。2*8集群中，在AI Server0上声明export HCCL_IF_IP=AI Server0 host ip, 在AI Server1上声明export HCCL_IF_IP=AI Server1 host ip。

 

本举例模型脚本中已存在相关改动点， 执行单机多卡训练和多机多卡训练的区别体现在启动命令上， 多机多卡脚本使用上面的mian.py即可。



## 启动训练

（1）将模型脚本上传至AI Server0和AIServer1； 

（2）对照requirements.txt，检查AI Server0和AIServer1是否缺少相关三方库， 该模型用到了DLLogger模块， 需源码安装；

比如AI Server0服务器的host ip为：192.168.xx.22， AI Server1服务器的host ip为：192.168.xx.23。AI Server0为master节点，我们现在拉起2*8的集群。在拉起之前请先将脚本防止服务器相应位置， 确保python相关库已安装。

**在AI** **server0服务器上启动命令：**

source env_npu.sh

export HCCL_WHITELIST_DISABLE=1

export HCCL_IF_IP=192.168.xx.22

python3.7 -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node 8 --master_addr 192.168.xx.22 --master_port 29501 main.py --addr 192.168.xx.22

**在AI** **server****1****服务器上启动命令：**

source env_npu.sh

export HCCL_WHITELIST_DISABLE=1

export HCCL_IF_IP=192.168.xx.23

python3.7 -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node 8 --master_addr 192.168.xx.22--master_port 29501 main.py --addr 192.168.xx.22

以上2个命令的差别是HCCL_IF_IP/node_rank/addr的值不同， 用户可将命令写入shell脚本， 对不同的参数以shell脚本外传值方式启动。

**host日志查看**

cd ~/ascend/log下， 查看host日志

![img](file:///C:\Users\ZoeJ\AppData\Local\Temp\ksohtml\wps499D.tmp.jpg) 



# 适配原理及步骤









 















