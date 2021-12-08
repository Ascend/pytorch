# 概述

用户可以从Ascend ModelZoo获得PyTorch训练模型，但不支持多机多卡训练，需要根据模型的实际代码进行修改。本文帮助用户快速实现Pytorch模型在多机多卡上使用DDP（Distributed Data Parallel）模式训练。

# 训练流程

PyTorch模型多机多卡训练流程一般包括准备环境、准备模型、修改模型、启动训练四个部分。

1. 准备环境。

   准备多机多卡训练的软件、硬件、网络环境，包括开发和运行环境搭建、集群组网链接、芯片IP设置、防火墙设置等。

2. 准备模型。

   准备PyTorch模型、数据加载器、优化器等训练需要的模块，从[开源社区](https://gitee.com/ascend/modelzoo/tree/master/built-in/PyTorch)获取，也可以自行实现。

3. 修改模型。

   在基础模型上进行修改，添加DDP需要的代码和环境变量，使模型支持在多机多卡训练。

4. 启动训练。

   实现在多机多卡环境启动模型训练并查看训练日志。

   

# 快速上手

## 概述

通过示例帮助用户快速了解PyTorch模型是如何在多机多卡上训练的。该示例使用自定义模型实现两台计算机8卡训练。两台计算机分别命名为AI Server0、AI Server1，每台计算机上的8个Ascend 910处理器命名为device 0~7。

## 准备环境

首先您需要具有至少两台装有Ascend 910处理器的计算机，并保证每台计算机都正确的安装NPU固件和驱动。

1. 准备开发和运行环境。

   在每台计算机上分别完成开发和运行环境准备。

   - 完成CANN开发和运行环境的安装，请参见《CANN 软件安装指南》，支持5.0.3以后版本。

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

## 准备模型

该示例创建一个简单的模型，供用户快速了解多机多卡训练。用户也可以从[开源社区](https://gitee.com/ascend/modelzoo/tree/master/built-in/PyTorch)获取基于Ascned NPU的PyTorch训练模型。

1. DDP模型。

   实现一个简单的样例main.py用于多机多卡训练。

   ```python
   import argparse
   import os
   import torch
   import torchvision
   import torch.nn as nn
   import torch.nn.functional as F
   import torch.distributed as dist
   from torch.nn.parallel import DistributedDataParallel as DDP
   
   ### 1.基础模块 ###
   # 搭建模型
   class ToyModel(nn.Module):
       def __init__(self):
           super(ToyModel, self).__init__()
           self.conv1 = nn.Conv2d(3, 6, 5)
           self.pool = nn.MaxPool2d(2, 2)
           self.conv2 = nn.Conv2d(6, 16, 5)
           self.fc1 = nn.Linear(16 * 5 * 5, 120)
           self.fc2 = nn.Linear(120, 84)
           self.fc3 = nn.Linear(84, 10)
   
       def forward(self, x):
           x = self.pool(F.relu(self.conv1(x)))
           x = self.pool(F.relu(self.conv2(x)))
           x = x.view(-1, 16 * 5 * 5)
           x = F.relu(self.fc1(x))
           x = F.relu(self.fc2(x))
           x = self.fc3(x)
           return x
   
   # 获取数据集方法
   def get_dataset():
       transform = torchvision.transforms.Compose([
           torchvision.transforms.ToTensor(),
           torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
       ])
       my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=transform)
      
       train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
       trainloader = torch.utils.data.DataLoader(my_trainset,
                                                 batch_size=16, num_workers=2, sampler=train_sampler)
       return trainloader
   
   
   ### 2. 初始化参数、数据、模型、损失函数、优化器  ####
   # 获取local_rank和addr参数
   parser = argparse.ArgumentParser()
   parser.add_argument("--local_rank", default=-1, type=int)
   parser.add_argument("--addr", default='127.0.0.1', type=str, help='master addr')
   
   FLAGS = parser.parse_args()
   local_rank = FLAGS.local_rank
   addr = FLAGS.addr
   
   # 设置系统的Master地址和端口
   os.environ['MASTER_ADDR'] = addr
   os.environ['MASTER_PORT'] = '29501'
   
   # DDP backend初始化
   loc = 'npu:{}'.format(local_rank)
   torch.npu.set_device(loc)
   dist.init_process_group(backend='hccl') # hccl是NPU设备上的后端
   
   
   # 准备数据，要在DDP初始化之后进行
   trainloader = get_dataset()
   
   # 实例化模型
   model = ToyModel().to(loc)
   
   # 加载模型权重，在构造DDP模型之前，且只需要在master上加载就行了。
   ckpt_path = None
   if dist.get_rank() == 0 and ckpt_path is not None:
       model.load_state_dict(torch.load(ckpt_path))
       
   # 构造DDP model
   model = DDP(model, device_ids=[local_rank], output_device=local_rank)
   
   # 初始化优化器，在构造DDP model之后，用model初始化optimizer。
   optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
   
   # 初始化损失函数
   loss_func = nn.CrossEntropyLoss().to(loc)
   
   ### 3. 网络训练  ###
   model.train()
   iterator = range(100)
   for epoch in iterator:
       trainloader.sampler.set_epoch(epoch)
       for data, label in trainloader:
           data, label = data.to(local_rank), label.to(local_rank)
           optimizer.zero_grad()
           prediction = model(data)
           loss = loss_func(prediction, label)
           loss.backward()
           print("loss = %0.3f \n" % loss)
           optimizer.step()
           
       # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
       #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
       # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
       if dist.get_rank() == 0:
           torch.save(model.module.state_dict(), "%d.ckpt" % epoch)
   ```

2. 在单机多卡上运行训练，确保模型正确。

   1. 用户自行安装模型脚本需要的Python第三方库。

   2. 配置NPU环境变量，env_npu.sh脚本请参见附录。

      ```shell
      source env_npu.sh
      ```

   3. 使用torch.distributed.launch执行main.py如下命令在单机多卡上训练模型。

      ```shell
      python -m torch.distributed.launch --nproc_per_node 8 main.py
      ```

      `--nproc_per_node` 为训练卡的数量。

      运行成功后，模型在该设备的8张NPU上进行训练。

   

## 修改模型

该快速上手提供的示例，已经对多机多卡训练进行了适配，不需要对脚本进行修改。其他模型的多机多卡适配修改，请参考”多机多卡训练“章节。

## 启动训练

1. 将main.py模型脚本上传至AI Server0和AI Server1任意路径下。

2. 查询服务器的host IP：

   ```shell
   hostname -I
   ```

   打印出所有IP，第一个为IP当前服务器的host IP。

   比如：AI Server0服务器的host IP为：192.168.xx.22， AI Server1服务器的host IP为：192.168.xx.23。

3. 使用AI Server0为master节点，我们现在拉起2 x 8的集群。

   在AI server0服务器上启动命令：

   ```shell
   # 设置环境变量,env_npu.sh脚本内容从附录获取
   source env_npu.sh
   # 关闭HCCL通道白名单
   export HCCL_WHITELIST_DISABLE=1
   # HCCL初始化通信网卡IP，设置为当前服务器的host IP
   export HCCL_IF_IP=192.168.xx.22
   # 
   python3.7 -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node 8 --master_addr 192.168.xx.22 --master_port 29501 main.py --addr 192.168.xx.22
   ```

   在AI server1服务器上启动命令：

   ```shell
   # 设置环境变量,env_npu.sh脚本内容从附录获取
   source env_npu.sh
   # 关闭HCCL通道白名单
   export HCCL_WHITELIST_DISABLE=1
   # HCCL初始化通信网卡IP，设置为当前服务器的host IP
   export HCCL_IF_IP=192.168.xx.23
   
   python3.7 -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node 8 --master_addr 192.168.xx.22 --master_port 29501 main.py --addr 192.168.xx.22
   ```

   参数说明。

   --nnode：指定用来分布式训练脚本的节点数。

   --node_rank：多节点分布式训练时，指定当前节点的 rank。

   --nproc_per_node：指定当前节点上，使用GPU训练的进程数。

   --master_addr：master节点（rank为0）的地址，应该为ip地址或者node 0 的 hostname。

   --master_port：指定分布式训练中，master 节点使用的端口号。

   --addr：main.py 脚本的入参，输入master节点的host IP。

3. host日志查看

   host日志保存在`~/ascend/log`路径下，用户可以到该路径下查看每个host的device日志

通过该简单的示例您已经完成了多机多卡模型的训练。 

# 多机多卡训练



## 常用概念及参数介绍

pytorch分布式训练基本概念

| 基本概念   | 说明                                                         |
| ---------- | ------------------------------------------------------------ |
| AI Server  | 带有Ascend 910处理器的计算机，多台计算机用AI Server0+序号表示，如AI Server0、AI Server1。 |
| device     | AI Server上Ascend 910卡，多卡用device 0、device 1……device 7表示。 |
| host       | AI Server主机。                                              |
| master     | 在多台AI Server选取一台作为master，作为数据通信的主机。      |
| group      | 即进程组。默认情况下，只有一个组，采用用默认的就行。         |
| world size | 表示全局的进程并行数，可通过torch.distributed.get_world_size()获取， 在不同进程里，该值是一样的。 |
| rank       | 表示当前进的序号， 用于进程间通讯。比如是2x8的集群，world size 就是16，rank在每个进程里依次是0,1,2,…,15。 |
| local_rank | 每台机子上的进程的序号。机器一上有0,1,2,3,4,5,6,7，机器二上也有0,1,2,3,4,5,6,7。一般情况下，你需要用这个local_rank来设置当前模型是跑在当前机器的哪块GPU/NPU上面的。 |

使用torch.distributed.launch启动多卡训练时的参数

| 参数名称           | 说明                                                         |
| ------------------ | ------------------------------------------------------------ |
| **nnodes**         | 指定用来分布式训练脚本的节点数                               |
| **node_rank**      | 多节点分布式训练时，指定当前节点的 rank                      |
| **nproc_per_node** | 指定当前节点上，使用GPU训练的进程数。建议将该参数设置为当前节点的GPU数量，这样每个进程都能单独控制一个GPU，效率最高。 |
| **master_addr**    | master节点（rank为0）的地址，应该为ip地址或者node 0 的 hostname。对于单节点多进程训练的情况，该参数可以设置为 127.0.0.1。 |
| **master_port**:   | 指定分布式训练中，master 节点使用的端口号，必须与其他应用的端口号不冲突。 |

## 多机多卡训练流程

### 准备环境

首先您需要具有至少两台AI Server（装有Ascend 910处理器的计算机），并保证每台计算机已安装正确版本的NPU固件和驱动。

1. 准备开发和运行环境。

   在每台计算机上分别完成开发和运行环境准备。

   - 完成CANN开发和运行环境的安装，请参见《CANN 软件安装指南》，支持5.0.3以后版本。

   - 安装适配NPU的PyTorch，安装方法请参见《PyTorch安装指南》。

2. 准备组网。

   集群训练由多台装有Ascend 910处理器的计算机完成（最多128台），需要配合交换机组成数据面全连接主备网络，支持8 x n卡训练场景，2台机器可以采用光口直连的方式。搭建方法请参见《[数据中心训练场景组网](https://support.huawei.com/enterprise/zh/doc/EDOC1100221993/229cc0e4)》。

3. 配置device IP

   使用hccn_tool工具配置device IP，hccn_tool工具CANN软件以提供。

   ```shell
   hccn_tool -i 0 -ip -s address 192.168.100.111 netmask 255.255.255.0
   ```

   配置device IP需遵守以下规则：

   1. AI Server中的第0/4，1/5，2/6，3/7号device需处于同一网段，第0/1/2/3号device在不同网段，第4/5/6/7号device在不同网段。
   2. 对于集群场景，各AI Server对应的位置的device需处于同一网段，AI Server0和AI Server1的0号网卡需处于同一网段、1号网卡需要在同一网段
   3. 每个IP都不能冲突，相同网段下的IP需在最后8位做区分

   使用hccn_tool工具验证device IP是否配置正确。

   - 查询每个device的ip。

     ```shell
     hccn_tool -i 0 -ip –g  
     ```

     打印查询结果: 

     > ipaddr:192.168.100.101                        
     >
     > netmask:255.255.255.0                                          

   -  使用hccn_tool 确保2机器的卡间连通性，从device0 - devcie7 测试8次，确保所有两机间所有卡都连通。

     ```shell
     hccn_tool -i 0 -netdetect -s address xx.xx.xx.xx             
     
     hccn_tool -i 0 -net_health –g  
     ```

     -i：device序号

     -s address：xx.xx.xx.xx是另外一台机器的device i的IP                       

     如果返回`success`则表示已经连通。

4. 配置防火墙

   在进行HCCL通信的时候可能出现防火墙拦截通信端口导致通信超时，因此在运行pytorch的集群训练的时候需要注意将服务器上的防火墙关闭。

   - Ubuntu系统防火墙关闭命令

     ```shell
     ufw disable
     ```

   - Redhat或CentOS 7系统防火墙关闭命令

     ```shell
     systemctl stop firewalld
     ```

### 准备模型

准备模型阶段主要有两种方式。

- 从[开源社区](https://gitee.com/ascend/modelzoo/tree/master/built-in/PyTorch)下载PyTorch训练模型

  从开源社区获取的模型已经支持单机多卡训练，请用户参照“修改模型”小节需要修改的项目，根据具体模型完成相应修改。

- 手动搭建PyTorch训练模型

1. 准备PyTorch训练模型、数据加载器

   准备PyTorch模型。

   ```python
   class ToyModel(nn.Module):
   	def __init__(self):
   		...
   	def forward(self,x):
   		...
   ```

   准备数据获取方法。

   ```python
   def get_dataset():
       transform = torchvision.transforms.Compose([
           torchvision.transforms.ToTensor(),
           torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
       ])
       my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=transform)
      
       trainloader = torch.utils.data.DataLoader(my_trainset,batch_size=16,)
       return trainloader
   
   trainloader=get_dataset()
   ```

2. 实例化模型

   ```python
   # 实例化模型
   model = ToyModel().to(loc)
   
   # 加载模型权重
   if ckpt_path is not None:
       model.load_state_dict(torch.load(ckpt_path))
   ```

3. 准备损失函数和优化器。

   ```python
   # 初始化优化器，在构造DDP model之后，用model初始化optimizer。
   optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
   
   # 初始化损失函数
   loss_func = nn.CrossEntropyLoss().to(loc)
   ```

4. 训练模型

   ```python
   ### 3. 网络训练  ###
   model.train()
   iterator = range(100)
   for epoch in iterator:
       for data, label in trainloader:
           data, label = data.to(local_rank), label.to(local_rank)
           optimizer.zero_grad()
           prediction = model(data)
           loss = loss_func(prediction, label)
           loss.backward()
           print("loss = %0.3f \n" % loss)
           optimizer.step()
           
           torch.save(model.state_dict(), "%d.ckpt" % epoch)
   ```

   

### 修改模型

模型修改主要涉及以下6项，包括master ip地址和端口的设置，distributed初始化，模型DDP初始化，数据DDP初始化，优化器初始化，DDP模型训练方法修改。请用户结合初始模型代码，灵活修改。

1. 设置master ip地址和端口，在NPU进行分布式训练使用HCCL进行通信，在PyTorch中使用的是自动拓扑探测的HCCL通信机制，即不需要使用RANK_TABLE_FLIE，但是其依赖于host侧的网卡进行通信，因此需要在代码中设置环境变量来设置通信网卡。

   ```python
   os.environ['MASTER_ADDR'] = xxx.xxx.xxx.xxx
   os.environ['MASTER_PORT'] = 'xxx'
   ```

   MASTER_ADDR：设置为集群中master的IP（任意挑选一台作为master即可）

   MASTER_PORT：设置为master的一个空闲端口

   master ip地址和端口在模型代码中一般会设置为传参的形式，也有可能某些开源代码中设置为"127.0.0.1"，需要进行修改。

   上述变量需在调用torch.distributed.init_process_group()之前声明。

2. distributed初始化

   PyTorch中使用`dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)`来初始化线程组其中参数含义如下。

   `backend`：进行分布式训练的使用的通信协议，在NPU上只能使用"hccl"

   `world_size`：进行训练时使用的device的总数

   `rank`： 当前初始化的device的rank_id，也就是全局的逻辑ID

   有两种方法启动多卡训练，分别初始化的方法如下。

   - 使用torch.distributed.launch启动多卡训练。

     ```python 
     import torch.distributed as dist
     
     dist.init_process_group(backend='hccl') # hccl是NPU设备上的后端
     ```

   - 使用mp.spawn启动多卡训练。

      ```python
      import torch.distributed as dist
      
      def main_worker(pid_idx, device_nums_per_node, args):
          args.distributed_rank = args.rank * device_nums_per_node + pid_idx
          dist.init_process_group(backend=args.dist_backend, world_size=args.distributed_world_size, rank=args.distributed_rank)
      ```

     其中：

     `pid_idx`：device序号。

     `device_nums_per_node`：每个AI Server的device数量。

3. 模型DDP初始化

   ```python
   # 实例化模型
   model = ToyModel().to(loc)
   
   # 加载模型权重，在构造DDP模型之前，且只需要在master上加载。
   if dist.get_rank() == 0 and ckpt_path is not None:
       model.load_state_dict(torch.load(ckpt_path))
       
   # 构造DDP model
   model = DDP(model, device_ids=[local_rank], output_device=local_rank)
   ```

4. 数据DDP初始化

   ```python
   def get_dataset():
       transform = torchvision.transforms.Compose([
           torchvision.transforms.ToTensor(),
           torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
       ])    
       my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=transform)
      
       train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
       trainloader = torch.utils.data.DataLoader(my_trainset,
                                                 batch_size=16, num_workers=2, sampler=train_sampler)
   	return trainloader
   
   trainloader = get_dataset()
   ```

5. 损失方法、优化器。 

   ```python
   # 初始化优化器，在构造DDP model之后，用model初始化optimizer。
   optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
   
   # 初始化损失函数
   loss_func = nn.CrossEntropyLoss().to(loc)
   ```

6. DDP模型训练

   ```python
   ### 3. 网络训练  ###
   model.train()
   iterator = range(100)
   for epoch in iterator:
       trainloader.sampler.set_epoch(epoch)
       for data, label in trainloader:
           data, label = data.to(local_rank), label.to(local_rank)
           optimizer.zero_grad()
           prediction = model(data)
           loss = loss_func(prediction, label)
           loss.backward()
           print("loss = %0.3f \n" % loss)
           optimizer.step()
           
       # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
       #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
       # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
       if dist.get_rank() == 0:
           torch.save(model.module.state_dict(), "%d.ckpt" % epoch)
   ```

### 启动训练

启动训练提供两种方式，手动启动和shell脚本启动

- 手动启动

  1. 添加环境变量，多机训练需要增加`HCCL_WHITELIST_DISABLE`和`HCCL_IF_IP`环境变量。

  -   HCCL_WHITELIST_DISABLE：HCCL通道白名单，一般性设置为1表示关闭白名单。

  - HCCL_IF_IP：HCCL初始化通信网卡IP，设置为当前服务器的host网卡IP。

  

  本部分的说明中使用的是torch.distributed.launch来启动多卡训练

  （1）将模型脚本上传至AI Server0和AIServer1； 

  （2）对照requirements.txt，检查AI Server0和AIServer1是否缺少相关三方库， 该模型用到了DLLogger模块， 需源码安装；

  比如AI Server0服务器的host ip为：192.168.xx.22， AI Server1服务器的host ip为：192.168.xx.23。AI Server0为master节点，我们现在拉起2*8的集群。在拉起之前请先将脚本防止服务器相应位置， 确保python相关库已安装。

  **在AI** **serveri服务器上启动命令：**

  source env_npu.sh

  export HCCL_WHITELIST_DISABLE=1

  export HCCL_IF_IP=192.168.xx.xx

  python3.7 -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node 8 --master_addr 192.168.xx.22 --master_port 29501 main.py --addr 192.168.xx.22

  以上2个命令的差别是HCCL_IF_IP/node_rank/addr的值不同， 用户可将命令写入shell脚本， 对不同的参数以shell脚本外传值方式启动。

- 使用sh脚本启动

  用户可将命令写入shell脚本， 对不同的参数以shell脚本外传值方式启动。

训练成功后可以查看日志信息

**host日志查看** 



# 附录

NPU环境变量配置脚本env_npu.sh，可使用该脚本进行运行和开发环境变量的配置。

```shell
#!/bin/bash
export install_path=/usr/local/Ascend

if [ -d ${install_path}/toolkit ]; then
    export LD_LIBRARY_PATH=/usr/include/hdf5/lib/:/usr/local/:/usr/local/lib/:/usr/lib/:${install_path}/fwkacllib/lib64/:${install_path}/driver/lib64/common/:${install_path}/driver/lib64/driver/:${install_path}/add-ons:${path_lib}:${LD_LIBRARY_PATH}
    export PATH=${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:$PATH
    export PYTHONPATH=${install_path}/fwkacllib/python/site-packages:${install_path}/tfplugin/python/site-packages:${install_path}/toolkit/python/site-packages:$PYTHONPATH
    export PYTHONPATH=/usr/local/python3.7.5/lib/python3.7/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=${install_path}/opp
else
    if [ -d ${install_path}/nnae/latest ];then
exportLD_LIBRARY_PATH=/usr/local/:/usr/local/python3.7.5/lib/:/usr/local/openblas/lib:/usr/local/lib/:/usr/lib64/:/usr/lib/:${install_path}/nnae/latest/fwkacllib/lib64/:${install_path}/driver/lib64/common/:${install_path}/driver/lib64/driver/:${install_path}/add-ons/:/usr/lib/aarch64_64-linux-gnu:$LD_LIBRARY_PATH
        export PATH=$PATH:${install_path}/nnae/latest/fwkacllib/ccec_compiler/bin/:${install_path}/nnae/latest/toolkit/tools/ide_daemon/bin/
        export ASCEND_OPP_PATH=${install_path}/nnae/latest/opp/
        export OPTION_EXEC_EXTERN_PLUGIN_PATH=${install_path}/nnae/latest/fwkacllib/lib64/plugin/opskernel/libfe.so:${install_path}/nnae/latest/fwkacllib/lib64/plugin/opskernel/libaicpu_engine.so:${install_path}/nnae/latest/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so
        export PYTHONPATH=${install_path}/nnae/latest/fwkacllib/python/site-packages/:${install_path}/nnae/latest/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:${install_path}/nnae/latest/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
        export ASCEND_AICPU_PATH=${install_path}/nnae/latest
    else
        export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib64/:/usr/lib/:/usr/local/python3.7.5/lib/:/usr/local/openblas/lib:${install_path}/ascend-toolkit/latest/fwkacllib/lib64/:${install_path}/driver/lib64/common/:${install_path}/driver/lib64/driver/:${install_path}/add-ons/:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
        export PATH=$PATH:${install_path}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${install_path}/ascend-toolkit/latest/toolkit/tools/ide_daemon/bin/
        export ASCEND_OPP_PATH=${install_path}/ascend-toolkit/latest/opp/
        export OPTION_EXEC_EXTERN_PLUGIN_PATH=${install_path}/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel/libfe.so:${install_path}/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel/libaicpu_engine.so:${install_path}/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so
        export PYTHONPATH=${install_path}/ascend-toolkit/latest/fwkacllib/python/site-packages/:${install_path}/ascend-toolkit/latest/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:${install_path}/ascend-toolkit/latest/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
        export ASCEND_AICPU_PATH=${install_path}/ascend-toolkit/latest
    fi
fi

#将Host日志输出到串口,0-关闭/1-开启
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#设置默认日志级别,0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3
#设置Event日志开启标志,0-关闭/1-开启
export ASCEND_GLOBAL_EVENT_ENABLE=0
#设置是否开启taskque,0-关闭/1-开启
export TASK_QUEUE_ENABLE=1
#HCCL白名单开关,1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1

#设置device侧日志登记为error
${install_path}/driver/tools/msnpureport -g error -d 0
${install_path}/driver/tools/msnpureport -g error -d 1
${install_path}/driver/tools/msnpureport -g error -d 2
${install_path}/driver/tools/msnpureport -g error -d 3
${install_path}/driver/tools/msnpureport -g error -d 4
${install_path}/driver/tools/msnpureport -g error -d 5
${install_path}/driver/tools/msnpureport -g error -d 6
${install_path}/driver/tools/msnpureport -g error -d 7
#关闭Device侧Event日志
${install_path}/driver/tools/msnpureport -e disable

path_lib=$(python3.7 -c """
import sys
import re
result=''
for index in range(len(sys.path)):
    match_sit = re.search('-packages', sys.path[index])
    if match_sit is not None:
        match_lib = re.search('lib', sys.path[index])

        if match_lib is not None:
            end=match_lib.span()[1]
            result += sys.path[index][0:end] + ':'

        result+=sys.path[index] + '/torch/lib:'
print(result)"""
)

echo ${path_lib}

export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib/:${path_lib}:$LD_LIBRARY_PATH
```















