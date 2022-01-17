# Overview 

Users can obtain PyTorch training models from Ascend ModelZoo, but the models do not support multi-server multi-device training. You need to modify the models based on the actual model code. This document describes how to quickly train a PyTorch model in Distributed Data Parallel (DDP) mode in multi-server multi-device scenario.

# Training Workflow

The process of training a PyTorch model in multi-server multi-device scenario includes environment preparation, model preparation, model modification, and training startup.

1. Environment Preparation

   Prepare the software, hardware, and network environment for multi-server multi-device training, including setting up the development and operating environment, connecting the cluster network, setting the processor IP address, and setting the firewall.

2. Model Preparation

   Prepare a PyTorch model, data loader, and optimizer for training. You can download them from the open source community (https://gitee.com/ascend/modelzoo/tree/master/built-in/PyTorch) or prepare them by yourself.

3. Model Modification

   Modify the basic model and add the code and environment variables required by DDP to enable the multi-server multi-device training.

4. Training Startup

   Start model training in the multi-server multi-device scenario and view training logs.

   

# Quick Start

## Overview

The example presented in this document helps you quickly understand how a PyTorch model is trained in multi-server multi-device scenario. The example uses a custom model for training in two-computer eight-device scenario. The two computers are named AI Server0 and AI Server1. The eight Ascend 910 Processors on each computer are named device0 to device7.

## Preparing the Environment

At least two computers with Ascend 910 Processors installed are required, and the NPU firmware and driver are correctly installed on each computer.

1. Prepare the development and operating environment on each computer.

   - Install the CANN development and operating environment. For details, see the *CANN Software Installation Guide*. Use CANN later than 5.0.3.

   - Install the PyTorch that adapts to NPU. For details, see the *PyTorch Installation Guide*.

2. Prepare the network.

   Set up the network by directly connecting switches or optical ports. For details, see the *Ascend Data Center Solution Networking Guide* at https://support.huawei.com/enterprise/en/doc/EDOC1100221995/229cc0e4.

   In this example, two computers with eight devices are used for training, so optical ports are used for network connection.

3. Configure the device IP address.

   Configure the device IP address on AI Server0.

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

   Configure the device IP address on AI Server1.

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

4. Configure the firewall.

   - Command for disabling the firewall on Ubuntu:

     ```shell
     ufw disable
     ```
  ```
   
- Command for disabling the firewall on Red Hat or CentOS 7:
   
     ```shell
     systemctl stop firewalld
  ```

## Preparing a Model

This example creates a simple model for you to quickly understand multi-server multi-device training. You can also obtain the Ascend NPU-based PyTorch training model from the open source community (https://gitee.com/ascend/modelzoo/tree/master/built-in/PyTorch).

1. Prepare a DDP model.

   The following is an example of main.py for multi-server multi-device training.

   ```python
   import argparse
   import os
   import torch
   import torchvision
   import torch.nn as nn
   import torch.nn.functional as F
   import torch.distributed as dist
   from torch.nn.parallel import DistributedDataParallel as DDP
   
   ### 1. Perform basic operations. ###
   # Build a model.
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
   
   # Obtain a dataset.
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
   
   
   ### 2. Initialize the parameters, data, model, loss function, and optimizer. ####
   # Obtain local_rank and addr.
   parser = argparse.ArgumentParser()
   parser.add_argument("--local_rank", default=-1, type=int)
   parser.add_argument("--addr", default='127.0.0.1', type=str, help='master addr')
   
   FLAGS = parser.parse_args()
   local_rank = FLAGS.local_rank
   addr = FLAGS.addr
   
   # Set the IP address and port of the master node.
   os.environ['MASTER_ADDR'] = addr
   os.environ['MASTER_PORT'] = '29501'
   
   # Initialize the DDP backend.
   loc = 'npu:{}'.format(local_rank)
   torch.npu.set_device(loc)
   dist.init_process_group(backend='hccl') # HCCL is the backend of the NPU device.
   
   
   # Prepare data after DDP initialization.
   trainloader = get_dataset()
   
   # Instantiate the model.
   model = ToyModel().to(loc)
   
   # Load the model weight. The weight needs to be loaded only on the master node before the DDP model is built.
   ckpt_path = None
   if dist.get_rank() == 0 and ckpt_path is not None:
       model.load_state_dict(torch.load(ckpt_path))
       
   # Build the DDP model.
   model = DDP(model, device_ids=[local_rank], output_device=local_rank)
   
   # Initialize the optimizer. After the DDP model is built, use the model to initialize the optimizer.
   optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
   
   # Initialize the loss function.
   loss_func = nn.CrossEntropyLoss().to(loc)
   
   ### 3. Train the network.
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
           
       # 1. Similar to the DP mode, when you save a model, note that model.module instead of model is saved.
       #    That is because the model actually refers to a DDP model, and the parameters are packaged by `model=DDP(model)`.
       # 2. You only need to save the model on process 0 once to avoid repeated saving.
       if dist.get_rank() == 0:
           torch.save(model.module.state_dict(), "%d.ckpt" % epoch)
   ```

2. Ensure that the model is correct for training in the single-server multi-device scenario.

   1. The Python third-party library is required if you install the model script yourself.

   2. Configure NPU environment variables. For information about the **env_npu.sh** script, see the appendix.

      ```shell
      source env_npu.sh
      ```

   3. Execute **torch.distributed.launch** to run the following main.py command to train a model in single-server multi-device scenario.

      ```shell
      python -m torch.distributed.launch --nproc_per_node 8 main.py
      ```

      `--nproc_per_node` indicates the number of training cards.

      After the command is run successfully, the model is trained on the eight NPUs of the device.

   

## Modifying the Model

The example provided in "Quick Start" is adapted to the multi-server multi-device training. You do not need to modify the script. For details about how to modify other models, see section "Multi-server Multi-device Training".

## Starting the Training

1. Upload main.py model script to any directory on AI Server0 and AI Server1.

2. Query the host IP addresses of the servers.

   ```shell
   hostname -I
   ```

   All IP addresses are displayed, and the first IP address is the host IP address of the current server.

   For example, the host IP address of AI Server0 is **192.168.*xx*.22**, and that of AI Server1 is **192.168.*xx*.23**.

3. Use AI Server0 as the master node, and start the 2 x 8 cluster.

   Startup commands for AI Server0:

   ```shell
   # Set environment variables. Obtain the env_npu.sh script content from the appendix.
   source env_npu.sh
   # Disable the trustlist of HCCL channel.
   export HCCL_WHITELIST_DISABLE=1
   # Initialize the IP address of the HCCL communication NIC, and set the IP address to the host IP address of the current server.
   export HCCL_IF_IP=192.168.xx.22
   # 
   python3.7 -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node 8 --master_addr 192.168.xx.22 --master_port 29501 main.py --addr 192.168.xx.22
   ```

   Startup commands for AI Server1:

   ```shell
   # Set environment variables. Obtain the env_npu.sh script content from the appendix.
   source env_npu.sh
   # Disable the trustlist of HCCL channel.
   export HCCL_WHITELIST_DISABLE=1
   # Initialize the IP address of the HCCL communication NIC, and set the IP address to the host IP address of the current server.
   export HCCL_IF_IP=192.168.xx.23
   
   python3.7 -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node 8 --master_addr 192.168.xx.22 --master_port 29501 main.py --addr 192.168.xx.22
   ```

   Parameter description:

   --nnode: specifies the number of nodes used for distributed training scripts.

   --node_rank: specifies the rank of the current node during multi-node distributed training.

   --nproc_per_node: specifies the number of GPU-based training processes on the current node.

   --master_addr: address of the master node (rank is 0). The value can be the IP address or the host name of node 0.

   --master_port: specifies the port number used by the master node during distributed training.

   --addr: input parameter of the main.py script, specifying the host IP address of the master node.

3. View the host logs.

   Host logs are stored in the `~/ascend/log` directory. You can go to this directory to view the device logs of each host.

# Multi-server Multi-device Training

## Common Concepts and Parameters

Basic concepts for PyTorch distributed training

| Basic Concept | Description                                                  |
| :-----------: | ------------------------------------------------------------ |
|   AI Server   | Computer with an Ascend 910 Processors. Multiple computers are identified as AI Server +serial number, for example, AI Server0 and AI Server1. |
|    device     | Ascend 910 Processors on the AI server. Multiple processors are represented as device 0, device 1, ..., and device 7. |
|     host      | AI server host.                                              |
|    master     | Select one of multiple AI servers as the master node for data communication. |
|     group     | Process group. By default, there is only one group. Use the default value. |
|  world size   | Number of global parallel processes, which can be obtained by running **torch.distributed.get_world_size()**. The value is the same for different processes. |
|     rank      | Sequence number of the current process, which is used for communication between processes. For example, for a 2 x 8 cluster, the **world size** is 16, and the rank in each process is [0, 1, 2, ..., 15]. |
|  local_rank   | Sequence number of processes on each host, for example, there are processes 0-7 on each host. Generally, **local_rank** is used to set the GPU/NPU on which the current model runs. |

Parameters for executing **torch.distributed.launch** to start multi-device training

| Parameter          | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| **nnodes**         | Specifies the number of nodes used for distributed training scripts. |
| **node_rank**      | Specifies the rank of the current node during multi-node distributed training. |
| **nproc_per_node** | Specifies the number of GPU-based training processes on the current node. You are advised to set this parameter to the number of GPUs on the current node. In this way, each process can independently control a GPU to achieve the highest efficiency. |
| **master_addr**    | Address of the master node (rank is 0). The value can be the IP address or the host name of node 0. For single-node multi-process training, set this parameter to **127.0.0.1**. |
| **master_port**:   | Specifies the port number used by the master node during distributed training. The port number must be different from the port numbers of other applications. |

## Multi-server Multi-device Training Process

### Preparing the Environment

At least two computers with Ascend 910 Processors installed are required, and the NPU firmware and driver are correctly installed on each server.

1. Prepare the development and operating environment on each computer.

   - Install the CANN development and operating environment. For details, see the *CANN Software Installation Guide*. Use CANN later than 5.0.3.

   - Install the PyTorch that adapts to NPU. For details, see the *PyTorch Installation Guide.*

2. Prepare the network.

   Cluster training is completed by multiple computers (a maximum of 128) with Ascend 910 Processors installed. The computers need to work with switches to form a fully-connected active/standby network on the data plane. The 8 x *n*-device training is supported. Two computers can be directly connected through optical ports. For details, see the *Ascend Data Center Solution Networking Guide * at https://support.huawei.com/enterprise/en/doc/EDOC1100221995/229cc0e4.

3. Configure the device IP address.

   Use hccn_tool to configure the device IP address. hccn_tool is provided by CANN.

   ```shell
   hccn_tool -i 0 -ip -s address 192.168.100.111 netmask 255.255.255.0
   ```

   Observe the following rules when configuring the device IP address:

   1. On the AI servers, devices 0/4, 1/5, 2/6, and 3/7 must be in the same network segment. But devices 0, 1, 2, and 3 must be in different network segments, and devices 4, 5, 6, and 7 must be in different network segments.
   2. In the cluster scenario, the devices corresponding to each AI server must be in the same network segment. NICs 0 and 1 of AI Server0 and AI Server1 must be in the same network segment.
   3. Each IP address must be unique. IP addresses in the same network segment must be distinguished by the last eight bits.

   Use hccn_tool to check whether the device IP address is correct.

   - Query the IP address of each device.

     ```shell
     hccn_tool -i 0 -ip –g  
     ```

     The IP addresses are displayed:

     > ipaddr:192.168.100.101                        
     >
     > netmask:255.255.255.0                                          

   -  Use hccn_tool to ensure that the devices of the two hosts are correctly connected by performing the test for eight times from device0 to devcie7.

     ```shell
     hccn_tool -i 0 -netdetect -s address xx.xx.xx.xx             
     
     hccn_tool -i 0 -net_health –g  
     ```

     **-i**: device ID.

     **-s address**: ***xx.xx.xx.xx*** is the IP address of device *i* on the other host.

     If `success` is returned, the connection is successful.

4. Configure the firewall.

   During HCCL communication, the firewall may intercept the communication port, causing communication timeout. Therefore, you need to disable the firewall on the server for PyTorch cluster training.

   - Command for disabling the firewall on Ubuntu:

     ```shell
     ufw disable
     ```

   - Command for disabling the firewall on Red Hat or CentOS 7:

     ```shell
     systemctl stop firewalld
     ```

### Preparing a Model

There are two methods for preparing a model.

- Download a PyTorch training model from the open source community (https://gitee.com/ascend/modelzoo/tree/master/built-in/PyTorch).

  The model obtained from the open source community supports single-server multi-device training. Modify the model based on the related parameters described in section "Modifying the Model".

- Manually build a PyTorch training model.

1. Prepare a PyTorch training model and data loader.

   Prepare a PyTorch model:

   ```python
   class ToyModel(nn.Module):
   	def __init__(self):
   		...
   	def forward(self,x):
   		...
   ```

   Prepare data:

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

2. Instantiate the model.

   ```python
   # Instantiate the model.
   model = ToyModel().to(loc)
   
   # Load the model weight
   if ckpt_path is not None:
       model.load_state_dict(torch.load(ckpt_path))
   ```

3. Prepare the loss function and optimizer.

   ```python
   # Initialize the optimizer. After the DDP model is built, use the model to initialize the optimizer.
   optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
   
   # Initialize the loss function.
   loss_func = nn.CrossEntropyLoss().to(loc)
   ```

4. Train the model.

   ```python
   ### 3. Train the network.
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

   

### Modifying the Model

Modify the IP address and port of the master node, initialize the **distributed ** function,  model DDP,  data DDP, and optimizer, and modify the DDP model training method based on the initial model code.

1. Set the IP address and port for the master node. In the NPU distributed training, HCCL is used for communication, while in PyTorch, the HCCL communication mechanism that is detected by automatic topology is used. That is, **RANK_TABLE_FLIE** is not required, but the communication depends on the NIC on the host side. Therefore, you need to set environment variables in the code to set the communication NIC.

   ```python
   os.environ['MASTER_ADDR'] = xxx.xxx.xxx.xxx
   os.environ['MASTER_PORT'] = 'xxx'
   ```

   **MASTER_ADDR**: Set this parameter to the IP address of the master node in the cluster. (Select any host as the master node.)

   **MASTER_PORT**: Set this parameter to the idle port of the master node.

   In the model code, the IP address and port number of the master node are generally presented as transferred parameters. In some open-source code, the IP address and port number may be presented as **127.0.0.1**. In this case, you need to modify them.

   The preceding variables must be declared before **torch.distributed.init_process_group()** is invoked.

2. Initialize **distributed**.

   In PyTorch, `dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)` is used to initialize thread groups. The parameters are described as follows:

   `backend`: communication protocol used for distributed training. Only hccl can be used on the NPU.

   `world_size`: total number of devices used for training.

   `rank`: rank ID of the currently initialized device, that is, the global logical ID.

   There are two methods to start multi-device training:

   - **torch.distributed.launch**:

     ```python 
     import torch.distributed as dist
     
     dist.init_process_group(backend='hccl') # HCCL is the backend of the NPU device.
     ```

   - **mp.spawn**:

      ```python
      import torch.distributed as dist
      
      def main_worker(pid_idx, device_nums_per_node, args):
          args.distributed_rank = args.rank * device_nums_per_node + pid_idx
          dist.init_process_group(backend=args.dist_backend, world_size=args.distributed_world_size, rank=args.distributed_rank)
      ```

     In the preceding commands:

     `pid_idx`: device ID.

     `device_nums_per_node`: number of devices on each AI server.

3. Initialize the model DDP.

   ```python
   # Instantiate the model.
   model = ToyModel().to(loc)
   
   # Load the model weight. It needs to be loaded only on the master node before the DDP model is built.
   if dist.get_rank() == 0 and ckpt_path is not None:
       model.load_state_dict(torch.load(ckpt_path))
       
   # Build the DDP model.
   model = DDP(model, device_ids=[local_rank], output_device=local_rank)
   ```

4. Initialize the data DDP.

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

5. Initialize the loss function and optimizer.

   ```python
   # Initialize the optimizer. After the DDP model is built, use the model to initialize the optimizer.
   optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
   
   # Initialize the loss function.
   loss_func = nn.CrossEntropyLoss().to(loc)
   ```

6. Train the DDP model.

   ```python
   model.train()
   iterator = range(100)
   for epoch in iterator:
       # Set epoch.
       trainloader.sampler.set_epoch(epoch)
       
       for data, label in trainloader:
           data, label = data.to(local_rank), label.to(local_rank)
           optimizer.zero_grad()
           prediction = model(data)
           loss = loss_func(prediction, label)
           loss.backward()
           print("loss = %0.3f \n" % loss)
           optimizer.step()
           
       # 1. Similar to the DP mode, when you save a model, note that model.module instead of model is saved.
       #    That is because the model actually refers to a DDP model, and the parameters are packaged by `model=DDP(model)`.
       # 2. You only need to save the model on process 0 once to avoid repeated saving.
       if dist.get_rank() == 0:
           torch.save(model.module.state_dict(), "%d.ckpt" % epoch)
   ```

### Start the Training

The training can be started manually or using the shell script.

- Start the training manually using **torch.distributed.launch**.

  1. Configure NPU environment variables. For details, see the **env_npu.sh** script in the appendix.

  2. Add environment variables. For multi-server training, add the `HCCL_WHITELIST_DISABLE` and `HCCL_IF_IP` environment variables.

     - **HCCL_WHITELIST_DISABLE**: HCCL channel trustlist. The value **1** indicates that the trustlist is disabled.
     - **HCCL_IF_IP**: initialized IP address of the HCCL communication NIC. Set it to the IP address of the host NIC IP of the current server.

  3. Upload the modified model script to each AI server.

  4. Install the required Python library on each AI server.

  5. Select an AI server as the master node and query the IP address of each AI server.

  6. Run the following commands on each AI server:

     ```
     python3 -m torch.distributed.launch --nnodes=${nnodes}  --node_rank=i --nproc_per_node 8 --master_addr 192.168.xx.22 --master_port 29501 main.py --addr 192.168.xx.22
     ```

     In the preceding commands:

     **--nnodes**: number of AI servers used for distributed training scripts.

     **--node_rank**: AI server ID.

     **--nproc_per_node**: number of devices of each AI server.

     **--master_addr**: IP address of the AI server that functions as the master node.

     **--master_port**: port number of the AI server that functions as the master node.

     **main.py**: Change it to the name of the startup script.

     **--addr**: indicates the IP address of the master node, which is a parameter transferred to the startup script.

- Start the training using Open MPI.

  1. Install the  PI open-source library.

     In the multi-server multi-device scenario, distributed training deployment depends on the Open MPI open-source library, which must be installed on each server that participates in model training. Currently, Open MPI 4.0.1, 4.0.2, or 4.0.3 is required.
     Run the **mpirun --version** command to check whether Open MPI has been installed. If `mpirun (Open MPI) 4.0.2 Report bugs to http://www.open-mpi.org/community/help/` is returned, Open MPI has been installed. If it has been installed and its version is 4.0.1, 4.0.2, or 4.0.3, you do not need to install it again.

     Otherwise, perform the following steps to install it

     1. Visit the following link to download the Open MPI software package, for example, openmpi-4.0.2.tar.bz2.
        https://www.open-mpi.org/software/ompi/v4.0/

     2. Log in to the installation environment as the root user.

     3. Upload the downloaded Open MPI software package to a directory in the installation environment.

     4. Go to the directory and run the following command to decompress the software package:

        ```shell
        tar -jxvf openmpi-4.0.2.tar.bz2
        ```

     5. Go to the directory generated after the decompression, and run the following commands to configure, compile, and install Open MPI:

        ```shell
        ./configure --prefix=/usr/local/mpirun4.0.2
        make
        make install
        ```

        The **--prefix** parameter specifies the Open MPI installation path. Change it based on the site requirements.

     6. Run the **vi ~/.bashrc** command to open the **.bashrc** file, and add the following environment variables to the end of the file:

        ```shell
        export OPENMPI=/usr/local/mpirun4.0.2
        export LD_LIBRARY_PATH=$OPENMPI/lib
        export PATH=$OPENMPI/bin:$PATH
        ```

        In the environment variables, **/usr/local/mpirun4.0.2** indicates the Open MPI installation path. Change it based on the site requirements.
        Run the **:wq!** command to save the file and exit.

     7. Make the configuration take effect.

        ```
        source ~/.bashrc
        ```

     8. After the installation is complete, run the following command to check the installation version. If the required version information is displayed, the installation is successful.

        ```
        mpirun --version
        ```

  2. Configure SSH password-free login for the AI servers.

     If Open MPI is used for distributed training deployment in the multi-server multi-device scenario, you need to configure SSH password-free login between every two servers to ensure that the servers can communicate with each other. The procedure is as follows:

     1. Log in to each server as the root user.

     2. Configure the reliability among hosts in the cluster.

        Open the **/etc/ssh/ssh_config** file and add the following fields to the end of the file:

        ```
        StrictHostKeyChecking no
        UserKnownHostsFile /dev/null
        ```

     3. Open the **/etc/hosts** file on each server and add the corresponding IP address and host name of the server to the first line of the file. If the file already contains the IP address and host name, skip this step. The following is an example of the content to be added:

        ```
        10.90.140.199 ubuntu
        ```

        In the preceding content, **10.90.140.199** is the IP address of the server, and **ubuntu** is the host name.

     4. Run the following commands on the first server to generate a public key (Assume that the IP address of the first server is **10.90.140.199**.):

        ```
        cd ~/.ssh/                       # If the directory does not exist, run the ssh localhost command first.
        ssh-keygen -t rsa                # After the key is generated, a message is displayed. Press Enter for three consecutive times.
        mv id_rsa.pub authorized_keys    # Renames the generated key id_rsa.pub to authorized_keys.
        ```

     5. Generate a key on each of the other servers, and copy the keys to the **authorized_keys** file on the first server.

        1.  Run the following commands on each of the other servers to generate a key:

                cd ~/.ssh/
                ssh-keygen -t rsa 
            
        2. Download the key file **id_rsa.pub** generated on each server to the local host and copy the key in the file.

        3. On the first server, run the following command to open the authorized_keys file and copy the keys of each of other servers to the end of the public key of the first server.
           
           ```
           vi ~/.ssh/authorized_keys
           ```

           Run the **:wq!** command to save the file.
        
      6. Run the following commands on each of the other servers to copy the public key of the first server to each of the other servers:
         
            cd ~/.ssh/
                  scp root@10.90.140.199:~/.ssh/authorized_keys ./
         
      7. Run the following command on each server to test password-free login:
         
         ```
            ssh User name@IP address
         ```
         
         For example, run the **ssh root@10.90.140.231** command to log in to the server whose IP address is 10.90.140.231 from the first server whose IP address is 10.90.140.199 without a password.
         
         If information similar to the following is displayed, the login without a password is successful.
         
         ```
            Linux ubuntu 4.19.28 #1 SMP Tue Jun 23 19:05:23 EDT 2020 x86_64
            
            The programs included with the ubuntu GNU/Linux system are free software;
            the exact distribution terms for each program are described in the
            individual files in /usr/share/doc/*/copyright.
            
            ubuntu GNU/Linux comes with ABSOLUTELY NO WARRANTY, to the extent
            permitted by applicable law.
            Last login: Tue Sep 15 14:37:21 2020 from 10.254.88.75
         ```
         
         You can run the **exit** command to log out of the server. If information similar to the following is displayed, the logout is successful.
         
         ```
            logout
            Connection to 10.90.140.231 closed.
         ```
   3. Use Open MPI to start model training.
  
      1. Compile a startup script for each AI server, for example, train.sh, and move the startup script to the same path of the corresponding AI server.
      
         ```
         # Configure NPU environment variables. For information about the env_npu.sh script, see the appendix.
         source env_npu.sh
         #Disable the trustlist of HCCL channel.
         export HCCL_WHITELIST_DISABLE=1
         # Initialize the IP address of the HCCL communication NIC , and set the IP address to the host IP address of the current server.
         export HCCL_IF_IP=xxx.xxx.xx.xxx
         python3 -m torch.distributed.launch --nnodes=${nnodes}  --node_rank=i --nproc_per_node 8 --master_addr xxx.xxx.xx.xxx --master_port 29501 main.py --addr xxx.xxx.xx.xxx
         ```
      
         For details about the script parameters, see section "Start the training manually using **torch.distributed.launch**".
      
      2. Compile the startup script.
      
         ```
         # Configuring the mpirun environment variables.
         export PATH=$PATH:/usr/local/mpirun4.0.2/bin
         # Run the mpirun tool.
         mpirun -H xxx.xxx.xxx.xxx:1,xxx.xxx.xxx.xxx:1 \
                  --bind-to none -map-by slot \
                  --mca btl_tcp_if_exclude lo,docker0,endvnic\
                  --allow-run-as-root \
                  --prefix /usr/local/mpirun4.0.2/ \
                  ./train.sh
         ```
      
         In the preceding command:
      
         **-H**: IP address of each AI server and the number of started processes.
      
         **--bind-to**: process-binding policy.
      
         **--mca**: MCA parameter in a specific context. **arg0** is the parameter name, and **arg1** is the parameter value.
      
         **--allow-run-as-root**: The root user is allowed to run this script.
      
         **--prefix**: path of mpirun4.0.2.
      
         **./train.sh**: path of the startup script of each AI server.
      
   4. View the log information after the training succeeds.



   Host logs are stored in the `~/ascend/log` directory. You can go to this directory to view the device logs of each host.

# Appendix

The following shows the NPU environment variable configuration script **env_npu.sh**, which can be used to configure the operating and development environment variables.

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

#Output host logs to the serial port. 0: disable; 1: enable.
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#Set the default log level. 0: debug; 1: info; 2: warning; 3: error.
export ASCEND_GLOBAL_LOG_LEVEL=3
#Enable or disable the event log. 0: disable; 1: enable.
export ASCEND_GLOBAL_EVENT_ENABLE=0
#Enable or disable taskque. 0: disable; 1: enable.
export TASK_QUEUE_ENABLE=1
#Enable or disable the HCCL trustlist. 1: disable; 0: enable.
export HCCL_WHITELIST_DISABLE=1

#Set the device-side log to error.
${install_path}/driver/tools/msnpureport -g error -d 0
${install_path}/driver/tools/msnpureport -g error -d 1
${install_path}/driver/tools/msnpureport -g error -d 2
${install_path}/driver/tools/msnpureport -g error -d 3
${install_path}/driver/tools/msnpureport -g error -d 4
${install_path}/driver/tools/msnpureport -g error -d 5
${install_path}/driver/tools/msnpureport -g error -d 6
${install_path}/driver/tools/msnpureport -g error -d 7
#Disable the event log on the device side.
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