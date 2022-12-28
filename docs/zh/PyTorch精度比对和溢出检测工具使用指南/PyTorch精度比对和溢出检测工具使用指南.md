# **PyTorch精度对比和溢出检测工具使用指南**
- [**PyTorch精度对比工具使用指南**](#pytorch精度对比工具使用指南)
  - [**使用场景**](#使用场景)
  - [**前提条件**](#前提条件)
  - [**注意事项**](#注意事项)
  - [**精度比对基本原理**](#精度比对基本原理)
    - [**算子匹配条件**](#算子匹配条件)
    - [**计算精度评价指标**](#计算精度评价指标)
  - [**快速上手样例参考**](#快速上手样例参考)


## **使用场景**

在同一模型或算子调试过程中，遇到算子相关的计算精度问题，定位时费时费力，所以推出了一个精度比对工具。

精度对比工具，通过在PyTorch模型中注入hook，跟踪计算图中算子的前向传播与反向传播时的输入与输出，排查存在计算精度误差，进行问题的精准定位。

主要的使用场景包括：

- 同一模型，从cpu(或npu)移植到npu中存在精度下降问题，对比npu芯片中的算子计算数值与cpu芯片中的算子计算数值，进行问题定位。
- 同一模型，进行迭代(模型、算子或设备迭代)时存在的精度下降问题，对比相同模型在迭代前后版本的算子计算数值，进行问题定位。

以gpu为参照定位精度问题，使用ptdbg_ascend插件来进行数据dump和比对，请参考[ptdbg_ascend](https://gitee.com/ascend/tools/tree/master/ptdbg_ascend)。

## **前提条件**

已完成PyTorch Adapter插件的编译安装，具体操作请参考[《AscendPyTorch安装指南》](../PyTorch安装指南/PyTorch安装指南.md)。

## **注意事项**

训练脚本中使用torch.utils.data.DataLoader加载数据时，请将num_workers参数设为0。
```python
from torch.utils.data import DataLoader

loader=DataLoader(...,num_workers=0,...)
```

## **精度比对基本原理**

普遍适用的方法是以模型为单位，采用hook机制挂在模型的上。当模型在CPU上进行正向传播时跟踪并dump每一层的数值输入与输出，在反向传播时跟踪并dump每一层的梯度输入值与输出值；同样的当模型在NPU中进行计算时采用相同的方式记录下相应的数据，通过对比dump出的数值，计算余弦相似度和均方根误差的方式,
定位和排查NPU算子存在的计算精度问题。

精度比对工具dump数据说明：在实际使用场景中网络模型通常较大，基于整网全量数据的dump，耗时长且储存文件大。因此内部默认使用部分网络参数和统计量的方式dump数据来提升效率。如需dump全量数据，请将register_hook函数中的sample参数设为False（默认为True）。

![图1：精度比对逻辑图](figures/module_compare.png)

图1即为精度对比的基本逻辑，思路清晰明了，但其中存在较多的细节问题：

1. 需要对大量的变量进行控制，要确保模型结构参数等相同。
2. 相同的模型在不同的硬件设备上进行运算时可能会出现相同的计算会调用不同的底层算子，造成npu算子可能出现的不匹配情形。
3. NPU与CPU/GPU的计算结果误差可能会随着模型的执行不断累积，最终会出现同一个算子因为输入的数据差异较大而无法匹配对比计算精度的情况。

其中细节问题2可能表现如下图2：

![图2：算子映射匹配](figures/op_compare.png)

由于可能会出现融合算子，所以在算子的逐一匹配时可能会出现错误匹配或无法匹配的问题，例如图2中NPU算子npu_op_1与npu_op_2无法和cpu_op_k进行匹配，才去跳过的方式，直到到npu_op_3和cpu_op_3才从新对齐开始匹配。

### **算子匹配条件**

判断运行在cpu和npu上的两个算子是否相同采用的步骤如下：

1. 两个算子的名字是否相同
1. 两个算子的输入输出Tensor数量和各个Tensor的shape是否相同

通常满足以上的两个条件，就认为是同一个算子，成功进行算子的匹配，后续进行相应的计算精度比对。

### **计算精度评价指标**

在进行计算精度匹配时，基本共识为默认CPU或GPU的算子计算结果是准确的，最终比对生成的csv文件中主要包括以下的几个属性：
| Name  | Npu Tensor Dtype | Bench Tensor Dtype | Npu Tensor Shape | Bench Tensor Shape | Cosine | RMSE  | MAPE  |
| :---: | :--------------: | :----------------: | :--------------: | :----------------: | :----: | :---: | :---: |

其中主要使用算子Name、Dtype、Shape用于描述算子的基本特征，Cosine(余弦相似)、RMSE(均方根误差)、MAPE(绝对百分比误差)作为评价计算精度的主要评估指标：

1. 余弦相似度(通过计算两个向量的余弦值来判断其相似度)：

$$
   \Large
   cos(\theta) = \frac{\sum_{i=1}^{n} (\hat{y_{i}} \times y_{i})}
   {\sqrt{\sum_{i=1}^{n}\hat{y_{i}}} \times \sqrt{\sum_{i=1}^{n}y_{i}^{2}}}
$$
当余弦夹角数值越接近于1说明计算出的两个张量越相似，在计算中可能会存在nan，主要由于可能会出现其中一个向量为0

2. 均方根误差(RMSE)：

$$
    \Large
    RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{y_{i}} - y_{i})^{2}}
$$
当均方根误差越接近0表示其计算的平均误差越小

3. 平均绝对百分比误差(MAPE)：

$$
    \Large 
    MAPE = \frac{1}{n} \sum_{i=1}^{n}|\frac{\hat{y_{i}} - y_{i}}{y_{i}}|
$$
绝对百分比误差衡量计算误差的百分比，越接近0越好，但当其中的实际计算结果中存在0时是无法进行计算的

## **快速上手样例参考**
相关函数导入
```python
from torch_npu.hooks import set_dump_path, seed_all, register_hook, wrap_acc_cmp_hook
from torch_npu.hooks.tools import compare
```
相关函数说明：
```python
seed_all(seed=1234)->None
功能:
    设置随机数种子, 固定网络初始化中的随机数
参数:
    seed: 随机数种子
例子:
    >>>seed_all()
    >>>seed_all(1234)
```
```python
set_dump_path(fpath=None)->None
功能:
    设置dump数据文件
参数:
    fpath: dump数据文件
例子:
    >>>set_dump_path("./cpu_module_op.pkl")
    >>>set_dump_path("/home/xxx/npu_module_op.pkl")
```
```python
register_hook(model, hook, **kwargs)->None
功能:
    对模型注入hooks, 开启dump数据功能
参数:
    model: 需要dump数据的网络模型
    hook: 需要注册的函数, 精度比对工具使用wrap_acc_cmp_hook函数
    sample:是否使用数据采样方式dump数据, 默认为True
例子:
    >>>register_hook(model, wrap_acc_cmp_hook)
    >>>register_hook(model, wrap_acc_cmp_hook, sample=False)
    >>>register_hook(model, wrap_acc_cmp_hook, sample=True)
```
```python
compare(pkl_path1, pkl_path2, output_path, shape_flag=False)->None
功能:
    对dump下来的两份数据进行比对
参数:
    pkl_path1: 模型dump数据文件1
    pkl_path2: 模型dump数据文件2
    output_path: 精度比对结果文件输出路径
    shape_flag: False时只使用算子名字进行匹配, True时使用算子名称, 输入输出大小进行匹配
例子:
    >>>compare("./npu_module_op.pkl", "./cpu_module_op.pkl", "./module_result.csv")
```

使用精度比对工具进行模型的精度比对，样例代码如下：

```python
# 根据需要import包
import os
import torch
import torch.nn as nn

from torch_npu.hooks import set_dump_path, seed_all, register_hook, wrap_acc_cmp_hook
from torch_npu.hooks.tools import compare


# 定义一个简单的网络
class ModuleOP(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=2, out_features=2)
        self.linear_2 = nn.Linear(in_features=2, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.linear_1(x)
        x2 = self.linear_2(x1)
        r1 = self.relu(x2)
        return r1

module = ModuleOP()
# 对模型注入forwar和backward的hooks，采样功能默认开启，如需关闭，请使用register_hook(model_cpu, wrap_acc_cmp_hook, sample=False)
register_hook(model_cpu, wrap_acc_cmp_hook) 
seed_all()
x = torch.randn(2, 2)

# cpu上计算，dump数据
set_dump_path("./cpu_module_op.pkl")
out = module(x)
loss = out.sum()
loss.backward()

module.npu()
x = x.npu()
# npu上计算，dump数据
set_dump_path("./npu_module_op.pkl")
out = module(x)
loss = out.sum()
loss.backward()

# 对比dump出的数据精度，生成csv文件
compare("./npu_module_op.pkl", "./cpu_module_op.pkl", "./module_result.csv")

```


使用精度比对工具进行torchvision下现有模型的计算精度比对，整体思路相同，其中cpu和npu的对比思路与npu和npu的对比思路也是相同，以resnet50模型为例代码如下：
```python
import os
import copy
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms

from torch_npu.hooks import set_dump_path, seed_all, register_hook, wrap_acc_cmp_hook
from torch_npu.hooks.tools import compare


# 选取需要的模型
model_cpu = models.resnet50()
model_cpu.eval()
model_npu = copy.deepcopy(model_cpu)
model_npu.eval()

seed_all()
# 对该计算进行hook注入和数据dump，采样功能默认开启，如需关闭，请使用register_hook(model_cpu, wrap_acc_cmp_hook, sample=False)
register_hook(model_cpu, wrap_acc_cmp_hook)
register_hook(model_npu, wrap_acc_cmp_hook)

# 需要根据不同的模型输入和标签生成相应的tensor(或读取实际数据)，损失函数等，如果是随机生成的标签需要保证数据的有效性
inputs = torch.randn(1, 3, 244, 244)
labels = torch.randn(1).long()
criterion = nn.CrossEntropyLoss()

# cpu
set_dump_path("./cpu_resnet50_op.pkl")
output = model_cpu(inputs)
loss = criterion(output, labels)
loss.backward()

# npu
set_dump_path("./npu_resnet50_op.pkl")
model_npu.npu()
inputs = inputs.npu()
labels = labels.npu()
output = model_npu(inputs)
loss = criterion(output, labels)
loss.backward()

# 对比dump出的数据精度，生成csv文件
compare("./npu_resnet50_op.pkl", "./cpu_resnet50_op.pkl", "./resnet50_result.csv")
```
 如需对生成的pkl文件解析读取数据，可参考下列代码
```python
import json

pkl_file=open("./cpu_resnet50_op.pkl",'r') 
tensor_line=pkl_file.readline() # 读取行数据
tensor_data=json.loads(tensor_line)

print(tensor_data[0]) # 算子名称
print(tensor_data[1]) # 对应数据
print(tensor_data[2]) # 数据类型
print(tensor_data[3]) # 数据尺寸

```
基于step粒度控制精度对比数据导出，通过在register_hook方法设置schedule参数控制需要导出数据的step。
* **schedule**用于控制使能的step，入参分别为：**起始step(begin=0)、终止step(end=sys.maxsize)、步长(stride=1)**，例：  
schedule() 使能所有step；  
schedule(4, 10, 2) 使能的step为：4,6,8,10；  
schedule(begin=5, stride=3) 使能的step为：5,8,11,14...；  
schedule(end=10, stride=3) 使能的step为：0,3,6,9。
* 变量step_num用于记录step数，调用**step_schedule.step**使变量step_num加1，step_num初始值为0;**step_schedule.reset_step**初始化step_num为0。
```python
from torch_npu.hooks import register_hook, wrap_acc_cmp_hook, schedule, step_schedule

# add step schedule, begin=4 end=10 stride=2
register_hook(model, wrap_acc_cmp_hook, schedule=schedule(4, 10, 2))

for epoch in range(num_train_epochs):
    # init step_num as 0
    step_schedule.reset_step()
    for step, inputs in enumerate(dataloader):
        output = model(inputs)
        loss = output.sum()
        loss.backward()
        # step_num increase 1
        step_schedule.step()
```
异步算子数据导出wrap_async_datadump_hook（仅应用于npu算子导出）；针对精度对比工具同步tensor数据导出，npu同cpu数据同步耗时较长的问题，利用npu上tdtchannel实现tensor数据的异步导出。
* 通过register_hook方法注册异步导出hook wrap_async_datadump_hook，并通过path参数指定导出tensor数据的路径。
* 导出的文件名为：唯一索引_算子名_前反向_输入输出_shape_stride_offset_format.npy，例：16_Functional_relu_backward_input0_shape[2,1]_stride[1,1]_offset[0]_format[2].npy；导出的文件可以使用numpy.load方便导入。
* wrap_async_datadump_hook依赖OutfeedEnqueueOpV2算子，该算子不支持runtime2.0；需设置环境变量规避**export RUNTIME_V2_BLACKLIST=OutfeedEnqueueOpV2**。
```python
from torch_npu.hooks import register_hook, wrap_async_datadump_hook, schedule, step_schedule

# register wrap_async_datadump_hook and add datadump path
register_hook(module, wrap_async_datadump_hook, path='./output', schedule=schedule(1, 2))

# load datadump data by numpy
numpy.load('./output/16_Functional_relu_backward_input0_shape[2,1]_stride[1,1]_offset[0]_format[2].npy')
```


- [**PyTorch溢出检测工具使用指南**](#pytorch精度对比工具使用指南)
  - [**使用场景**](#使用场景)
  - [**前提条件**](#前提条件)
  - [**基本原理**](#基本原理)
  - [**使用方法**](#使用方法)
  - [**快速上手样例参考**](#快速上手样例参考)


## **使用场景**

在模型训练或调测中，遇到溢出问题，定位费时费力，可以使用溢出检测工具，快速定位溢出层位置。

溢出检测工具，通过在pytorch模型中注入hook，监测网络每一层是否有溢出问题，从而进行问题的精准定位。

## **前提条件**

已完成PyTorch Adapter插件的编译安装，具体操作请参考[《AscendPyTorch安装指南》](../PyTorch安装指南/PyTorch安装指南.md)。

## **基本原理**
采用hook机制，对模型每一层执行前向和反向的过程调用溢出检测函数。溢出检测函数通过设定全局溢出标志位来判断是否有数值溢出情况，若有溢出，则会抛出error，输出对应api层名字，且在当前目录下自动生成含有溢出信息的文件。

## **使用方法**
步骤1：引入库代码
```python
from torch_npu.hooks import register_hook, wrap_checkoverflow_hook
```
步骤2：对模型注册溢出检测函数hook
```python
register_hook(model,wrap_checkoverflow_hook)
```

## **快速上手样例参考**
使用溢出检测工具进行模型的溢出问题定位，样例代码如下：


```python
import os
import copy
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms

#import溢出检测工具相关函数
from torch_npu.hooks import register_hook, wrap_checkoverflow_hook

# 需要定位问题的网络模型
model = models.resnet50()
model.npu()

#对模型注册溢出检测函数hook
register_hook(model,wrap_checkoverflow_hook)

inputs = torch.randn(1, 3, 244, 244)
labels = torch.randn(1).long()
criterion = nn.CrossEntropyLoss()

inputs = inputs.npu()
labels = labels.npu()
output = model_npu(inputs)
loss = criterion(output, labels)
loss.backward()

