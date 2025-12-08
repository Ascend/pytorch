# Ascend Extension for PyTorch是什么

Ascend Extension for PyTorch插件是基于昇腾的深度学习适配框架，使昇腾NPU可以支持PyTorch框架，为PyTorch框架的使用者提供昇腾AI处理器的超强算力。

项目源码地址请参见[Link](https://gitcode.com/Ascend/pytorch)。

昇腾为基于昇腾处理器和软件的行业应用及服务提供全栈AI计算基础设施。您可以通过访问[昇腾社区](https://www.hiascend.com/zh/)，了解关于昇腾的更多信息。

## 总体架构

Ascend Extension for PyTorch整体架构如下所示。

**图 1** Ascend Extension for PyTorch整体架构  
![](figures/architecture_torch_npu.png "Ascend-Extension-for-PyTorch整体架构")

-   Ascend Extension for PyTorch（即torch\_npu插件）：昇腾PyTorch适配插件，继承开源PyTorch特性，针对昇腾AI处理器系列进行深度优化，支持用户基于PyTorch框架实现模型训练和调优。
-   PyTorch原生库/三方库适配：适配支持PyTorch原生库及主流三方库，补齐生态能力，提高昇腾平台易用性。

## 关键功能特性

-   适配昇腾AI处理器：基于开源PyTorch，适配昇腾AI处理器，提供原生Python接口。
-   框架基础功能：PyTorch动态图、自动微分、Profiling、优化器等。
-   自定义算子开发：支持在PyTorch框架中添加自定义算子。
-   分布式训练：支持原生分布式数据并行训练，包含单机多卡、多机多卡场景支持的集合通信原语，如Broadcast、AllReduce等。
-   模型推理：支持输出标准的ONNX模型可通过离线转换工具将ONNX模型转换为离线推理模型。

## 更多介绍

关于Ascend Extension for PyTorch的更多介绍，可参见在线课程：[Ascend Extension for PyTorch](https://www.hiascend.com/edu/courses?activeTab=Ascend+Extension+for+PyTorch)。

