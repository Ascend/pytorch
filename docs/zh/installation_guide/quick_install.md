# 快速安装

由于master版本在持续更新迭代中，因此未提供软件包形式的快速安装方式，请使用[源码编译](./references/building_from_source.md)进行安装操作。

> [!NOTICE]
>
> 如果您希望使用基于最新PyTorch 2.13.0发布的即时适配版本，可前往[下载](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=174&ids=89dda9ba9de741349efa03687a487678%2C202%2C200%2C1%2C6%2C177%2C)页面获取。
>
> 请注意，该版本是基于master分支的较早快照，文档中部分新特性可能暂不可用，具体支持情况请参考[Release](https://gitcode.com/Ascend/pytorch/releases)。
>
> 如果使用最新的master版本，仍需通过[源码编译](./references/building_from_source.md)安装。

对于大多数用户，安装PyTorch框架和TorchNPU插件后即可满足基本的训练与推理需求。但是，在特定开发场景下，您可能还需要安装相应的扩展模块。例如，如需使用C++接口进行开发，请参见[编译libtorch_npu](./references/building_libtorch_npu.md)；如需开展计算机视觉任务，请参见[安装torchvision](./references/installing_torchvision.md)。
