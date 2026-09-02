# 快速安装

由于master分支在持续更新迭代中，因此未提供软件包形式的快速安装方式，请使用[源码编译](./references/building_from_source.md)进行安装操作。

TorchNPU插件针对不同的开发场景提供了相应的扩展模块，您可以按照需求安装不同的扩展模块：

- 如需使用C++接口进行开发，请参考[编译libtorch_npu](./references/building_libtorch_npu.md)。
- 如需开展计算机视觉任务，请参考[安装torchvision](./references/installing_torchvision.md)。

> [!NOTICE]
>
> 为跟随PyTorch社区的发布节奏，提供针对PyTorch 2.13.0版本的即时适配版本，可前往[下载](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=174&ids=89dda9ba9de741349efa03687a487678%2C202%2C200%2C1%2C6%2C177%2C)页面获取。该版本是基于master分支的较早快照，文档中部分新特性可能暂不可用，具体支持情况请参考[Release](https://gitcode.com/Ascend/pytorch/releases/compatibility_release-pytorch2.13.0)。
