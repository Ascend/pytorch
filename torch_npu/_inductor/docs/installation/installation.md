# 安装指南

## 2.1 兼容性说明

目前Inductor-Ascend暂未提供独立软件包，而是作为TorchNPU的子目录（与PyTorch社区相同），随着TorchNPU包一起发布。请先安装TorchNPU插件，并按2.2节的依赖安装说明安装Triton-Ascend的依赖后使用Inductor-Ascend。
TorchNPU的安装操作具体参考《[TorchNPU 软件安装指南](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_description.md)》，请保证与CANN相关包的版本匹配（参见《[版本说明](https://www.hiascend.com/document/detail/zh/Pytorch/730/releasenote/docs/zh/release_notes/release_notes.md)》），否则功能可能无法正常使用。

### PyTorch版本支持

当前支持 PyTorch 2.7.1、PyTorch 2.10.0 和 PyTorch 2.13.0 版本。

## 2.2 依赖安装

### Triton-Ascend（TA）依赖

Inductor-Ascend会生成、编译并执行Triton-Ascend算子，如果使用Inductor-Ascend后端，必须先安装Triton-Ascend（TA）。

**安装指南：** [Triton-Ascend安装指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/installation_guide.md)

### Catlass依赖

如果需要使用Inductor-Ascend的CV融合能力，需要安装Catlass。

**安装指南：** [Catlass快速开始指南](https://gitcode.com/cann/catlass/blob/master/docs/zh/1_Practice/01_quick_start.md)
