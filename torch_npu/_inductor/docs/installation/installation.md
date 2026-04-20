# 安装指南

## 2.1 兼容性说明

目前Inductor Ascend暂未提供独立软件包，而是作为Ascend Extension for PyTorch的子目录（与PyTorch社区相同），随着torch_npu包一起发布。请直接安装torch_npu插件，即可使用Inductor Ascend。
torch_npu的安装操作具体参考《[Ascend Extension for PyTorch 软件安装指南](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_description.md)》，请保证与CANN相关包的版本匹配（参见《[版本说明](https://www.hiascend.com/document/detail/zh/Pytorch/730/releasenote/docs/zh/release_notes/release_notes.md)》），否则功能可能无法正常使用。
需要注意的是，当安装的torch_npu版本为7.3.0及之后版本，均可正常使用Inductor Ascend，对于其他torch_npu版本请参见对应版本文档中的安装介绍。


### PyTorch版本支持

当前仅支持 PyTorch 2.7.1 版本。

## 2.2 依赖安装

### Triton Ascend (TA)依赖

Inductor Ascend会生成、编译并执行Triton Ascend算子，如果使用Inductor Ascend后端，必须先安装Triton Ascend（TA）。

**安装指南：** [Triton Ascend安装指南](https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/installation_guide.md)

### Catlass依赖

如果需要使用Inductor Ascend的CV融合能力，需要安装catlass。

**安装指南：** [Catlass快速开始指南](https://gitcode.com/cann/catlass/blob/master/docs/1_Practice/01_quick_start.md)
