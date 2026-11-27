# 安装指南

## 2.1 兼容性说明

目前Inductor-Ascend暂未提供独立软件包，而是作为TorchNPU的子目录（与PyTorch社区相同），随着TorchNPU包一起发布。请先安装TorchNPU插件，并按2.2节的依赖安装说明安装Triton-Ascend的依赖后使用Inductor-Ascend。
TorchNPU的安装操作具体参考《[TorchNPU 软件安装指南](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_description.md)》，请保证与CANN相关包的版本匹配（参见《[版本说明](https://www.hiascend.com/document/detail/zh/Pytorch/730/releasenote/docs/zh/release_notes/release_notes.md)》），否则功能可能无法正常使用。

### PyTorch版本支持

当前支持PyTorch 2.10.0和PyTorch 2.13.0版本。

### Python版本支持

Inductor Ascend支持在Python 3.14环境中使用，无需为此新增或修改特性开关。Python 3.14需使用提供cp314安装包的配套版本，当前配套组合如下：

| PyTorch版本 | TorchNPU安装包版本 | Triton-Ascend版本 |
| --- | --- | --- |
| 2.13.0 | 2.13.0 | 3.6.0 |
| 2.10.0 | 2.10.0 | 3.6.0 |

使用其他版本组合时，请先确认PyTorch、TorchNPU、Triton-Ascend及其他第三方依赖均提供Python 3.14安装包并保持版本配套。

可通过以下命令确认Python解释器版本、基础导入和NPU后端注册：

```shell
python --version
python -c "import torch; import torch_npu; print(torch.__version__); print(torch.npu.is_available())"
```

完成上述检查后，可通过一个最小示例确认torch.compile能正常触发NPU编译与执行（首次执行会触发编译，耗时高于后续执行）：

```python
import torch
import torch_npu


@torch.compile
def fn(x):
    return torch.relu(torch.sin(x) + x)


x = torch.randn(1024, 1024, device="npu", dtype=torch.float16)
out = fn(x)
torch.npu.synchronize()
print(out.shape)
```

`torch.npu.is_available()`返回`True`且上述示例正常输出后，可继续按各特性文档中的快速入门示例进行验证。使用多进程编译、DataLoader或自定义worker时，应确保传递给子进程的对象和函数可序列化，避免依赖仅在父进程有效的局部状态。

## 2.2 依赖安装

### Triton-Ascend（TA）依赖

Inductor-Ascend会生成、编译并执行Triton-Ascend算子，如果使用Inductor-Ascend后端，必须先安装Triton-Ascend（TA）。

**安装指南：** [Triton-Ascend安装指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/installation_guide.md)

### Catlass依赖

使用CATLASS后端进行CV融合时，需要安装CATLASS。

**安装指南：** [Catlass快速开始指南](https://gitcode.com/cann/catlass/blob/master/docs/zh/1_Practice/01_quick_start.md)
