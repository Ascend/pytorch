# 编译加速

本文介绍TorchNPU源码构建的编译加速方法，涵盖Ninja、Mold、CCache的使用及按需编译目标的方法。在使用之前，请参考[源码编译](building_from_source.md)完成环境准备。

> [!NOTE]
>
> 以下操作命令均需在TorchNPU源码根目录执行。

## 使用Ninja构建

CMake默认使用Makefile生成器。安装Ninja构建系统可以显著加快编译速度。

本项目`setup.py`会自动检测Ninja环境，若系统PATH中包含ninja命令，或环境变量`CMAKE_GENERATOR`设置为ninja，则自动启用Ninja构建。

安装命令如下：

```bash
pip install ninja
```

安装完成后，无需额外配置，后续编译即可自动生效。若此前已执行编译，安装Ninja后需要清理构建缓存。

清理构建缓存命令如下：

```bash
python setup.py clean
```

## 使用Mold链接器

在频繁修改代码并重新编译的开发循环中，链接阶段会成为性能瓶颈。相比Linux发行版默认的GNU ld链接器，Mold链接器具有显著的速度优势，能大幅度缩短构建时间，提升开发体验。

本项目的`CMakeLists.txt`文件已集成了链接器自动检测逻辑，优先检测Mold链接器，若存在则自动通过`-fuse-ld=mold`参数来启动。

安装命令如下：

```bash
sudo apt install mold
# 或从源码安装：https://github.com/rui314/mold
```

安装成功后，重新编译即可自动生效。若需确认链接器是否正确启用，可检查编译输出中的链接选项是否包含`-fuse-ld=mold`。

## 使用CCache

尽管CMake的依赖跟踪机制能识别文件变化，但是某些场景下，仍可能触发重复编译。引入CCache可以有效避免重复编译，显著减少重复编译时间。

本项目的`CMakeLists.txt`文件已集成CCache自动检测逻辑，安装CCache后即可自动启用。为了获得最佳效果，建议根据实际环境调整CCache配置，例如如缓存目录、缓存大小、压缩等。

安装命令如下：

```bash
sudo apt install ccache
# 或
sudo yum install ccache
```

验证CCache是否生效：

1. 执行一次完整编译。
2. 再次执行相同的完整编译命令。第二次编译应明显快于第一次，因为大部分目录文件将从缓存中直接读取。

如果编译速度没有提升，请检查编译缓存文件（`build/CMakeCache.txt`）中的`CMAKE_C_COMPILER_LAUNCHER`和`CMAKE_CXX_COMPILER_LAUNCHER`变量是否包含CCache：

```cmake
//C compiler launcher
CMAKE_C_COMPILER_LAUNCHER:PATH=/usr/bin/ccache

//CXX compiler launcher
CMAKE_CXX_COMPILER_LAUNCHER:PATH=/usr/bin/ccache
```

## 仅编译所需目标

如果只需重新构建`torch_npu.so`，为避免全量编译耗时，可仅编译目标文件。

进入构建目录及编译命令如下：

```bash
cd build && ninja torch_npu
```

若环境中未安装Ninja，可将命令中的ninja替换为make。
