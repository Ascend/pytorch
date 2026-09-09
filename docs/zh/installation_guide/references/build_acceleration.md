# 编译加速

本文介绍 TorchNPU 源码构建中的编译加速方法，包括使用 Ninja、Mold、CCache 和仅编译所需目标。请先参考[源码编译](../building_from_source.md)完成环境准备，以下命令均从 TorchNPU 源码根目录开始执行。

## 使用 Ninja 构建

默认情况下，CMake 使用 Makefile 生成器。安装 Ninja 构建系统可以显著加快编译速度。

本项目 `setup.py` 会自动检测系统中是否安装了 Ninja：如果环境变量 `CMAKE_GENERATOR` 设置为 `ninja`，或者 `ninja` 命令在 `PATH` 中可用，将自动使用 Ninja 作为构建系统。

```bash
pip install ninja
```

安装 Ninja 后，编译即可自动生效，无需额外配置。如果之前已经编译过，安装 Ninja 后需要先执行一次清理：

```bash
python setup.py clean
```

## 使用 Mold 链接器

在频繁修改单个文件并重新编译的开发循环中，链接时间会占据主导。大多数 Linux 发行版自带的系统链接器（GNU `ld`）速度较慢，使用更快的链接器可以显著改善构建体验。

本项目的 `CMakeLists.txt` 已内置链接器自动检测逻辑：优先检测 mold 链接器，若存在则自动启用（`-fuse-ld=mold`）。

```bash
sudo apt install mold
# 或从源码安装：https://github.com/rui314/mold
```

安装后重新编译即可自动生效。若需确认链接器是否正确启用，可检查编译输出中的链接选项是否包含 `-fuse-ld=mold`。

## 使用 CCache

即使依赖跟踪基于文件修改时间，仍有许多场景下文件会被重复编译。使用 ccache 可以有效避免重复编译，节省大量时间。

本项目的 `CMakeLists.txt` 已内置 ccache 自动检测逻辑，安装 ccache 后即可自动启用。但建议根据自身环境调整 ccache 配置（如缓存目录、缓存大小、压缩等）以获得最佳效果：

```bash
sudo apt install ccache
# 或
sudo yum install ccache
```

验证 ccache 是否生效：连续执行两次完整编译，第二次应明显快于第一次。如果未生效，可检查 `build/CMakeCache.txt` 中的 `CMAKE_C_COMPILER_LAUNCHER` 和 `CMAKE_CXX_COMPILER_LAUNCHER` 变量是否包含 ccache：

```cmake
//C compiler launcher
CMAKE_C_COMPILER_LAUNCHER:PATH=/usr/bin/ccache

//CXX compiler launcher
CMAKE_CXX_COMPILER_LAUNCHER:PATH=/usr/bin/ccache
```

## 仅编译所需目标

如果只需重新构建 `torch_npu.so`，可以在 build 目录下直接指定目标，避免全量构建：

```bash
cd build && ninja torch_npu
```

如果未安装 Ninja，将 `ninja` 替换为 `make` 即可。
