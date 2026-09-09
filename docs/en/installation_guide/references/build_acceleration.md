# Build Acceleration

This document describes how to speed up TorchNPU source builds using Ninja, Mold, CCache, and target-specific builds. Prepare the build environment by following [Building from Source (Chinese)](../../../zh/installation_guide/building_from_source.md). Run the commands below from the TorchNPU source root.

## Build with Ninja

By default, CMake uses the Makefile generator. Installing the Ninja build system can significantly speed up compilation.

This project`setup.py`Automatically detects whether Ninja is installed on the system: If the environment variable`CMAKE_GENERATOR`Set to`ninja`, or`ninja`Ordered in the`PATH`available in, will automatically use Ninja as the build system.

```bash
pip install ninja
```

After the Ninja is installed, the compilation takes effect automatically. No additional configuration is required. If the Ninja has been compiled, perform the following operations after installing the Ninja:

```bash
python setup.py clean
```

## Using Mold Linker

In a development cycle where individual files are frequently modified and recompiled, link time dominates. The system linker (GNU) that comes with most Linux distributions `ld` Slower, using a faster linker can significantly improve the build experience.

This project's`CMakeLists.txt`Built-in linker automatic detection logic: The mold linker is detected first, and if it exists, the mold linker is automatically enabled.`-fuse-ld=mold`.

```bash
sudo apt install mold
#Alternatively, install the software from the source code: https://github.com/rui314/mold
```

After the installation, recompile the software automatically takes effect. To confirm that the linker is enabled correctly, check whether the link option in the compilation output contains`-fuse-ld=mold`.

## Using the CCache

Even if dependency tracking is based on file modification time, there are many scenarios where files are repeatedly compiled. Using ccache can effectively avoid repeated compilation and save a lot of time.

This project's`CMakeLists.txt`The built-in ccache automatic detection logic is automatically enabled after the ccache is installed. However, you are advised to adjust the ccache configuration (such as the cache directory, cache size, and compression) based on the environment to obtain the best results.

```bash
sudo apt install ccache
#or the
sudo yum install ccache
```

Check whether the ccache takes effect by performing two complete compilations in a row. The second compilation should be significantly faster than the first compilation. If not, check the`build/CMakeCache.txt`medium`CMAKE_C_COMPILER_LAUNCHER`And to the`CMAKE_CXX_COMPILER_LAUNCHER`Whether the variable contains ccache:

```cmake
//C compiler launcher
CMAKE_C_COMPILER_LAUNCHER:PATH=/usr/bin/ccache

//CXX compiler launcher
CMAKE_CXX_COMPILER_LAUNCHER:PATH=/usr/bin/ccache
```

## Compile only the required targets

If you just have to rebuild`torch_npu.so`, you can directly specify the target in the build directory to avoid full build.

```bash
cd build && ninja torch_npu
```

If Ninja is not installed, the`ninja`Replace with`make`That's it.
