# Build Acceleration

This document describes the build acceleration methods for building TorchNPU from source code, covering the use of Ninja, Mold, and CCache, and the method of compiling only required targets. Before using these methods, complete environment preparation by referring to [Building from Source](../building_from_source.md).

> [!NOTE]
>
> All the following commands must be executed in the root directory of the TorchNPU source code.

## Building with Ninja

CMake uses the Makefile generator by default. Installing the Ninja build system can significantly speed up compilation.

The `setup.py` of this project automatically detects the Ninja environment. If the `ninja` command is included in the system PATH, or the environment variable `CMAKE_GENERATOR` is set to ninja, Ninja build is automatically enabled.

The installation command is as follows:

```bash
pip install ninja
```

After the installation is complete, no additional configuration is required, and subsequent compilation takes effect automatically. If you have compiled before, you need to clean the build cache after installing Ninja.

The command to clean the build cache is as follows:

```bash
python setup.py clean
```

## Using the Mold Linker

In a development loop where code is frequently modified and recompiled, the linking stage can become a performance bottleneck. Compared with the GNU ld linker used by Linux distributions by default, the Mold linker provides a significant speed advantage, greatly shortening the build time and improving the development experience.

The `CMakeLists.txt` file of this project has integrated automatic linker detection logic, which detects the Mold linker first. If it exists, the Mold linker is automatically started using the `-fuse-ld=mold` parameter.

The installation command is as follows:

```bash
sudo apt install mold
# Or install it from the source: https://github.com/rui314/mold
```

After a successful installation, recompilation takes effect automatically. To confirm whether the linker is correctly enabled, check whether the link options in the compilation output contain `-fuse-ld=mold`.

## Using CCache

Although CMake's dependency tracking mechanism can identify file changes, repeated compilation may still be triggered in some scenarios. Introducing CCache can effectively avoid repeated compilation and significantly reduce the time of repeated compilation.

The `CMakeLists.txt` file of this project has integrated automatic CCache detection logic, which is automatically enabled after CCache is installed. For the best results, adjust the CCache configuration based on the actual environment, such as the cache directory, cache size, and compression.

The installation command is as follows:

```bash
sudo apt install ccache
# Or
sudo yum install ccache
```

Verify whether CCache takes effect:

1. Perform a complete compilation.
2. Run the same complete compilation command again. The second compilation should be noticeably faster than the first, because most files will be read directly from the cache.

If the compilation speed has not improved, check whether the `CMAKE_C_COMPILER_LAUNCHER` and `CMAKE_CXX_COMPILER_LAUNCHER` variables in the build cache file (`build/CMakeCache.txt`) contain CCache:

```cmake
//C compiler launcher
CMAKE_C_COMPILER_LAUNCHER:PATH=/usr/bin/ccache

//CXX compiler launcher
CMAKE_CXX_COMPILER_LAUNCHER:PATH=/usr/bin/ccache
```

## Compiling Only Required Targets

If you only need to rebuild `torch_npu.so`, you can compile only the target file to avoid time-consuming full compilation.

The command to enter the build directory and compile is as follows:

```bash
cd build && ninja torch_npu
```

If Ninja is not installed in the environment, replace ninja with make in the command.
