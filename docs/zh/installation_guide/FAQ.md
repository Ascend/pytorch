# FAQ

## 编译问题

### torch.libs/libopenblasp-r0-56e95da7.3.24.so链接报错或libgfortran缺失

**问题描述**

aarch64环境下进行libtorch推理测试，编译时依赖torch.libs/\*.so库，需要手动加载。

- 报错截图

    ![](../figures/libtorch_error.png)

- 报错文本

    ```text
    [100%] Linking CXX executable libtorch_resnet
    /usr/bin/ld: warning: libgfortran-b6d57c85.so.5.0.0, needed by /usr/local/python3.8.5/lib/python3.8/site-packages/torch/lib/../../torch.libs/libopenblasp-r0-56e95da7.3.24.so, not found (try using     -rpath or -rpath-link)
    /usr/bin/ld: /usr/local/python3.8.5/lib/python3.8/site-packages/torch/lib/../../torch.libs/libopenblasp-r0-56e95da7.3.24.so: undefined reference to `_gfortran_concat_string@GFORTRAN_8'
    /usr/bin/ld: /usr/local/python3.8.5/lib/python3.8/site-packages/torch/lib/../../torch.libs/libopenblasp-r0-56e95da7.3.24.so: undefined reference to `_gfortran_etime@GFORTRAN_8'
    collect2: error: ld returned 1 exit status
    make[2]: *** [CMakeFiles/libtorch_resnet.dir/build.make:101: libtorch_resnet] Error 1
    make[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/libtorch_resnet.dir/all] Error 2
    make: *** [Makefile:91: all] Error 2
    ```

**处理方法**

在CMakeLists.txt编译文件中增加torch.libs/\*.so库的链接，代码示例如下：

```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 在编译链接阶段增加torch.libs/*.so库的搜索路径，请根据实际情况替换以下命令行中的该库的路径
link_directories(/usr/local/python3.8.5/lib/python3.8/site-packages/torch.libs)  

add_executable(libtorch_resnet libtorch_resnet.cpp)
target_link_libraries(libtorch_resnet "${TORCH_LIBRARIES}")
target_link_libraries(libtorch_resnet "${TORCH_NPU_LIBRARIES}")
```

### 编译时third_party目录下项目文件缺失或需切换项目commit ID

**问题描述**

编译构建时，子模块缺失或需要更换子模块版本。

报错文本

```text
Traceback (most recent call last):
  File "/opt/_internal/cpython-3.9.21/lib/python3.9/runpy.py", line 197, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/opt/_internal/cpython-3.9.21/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/pytorch/torchnpugen/gen_backend_stubs.py", line 948, in <module>
    main()
  File "/home/pytorch/torchnpugen/gen_backend_stubs.py", line 400, in main
    run(options.source_yaml, options.output_dir, options.dry_run,
  File "/home/pytorch/torchnpugen/gen_backend_stubs.py", line 823, in run
    merge_custom_yaml(source_yaml, op_plugin_yaml_path)
  File "/home/pytorch/torchnpugen/utils.py", line 153, in merge_custom_yaml
    PathManager.check_directory_path_readable(op_plugin_path)
  File "/home/pytorch/torchnpugen/utils.py", line 94, in check_directory_path_readable
    cls.check_path_owner_consistent(path)
  File "/home/pytorch/torchnpugen/utils.py", line 80, in check_path_owner_consistent
    raise RuntimeError(msg)
RuntimeError: The path does not exist: /home/pytorch/third_party/op-plugin/op_plugin/config/v2r7/op_plugin_functions.yaml
```

**处理方法**

在PyTorch目录下执行以下命令初始化并更新所有子模块：

```bash
git submodule update --init --recursive
```

如需切换某个third_party项目的 commit ID，进入对应项目目录后执行：

```bash
git checkout <commit_id>
```

### 编译时换行符报错

**问题描述**

编译时无法识别换行符。

报错文本

```text
ci/build.sh: line 2: $'\r': command not found
: invalid optione 3: set: -
set: usage: set [-abefhkmnptuvxBCHP] [-o option-name] [--] [arg ...]
ci/build.sh: line 4: $'\r': command not found
ci/build.sh: line 9: $'\r': command not found
ci/build.sh: line 11: syntax error near unexpected token `$'{\r''
'i/build.sh: line 11: `function parse_script_args() {
```

**报错原因**

Windows换行符问题 。文件使用了Windows风格的换行符（CRLF: \r\n ），而Linux bash只认Unix风格的换行符（LF: \n ），导致“\r”被当作命令的一部分执行。

**处理方法**

可以尝试使用dos2unix工具解决。

```bash
# 安装 dos2unix 工具
yum install -y dos2unix

# 批量转换文本文件（根据自身需求和报错内容选择需要转换的文件）
find /home/pytorch -type f \
  \( -name "*.sh" -o -name "*.py" -o -name "*.cpp" -o -name "*.h" \
  -o -name "*.c" -o -name "*.cmake" -o -name "CMakeLists.txt" \
  -o -name "configure" -o -name "*.txt" -o -name "*.yaml" -o -name "*.yml" \
  -o -name "*.md" -o -name "*.cfg" -o -name "*.in" \) \
  -exec dos2unix {} + 2>/dev/null

# 清理之前的构建缓存（CMake 缓存可能保留了错误配置）
rm -rf /home/pytorch/build

# 重新编译
bash ci/build.sh
```

### 编译时报错CMake\_minimum\_required

**问题描述**

编译时报错CMake版本不符。

报错文本

```text
CMake Error at third_party/Tensorpipe/third_party/libuv/CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 has been removed from CMake.
```

**报错原因**

这意味着您容器中的CMake版本较新（≥ 3.27），从CMake3.27开始，不再兼容cmake_minimum_required低于3.5的项目，直接报错退出。

**处理方法**

您可以在setup.py文件中`class CPPLibBuild`的`run`函数里添加`cmake_args.append('-DCMAKE_POLICY_VERSION_MINIMUM=3.5')`来尝试解决，让CMake以兼容模式处理旧版cmake_minimum_required声明。

### 编译时报错链接符问题

**问题描述**

编译时报错无法正确识别链接符。

报错文本

```text
/home/pytorch/third_party/torchair/torchair/third_party/ascend/include/ascendcl/external/acl/error_codes/rt_error_codes.h:1:1: error: expected unqualified-id before ‘.’ token
```

**报错原因**

该文件原先应是符号链接，在Windows上，Git自动将其转成了包含路径的文本文件。

**处理方法**

```bash
# 修复损坏的符号链接（找到内容为相对路径的文件，重建为符号链接）
find /home/pytorch -type f -name "*.h" -exec grep -l '^\.\./' {} \; 2>/dev/null | while read f;do
    target=$(cat "$f")
    ln -sf "$target" "$f"
done
```

或尝试用实际内容替换：

譬如，用`third_party\torchair\torchair\third_party\ascend\include\air\external\ge\ge_error_codes.h`中内容替换`third_party\torchair\torchair\third_party\ascend\include\ascend/include/ascendcl/external/acl/error_codes/rt_error_codes.h`中的内容。

### fatal error: 'filesystem' file not found

**问题描述**

编译时报错“filesystem”文件缺失。

报错文本

```text
fatal error: 'filesystem' file not found
```

**报错原因**

此错误通常由GCC版本过低引起。请执行以下命令查询当前GCC版本：

```bash
gcc --version
```

**处理方法**

如果GCC版本低于8，请参考《[安装GCC 11.2.0版本](installing_gcc_11-2-0.md)》安装GCC 8及以上版本。

## 安装问题

### 构建完成的whl包与当前使用环境不匹配

**问题描述**

编译完成后，无法安装对应的whl包。

报错文本

```text
ERROR: torch_npuxxx.whl is not a supported wheel on this platform
```

**报错原因**

构建时的Python环境与安装时的Python环境不匹配。

**处理方法**

编译安装包前请确认目标环境所需的Python版本，在编译构建时通过`--python`参数指定对应的Python版本：

```bash
bash ci/build.sh --python=3.xx
```

### ImportError: libhccl.so: cannot open shared object file: No such file or directory

**问题描述**

导入torch_npu时，系统报错缺失libhccl.so文件。

报错文本

```text
ImportError: libhccl.so: cannot open shared object file: No such file or directory
```

**报错原因**

当前环境未满足运行torch_npu的条件，编译构建时torch_npu依赖torch，而运行时依赖配套的NPU驱动固件与CANN软件。

**处理方法**

请检查是否已安装配套版本的 NPU 驱动固件、CANN 软件（Toolkit、ops 和 NNAL）并正确配置 CANN 环境变量，具体请参考《[CANN 软件安装](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)》（商用版）或《[CANN 软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)》（社区版）。

### 导入torch_npu时，系统报错core dump

**问题描述**

编译安装torch_npu后，执行“import torch_npu”，系统报错core dump。

报错文本

```text
Segmentation fault
(core dumped)
```

**报错原因**

编译构建时GCC版本不符合预期。pybind存在abi校验，不同GCC版本的abi_version不一致，导致校验失败。

**处理方法**

使用对应的GCC版本进行编译，具体对应版本可参考[GCC和Cmake版本要求](compilation_installation_using_source_code.md#gcc_cmake)。

### “import torch_npu”报错找不到torch_npu._C

**问题描述**

安装torch_npu后，“import torch_npu”报错torch_npu._C。

报错文本

```text
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/pytorch/torch_npu/__init__.py", line 58, in <module>
    import torch_npu.utils.patch_getenv
  File "/home/pytorch/torch_npu/utils/__init__.py", line 12, in <module>
    from torch_npu.npu.utils import get_cann_version
  File "/home/pytorch/torch_npu/npu/__init__.py", line 158, in <module>
    from .utils import (obfuscation_initialize, obfuscation_calculate, obfuscation_finalize, 
  File "/home/pytorch/torch_npu/npu/utils.py", line 11, in <module>
    import torch_npu._C
ModuleNotFoundError: No module named 'torch_npu._C'
```

**报错原因**

因为安装的torch_npu与该项目下的文件夹重名，不能在项目目录下运行“import torch_npu”。

**处理方法**

进入合适的运行目录下重试，如先`cd test`或`cd /home/test`后再“import torch_npu”。
