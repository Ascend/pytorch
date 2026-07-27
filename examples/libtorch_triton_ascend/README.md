# libtorch_triton_ascend

## 环境约束

1）**TorchNPU 和 CANN 环境**

本样例依赖 TorchNPU 运行环境，推荐 TorchNPU 26.1.0 及配套 CANN 9.1.0 版本。

2）**triton-ascend >= 3.2.2**

在 https://github.com/triton-lang/triton-ascend/releases 获取最新whl包。或拉取[源码](https://github.com/triton-lang/triton-ascend/tree/release/3.2.2-dev)编译，依赖pr https://github.com/triton-lang/triton-ascend/pull/246

3）**pybind11**

```bash
pip install pybind11
```

## 跑demo

demo.cpp 包含三个示例：

1. **Add Kernel (explicit grid)** - 使用显式 grid 的向量加法 kernel，展示最基本的 Triton kernel 调用方式
2. **Add Kernel (C++ lambda grid)** - 使用 C++ lambda 函数计算 grid，类似 Triton 的 `lambda meta`，完全在 C++ 端解析无需 Python 参与
3. **LayerNorm Kernel** - LayerNorm 算子示例，展示更复杂的 kernel 参数传递和计算

运行流程：

1）generate libtriton_runtime.so

```bash
cd triton_runtime
bash build.sh
cd ..
```

2）run demo

```bash
bash examples/run_demo.sh
```

3）期望输出

```text
...
=== LayerNorm Kernel ===
[KernelRegistry] 'layer_norm_kernel': def layer_norm_kernel(output_ptr, input_ptr, weight_ptr, bias_ptr, n_rows: [constexpr], n_cols: [constexpr], stride, eps, BLOCK_SIZE: [constexpr])
C++ BLOCK_SIZE: 512
C++ triton  out[0,:5]:  0.1781
-1.6806
-1.2184
-0.3596
-0.9843
[ npuFloatType{5} ]
C++ torch   ref[0,:5]:  0.1781
-1.6806
-1.2184
-0.3596
-0.9843
[ npuFloatType{5} ]
C++: max diff: 7.15256e-07
C++: match: YES
  registered: layer_norm_kernel
  registered: add_kernel
```

4）打开环境变量查看详细日志

```bash
export LOG_TORCH_TRITON_RUNTIME=1
```

## FAQ

1）**编译报错 unsupported relocation，python环境中没有 libpython\*.so**

本样例会链接 `libpython*.so`，要求 Python 安装目录下存在该共享库（即 CPython 编译时须启用 `--enable-shared`）。

检查方法：

```bash
ls $(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython*.so
```

如找不到当前 Python 版本的 `libpython*.so`，或仅有 `libpython*.a` 静态链接库，则可能会产生编译报错。

修复方法：使用 `apt install -y python3-dev` 安装python-dev包，或使用Conda创建python环境
