# requirements-test.txt 依赖说明文档

> **用途**: 安装在 torch-npu CI Docker 测试镜像中，用于构建和运行上游 PyTorch 测试用例。
>
> **对齐基准**: 上游 PyTorch `.ci/docker/requirements-ci.txt` (py3.10 / jammy profile)
>
> **安装位置**: `.ci/docker/test/Dockerfile.aarch64:78`
> ```dockerfile
> RUN conda run -n py_${PYTHON_VERSION} pip install --no-cache-dir -r /opt/buildtools/requirements-test.txt
> ```

---

## 目录

- [Core test frameworks](#core-test-frameworks)
- [Type checking](#type-checking)
- [Core scientific](#core-scientific)
- [Data & serialization](#data--serialization)
- [ONNX](#onnx)
- [ML / data processing](#ml--data-processing)
- [Image / audio / ML](#image--audio--ml)
- [Utilities](#utilities)
- [Build tools](#build-tools)
- [Solver / optimization](#solver--optimization)
- [Test utilities](#test-utilities)
- [Infrastructure](#infrastructure)

---

## Core test frameworks

| 包名 | 版本 | 上游对应 | 引用位置 | 说明 |
|------|------|---------|---------|------|
| `pytest` | `7.3.2` | ✅ 一致 | 全部 `test/*.py` | 测试框架主体，`test/run_test.py` 核心调度 |
| `pytest-xdist` | `3.3.1` | ✅ 一致 | `test/conftest.py:142`, `test/run_test.py:1361` | 并行测试执行，`-n` 参数 |
| `pytest-flakefinder` | `1.1.0` | ✅ 一致 | `test/run_test.py:547,2289` | 重跑测试以发现 flaky tests |
| `pytest-rerunfailures` | `>=10.3` | ✅ 一致 | `test/conftest.py:42` | 失败用例自动重试 |
| `pytest-subtests` | `0.13.1` | ✅ 一致 | `test/distributed/_composable/test_replicate_mixed_precision.py:73` | 子测试支持 |
| `pytest-cpp` | `2.3.0` | ✅ 一致 | `test/run_test.py:548` | pytest 调用 C++ 测试 |
| `xdoctest` | `1.3.0` | ✅ 一致 | `test/run_test.py:1162-1265` | 运行 docstring 中的 doctest |
| `hypothesis` | `6.56.4` | ✅ 一致 | `test/quantization/fx/test_quantize_fx.py:157`, `test/nn/test_nn.py` 等 | 基于属性的测试生成 |
| `expecttest` | `0.3.0` | ✅ 一致 | `test/test_tensorboard.py:11`, `test/inductor/test_cutedsl_template.py:5`, `test/benchmark_utils/test_benchmark_utils.py:12` 等 | 期望值自动填充测试 |
| `parameterized` | `0.8.1` | ✅ 一致 | `test/onnx/torchlib/test_ops.py:38`, `test/higher_order_ops/test_invoke_subgraph.py:12`, `test/onnx/test_models_onnxruntime.py:9` 等 | 参数化测试 |
| `unittest-xml-reporting` | `<=3.2.0,>=2.0.0` | ✅ 一致 | CI 测试结果 XML 输出 | `lxml` 是其依赖 |

## Type checking

| 包名 | 版本 | 上游对应 | 引用位置 | 说明 |
|------|------|---------|---------|------|
| `mypy` | `1.16.0` | ✅ 一致 | `test/test_type_hints.py:22` `import mypy.api`, `test/test_typing.py:22` `from mypy import api` | 类型检查测试 |

## Core scientific

| 包名 | 版本 | 上游对应 | 引用位置 | 说明 |
|------|------|---------|---------|------|
| `numpy` | `1.23.2` | ✅ 一致 (py3.10) | `test/` 下 277 个文件 import | 核心数值计算库 |
| `scipy` | `1.10.1` | ✅ 一致 (py<=3.11) | `test/` 下 41 个文件 import | 科学计算，`test_linalg.py`, `test_binary_ufuncs.py` 等 |
| `opt-einsum` | `3.3` | ✅ 一致 | `test/test_linalg.py` (einsum 优化) | 张量缩并顺序优化 |
| `sympy` | `1.13.3` | ✅ 一致 | `test/` 下 69 个文件引用 | 符号计算，形状推断、guard 条件 |
| `networkx` | `2.8.8` | ✅ 一致 | `test/functorch/test_aotdispatch.py:134,7088` | 复杂网络库，AOT dispatch 测试 |

## Data & serialization

| 包名 | 版本 | 上游对应 | 引用位置 | 说明 |
|------|------|---------|---------|------|
| `pillow` | `12.3.0` | ✅ 一致 | `test/test_tensorboard.py:118` `from PIL import Image`, `test/onnx/test_models_onnxruntime.py:10` | 图像处理 |
| `protobuf` | `6.33.5` | ✅ 一致 | `test/test_tensorboard.py:117` `from google.protobuf import text_format`, `test/onnx/torchlib/error_reproduction.py:28` | 序列化协议 |
| `dill` | `0.3.7` | ✅ 一致 | `test/test_datapipe.py:593` `@skipIfNoDill` | 序列化库，DataLoader 测试 |
| `flatbuffers` | `24.12.23` | ✅ 一致 | `test/onnx/` 目录 (ONNX 序列化) | 跨平台序列化 |
| `lxml` | `5.3.0` | ✅ 一致 (py<3.14) | `unittest-xml-reporting` 的依赖 | XML 处理，测试报告 |

## ONNX

| 包名 | 版本 | 上游对应 | 引用位置 | 说明 |
|------|------|---------|---------|------|
| `onnx` | `1.21.0` | ✅ 一致 | `test/` 下 57 个文件 import | ONNX 模型格式 |
| `onnx-ir` | `0.1.16` | ✅ 一致 | `onnxscript` 的依赖 | ONNX 内存 IR |
| `onnxruntime` | `1.23.1` | 上游注释掉，通过 `install_onnx.sh` 安装 | **torch-npu**: `test/onnx/test_pytorch_onnx_onnxruntime_npu.py:7`, `test/onnx/dynamo/test_dynamo_with_onnxruntime_backend.py:12`, `test/onnx/onnx_test_common.py:16`, `test/onnx/test_pytorch_jit_onnx.py:2`; **上游**: `test/onnx/` 7 个文件 | ONNX 推理引擎，ONNX 测试必需 |
| `onnxscript` | `0.6.2` | ✅ 一致 | `test/onnx/exporter/test_tensors.py:6`, `test/onnx/exporter/test_api.py:10`, `test/onnx/test_onnxscript_runtime.py:8` | ONNX Script 语言 |

## ML / data processing

| 包名 | 版本 | 上游对应 | 引用位置 | 说明 |
|------|------|---------|---------|------|
| `jinja2` | `3.1.6` | ✅ 一致 | `torch/_inductor/codegen/common.py:2619,2652`, `torch/_inductor/select_algorithm.py:2506`, `torch/distributed/debug/_frontend.py:15` | 模板引擎，Inductor 代码生成 |
| `filelock` | `3.20.3` | ✅ 一致 | `test/inductor/test_caching.py:18`, `torch/_inductor/debug.py:482`, `torch/_inductor/runtime/caching/locks.py:18` | 文件锁，Inductor 缓存 |
| `optree` | `0.13.0` | ✅ 一致 (py<3.14) | `test/` 下 4 个文件 import | 树结构操作，pytree |
| `ml-dtypes` | `0.5.1` | 上游不存在 | **torch-npu**: `test/custom_ops/test_npu_anti_quant.py:3` `from ml_dtypes import int4`; **上游**: `test/onnx/exporter/test_core.py:10` | 机器学习数据类型，NPU 量化算子测试 |
| `transformers` | `4.36.2` | 上游通过 `install_onnx.sh` 安装 `4.36.2` | **torch-npu**: `test/dynamo/test_model_output.py:13-16`, `test/onnx/dynamo/test_dynamo_with_onnxruntime_backend.py:398-399`; **上游**: `test/dynamo/test_model_output.py`, `test/inductor/test_padding.py`, `test/onnx/exporter/test_small_models_e2e.py` | HuggingFace Transformers，模型导出测试 |

## Image / audio / ML

| 包名 | 版本 | 上游对应 | 引用位置 | 说明 |
|------|------|---------|---------|------|
| `scikit-image` | `0.22.0` | ✅ 一致 | `test/test_nn.py` (图像处理) | 图像处理库 |
| `librosa` | `>=0.6.2` | ✅ 一致 (py3.10 不 pin) | `test/test_spectral_ops.py` | 音频分析 |
| `numba` | `0.57.1` | ✅ 一致 (py3.10) | `test/test_numba_integration.py:19` `import numba.cuda` | JIT 编译 |
| `tensorboard` | `2.13.0` | ✅ 一致 (py<3.13) | `test/test_tensorboard.py:17-18,119-122` | 可视化工具 |
| `pywavelets` | `1.4.1` | ✅ 一致 (py<3.12) | `scikit-image` 的依赖 | 小波变换 |

## Utilities

| 包名 | 版本 | 上游对应 | 引用位置 | 说明 |
|------|------|---------|---------|------|
| `tabulate` | `0.9.0` | ✅ 一致 | `test/test_compile_benchmark_util.py:16`, `torch/fx/graph.py:2595`, `torch/ao/quantization/fx/_model_report/model_report_visualizer.py:11` | 表格格式化输出 |
| `psutil` | 无pin | ✅ 一致 | `test/test_cuda.py:26`, `test/distributed/checkpoint/test_state_dict_stager.py:9`, `test/test_dataloader.py:69`, `test/profiler/test_profiler.py:119` | 系统资源监控 |
| `fsspec` | `>=0.8.5` | 上游在 `requirements.txt` 中 | `test/distributed/checkpoint/test_fsspec.py:10-11`, `torch/distributed/checkpoint/_fsspec_filesystem.py:11,23` | 文件系统抽象 |
| `packaging` | `24.2` | ✅ 一致 | `test/` 下 7 个文件 | 包版本管理 |
| `pyyaml` | `6.0.3` | ✅ 一致 | `test/` 下 9 个文件 | YAML 解析 |
| `typing-extensions` | `4.12.2` | ✅ 一致 (py<3.14) | `test/` + `torch/` 下 307 处引用 | 类型提示扩展 |
| `requests` | `2.32.0` | 上游注释掉 | `test/distributed/test_debug.py:13-14`, `test/distributed/test_c10d_fr_hook.py:431,700`, `test/distributed/elastic/test_control_plane.py:194` | HTTP 客户端 |
| `pybind11` | `3.0.1` | 上游通过 `install_triton.sh` 安装 | **torch-npu C++源码**: `torch_npu/csrc/npu/Graph.cpp:716`, `Stream.cpp:1`, `Event.cpp:3`, `Graph.h:4` 等 56 处 `#include <pybind11/...>`; `CMakeLists.txt:304` | C++/Python 绑定，torch_npu 编译必需 |
| `tqdm` | `>=4.66.0` | ✅ 一致 | `test/profiler/test_profiler.py:112`, `test/scripts/run_cuda_memcheck.py:23`, `benchmarks/dynamo/*` | 进度条 |
| `click` | 无pin | ✅ 一致 | `test/cpp/aoti_inference/generate_lowered_cpu.py:3`, `torch/csrc/jit/tensorexpr/scripts/bisect.py:68` | CLI 框架 |
| `pygments` | `2.20.0` | ✅ 一致 | doctest 语法高亮 | 代码高亮 |

## Build tools

| 包名 | 版本 | 上游对应 | 引用位置 | 说明 |
|------|------|---------|---------|------|
| `cmake` | `3.31.6` | ✅ 一致 | `ci/build.sh`, C++ 扩展测试构建 | 构建系统 |
| `ninja` | `1.13.0` | ✅ 一致 | `test/run_test.py`, `test_cpp_extensions_aot.py`, `test_determination.py` | 构建加速 |
| `build` | `1.3.0` | ✅ 一致 | `ci/build.sh` `python -m build --wheel --no-isolation` | Python 构建前端 |
| `pip` | `26.1.2` | ✅ 一致 | 包安装工具 | pip 自身 |
| `setuptools` | `78.1.1` | ✅ 一致 | `ci/build.sh` `python setup.py build bdist_wheel` | 构建系统 |
| `pyzstd` | 无pin | ✅ 一致 | Zstandard 压缩（间接依赖） | 压缩库 |
| `six` | 无pin | ✅ 一致 | `torch/` 间接依赖 (NNPACK -> PeachPy -> six) | Python 2/3 兼容 |
| `wheel` | 无pin | 上游在 `requirements.txt` 中 | **torch-npu**: `setup.py:26` `from wheel.bdist_wheel import bdist_wheel`, `setup.py:610,626,631,677` | Wheel 打包，torch_npu `bdist_wheel` 构建必需 |

## Solver / optimization

| 包名 | 版本 | 上游对应 | 引用位置 | 说明 |
|------|------|---------|---------|------|
| `z3-solver` | `4.15.1.0` | ✅ 一致 (非 s390x) | `test/dynamo/test_exc.py:397`, `test/fx/test_z3_gradual_types.py:35`, `test/test_sympy_utils.py:1048`, `test/test_fx_experimental.py:2295` | Z3 定理证明器 |
| `pulp` | `2.9.0` | ✅ 一致 | `test/distributed/_tools/test_sac_ilp.py` | ILP 求解器 |

## Test utilities

| 包名 | 版本 | 上游对应 | 引用位置 | 说明 |
|------|------|---------|---------|------|
| `coverage` | 无pin | 上游不存在 | `test/run_test.py:2418` `from coverage import Coverage`, `test/run_test.py:506` `["coverage", "run", "--parallel-mode", "--source=torch"]` | 代码覆盖率，`--coverage` 选项 |
| `junitparser` | `2.1.1` | ✅ 一致 | CI 报告解析 JUnit XML | 测试结果解析 |
| `lark` | `0.12.0` | ✅ 一致 | `test/` (解析器) | 解析器库 |
| `pytest-timeout` | `2.3.1` | 上游不存在 | **torch-npu CI**: `run_npu_test_shard.py:1256` `pytest_args.append(f"--timeout={timeout}")`, `run_npu_test_file.py:268` `f"--timeout={case_timeout}"`, `test-npu.sh:55` `PYTEST_ADDOPTS="... -p timeout --timeout=600"`, `npu_poisoning_plugin.py:77` | pytest 超时控制，nightly CI 必需 |
| `tlparse` | `0.4.0` | ✅ 一致 | CI 日志解析 | 日志解析工具 |

## Infrastructure

| 包名 | 版本 | 上游对应 | 引用位置 | 说明 |
|------|------|---------|---------|------|
| `boto3` | `1.35.42` | ✅ 一致 | CI 基础设施（OBS/S3 上传） | AWS SDK |
| `PyGithub` | `2.3.0` | ✅ 一致 | CI 状态上报 | GitHub API |
| `redis` | `>=4.0.0` | ✅ 一致 | `test/distributed/tensor/` (OSS 缓存测试), `inductor/test_codecache.py`, `inductor/test_max_autotune.py` | Redis 缓存测试 |
| `aiohttp` | `3.14.3` | ✅ 一致 | `test/distributed/test_debug.py:34` `import aiohttp`, `torch/distributed/debug/_frontend.py:105` | 异步 HTTP |
| `dataclasses_json` | `0.6.7` | ✅ 一致 | `tools/stats/` (数据管道) | dataclass JSON 序列化 |

---

## 已删除的包

以下包在本次对齐中从 `requirements-test.txt` 中删除，因为在 torch-npu 仓库、上游 PyTorch 仓库、nightly CI 流程中均无引用：

| 包名 | 原版本 | 删除原因 |
|------|--------|---------|
| `beartype` | `0.17.0` | 整个仓库（含 torch-npu 和上游 pytorch）无任何 `import beartype` 或 `@beartype` |
| `attrs` | 无pin | 无 `import attrs` 或 `from attrs`，是其他包的间接依赖，pip 自动处理 |
| `decorator` | 无pin | 无 `import decorator` 或 `from decorator`，`test_decorators.py` 是测试 Python decorator 语法而非 `decorator` 包 |
| `importlib_metadata` | 无pin | 测试使用 stdlib `importlib.metadata`（`torch_npu/contrib/transfer_to_npu.py:5`），非 backport 包 |
| `zstandard` | `0.25.0` | 整个仓库（含 torch-npu 和上游 pytorch）无任何 `import zstandard`；注意 `pyzstd` 是不同的包 |

## 版本对齐修改

| 包名 | 修改前 | 修改后 | 原因 |
|------|--------|--------|------|
| `pillow` | `12.2.0` | `12.3.0` | 与上游 `requirements-ci.txt:172` 对齐 |
| `pip` | `26.0.1` | `26.1.2` | 与上游 `requirements-ci.txt:379` 对齐 |
| `aiohttp` | `3.13.4` | `3.14.3` | 与上游 `requirements-ci.txt:429` 对齐 |
| `typing-extensions` | `4.15.0` | `4.12.2` | 与上游 `requirements-ci.txt:273` 对齐 (py<3.14) |
| `librosa` | `0.10.2` | `>=0.6.2` | 与上游 `requirements-ci.txt:71` 对齐 (py3.10 不 pin) |
| `transformers` | `4.40.0` | `4.36.2` | 与上游 `install_onnx.sh:17` 对齐 |
