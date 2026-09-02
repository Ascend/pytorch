# MegaCache特性介绍

## 特性简介

MegaCache是面向`torch.compile`编译场景的端到端缓存复用能力，用于统一保存和恢复模型编译过程中产生的多级缓存，从而减少模型冷启动时重复执行图捕获、动态图分析、代码生成、Kernel编译和Autotune带来的耗时。PyTorch社区将MegaCache定位为一种可持久化、可在不同进程以及满足兼容条件的机器间传递的编译缓存方案，通过`torch.compiler.save_cache_artifacts()`和`torch.compiler.load_cache_artifacts()`两个接口完成缓存的导出与加载。

主要功能包括：

- 支持PRECOMPILE、PGO、AOT_AUTOGRAD、INDUCTOR和AUTOTUNE缓存的统一保存与加载。
- 支持动态Shape编译缓存复用。
- 支持Catlass Kernel相关编译缓存复用。
- 支持在硬件型号及软件编译环境一致的情况下，不同进程或机器之间的缓存复用。

## 如何使用

MegaCache的基本使用流程分为**生成并保存缓存**和**加载并使用缓存**两个阶段。

### 1. 生成并保存缓存

首先正常使用`torch.compile`编译模型，并使用具有代表性的输入Shape完成模型预热。模型完成编译后，调用`torch.compiler.save_cache_artifacts()`导出当前进程收集到的MegaCache缓存。

```python
import torch

CACHE_FILE = "megacache_artifacts.bin"


class Model(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x) + x


model = Model().npu()
compiled_model = torch.compile(model)

# 使用代表性输入完成编译和预热
x = torch.randn(1024, device="npu")
compiled_model(x)

# 导出MegaCache
artifacts = torch.compiler.save_cache_artifacts()
if artifacts is None:
    raise RuntimeError("No MegaCache artifact was generated")

artifact_bytes, cache_info = artifacts

with open(CACHE_FILE, "wb") as f:
    f.write(artifact_bytes)

print(cache_info)
```

`save_cache_artifacts()`返回序列化后的缓存数据和对应的`CacheInfo`；如果当前进程没有可导出的缓存，则可能返回`None`。实际使用中只需要在模型运行完成后调用该接口并将返回的`artifact_bytes`写入文件。

生成的缓存信息示例如下，具体Artifact Key及数量由模型和实际编译结果决定：

```json
{
  "artifacts": {
    "inductor": [
      "fa5fwpqwtyezsgbqlsfxsvvrcnct4fsexnqjfwuvmyawzomjfsoj"
    ],
    "aot_autograd": [
      "a4qzwifxbq5hxx335cg6inm5nnerbioysxdlz4ssceshq5aylm5l"
    ],
    "autotune": [
      "yi/8d4a35b42a081de7bb7d107ae87118262b3e7d539e4712b138e6925aefde5ba4.best_config",
      "..."
    ],
    "precompile": [
      "63aed6eacc3c99d2c4dba4bfb80334da1d105563d98bd58ec6da23290fad66c7"
    ]
  }
}
```

其中 precompile cache 是当前社区的实验性功能，可以进一步扩大缓存范围，默认关闭。如果希望启用，需要配置：

```bash
export TORCH_CACHING_PRECOMPILE=1
```

### 2. 加载并使用缓存

在新的进程或机器中，首先读取已经保存的MegaCache文件，并通过`torch.compiler.load_cache_artifacts()`加载缓存，然后按照正常方式调用`torch.compile`并运行模型。

```python
import torch

CACHE_FILE = "megacache_artifacts.bin"

with open(CACHE_FILE, "rb") as f:
    cache_info = torch.compiler.load_cache_artifacts(f.read())

if cache_info is None:
    raise RuntimeError("Failed to load MegaCache artifact")


class Model(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x) + x


model = Model().npu()
compiled_model = torch.compile(model)

x = torch.randn(1024, device="npu")
compiled_model(x)
```

`load_cache_artifacts()`会将序列化Artifact恢复到对应的编译缓存中。后续仍需要正常调用`torch.compile`，由编译链路按照原有缓存查询机制判断是否命中。

## 使用约束

- 暂不支持CppWrapper。CppWrapper会单独保存相关二进制文件，尚未纳入MegaCache制品打包范围。

## 设备支持说明

- Atlas A5 系列产品
