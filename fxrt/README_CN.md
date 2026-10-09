# FXRT

FXRT 是面向推理阶段的轻量、高性能运行时。它将 PyTorch 计算图（`torch.compile`）下沉到自有 IR 与运行时，在 Ascend 后端执行，重点优化推理时延与内存复用，并原生支持 view 类零拷贝算子。

> English version: [README.md](./README.md)

## 目录

- [构建与安装](#构建与安装)
- [跑通第一个例子](#跑通第一个例子)
- [常用环境变量](#常用环境变量)

## 构建与安装

FXRT 随 torch_npu 一起构建和发布，不单独成包。按 torch_npu 的源码构建方式构建即可，
产物 wheel 中 FXRT 位于 `torch_npu/fxrt`：

```bash
bash ci/build.sh --python=3.10
```

如需构建不带 FXRT 的 torch_npu，传入 `--disable_fxrt`。FXRT 额外的构建依赖只有
nanobind，已在 torch_npu 的 `requirements*-build.txt` 中声明，且仅构建时需要。
详见 [源码构建指南](../docs/zh/installation_guide/building_from_source.md)。

FXRT 通过 `torch_npu.fxrt` 导入。仅 `import torch_npu` 不会加载 FXRT；导入
`torch_npu.fxrt` 之后，同一个包也可以通过 `fxrt` 访问。

## 跑通第一个例子

FXRT 以 `torch.compile` 后端形式接入。把模型或函数用 `backend` 编译后，即按 FXRT 运行时执行：

```python
# quickstart.py
import torch
from torch_npu.fxrt import backend


def model(x, y):
    z = x + y
    z = z.view(4, -1)        # view 类算子走零拷贝路径
    return torch.relu(z)


# 将输入放到 npu 上
x = torch.randn(2, 8).npu()
y = torch.randn(2, 8).npu()

compiled = torch.compile(model, backend=backend)
out = compiled(x, y)
print(out.shape)
```

要点：

- `from torch_npu.fxrt import backend` 拿到编译后端
- `torch.compile(fn, backend=backend)` 之后照常调用即可
- view / reshape / permute / slice 等算子默认走 FXRT 的零拷贝 view 实现

## 常用环境变量

| 变量 | 作用 |
| --- | --- |
| `FXRT_DISABLE_VIEW_OPS` | 关闭指定 view 算子的零拷贝实现，回退到非 view 路径，用于排障与对比。取值为 `permute`、`transpose`、`t`、`movedim`、`split`、`chunk` 中若干项的逗号分隔列表，或用 `all` 一次性覆盖这些算子。 |
| `FXRT_DEV_DUMP_IR` | 置为 `1` 时 dump 编译生成的 IR，便于调试 |
