# torch.Storage

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.7/storage.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Special cases](#special-cases)
- [Legacy Typed Storage](#legacy-typed-storage)

</div>

<div style="display:none;">

## &#8203;torch.Storage

</div>

## Special cases

### <code><i>class</i></code> torch.UntypedStorage

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">bfloat16()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.bfloat16](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.bfloat16)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">bool()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.bool](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.bool)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">byte()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.byte](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.byte)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">byteswap()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.byteswap](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.byteswap)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">char()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.char](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.char)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">clone()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.clone](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.clone)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">complex_double()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.complex_double](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.complex_double)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">complex_float()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.complex_float](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.complex_float)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">copy_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.copy_](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.copy_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">cpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.cpu](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.cpu)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">cuda()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.cuda](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.cuda)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">data_ptr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.data_ptr](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.data_ptr)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">device()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.device](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.device)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">double()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.double](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.double)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">element_size()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.element_size](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.element_size)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">filename()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.filename](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.filename)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">fill_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.fill_](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.fill_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">float()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.float](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.float)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">float8_e4m3fn()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.float8_e4m3fn](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.float8_e4m3fn)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">float8_e5m2()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.float8_e5m2](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.float8_e5m2)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">float8_e4m3fnuz()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.float8_e4m3fnuz](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.float8_e4m3fnuz)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">float8_e5m2fnuz()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.float8_e5m2fnuz](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.float8_e5m2fnuz)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">from_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.from_buffer](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.from_buffer)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">from_file()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.from_file](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.from_file)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">get_device()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.get_device](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.get_device)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">half()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.half](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.half)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">hpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.hpu](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.hpu)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">int()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.int](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.int)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">is_cuda()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.is_cuda](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.is_cuda)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">is_hpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.is_hpu](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.is_hpu)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">is_pinned()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.is_pinned](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.is_pinned)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">is_shared()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.is_shared](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.is_shared)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">is_sparse()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.is_sparse](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.is_sparse)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">is_sparse_csr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.is_sparse_csr](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.is_sparse_csr)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">long()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.long](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.long)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mps()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.mps](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.mps)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">nbytes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.nbytes](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.nbytes)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">new()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.new](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.new)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">pin_memory()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.pin_memory](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.pin_memory)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">resize_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.resize_](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.resize_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">share_memory_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.share_memory_](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.share_memory_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">short()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.short](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.short)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">size()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.size](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.size)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">tolist()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.tolist](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.tolist)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.type](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.type)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">untyped()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.UntypedStorage.untyped](https://pytorch.org/docs/2.7/storage.html#torch.UntypedStorage.untyped)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## Legacy Typed Storage

### <code><i>class</i></code> torch.DoubleStorage

<div style="margin-left: 2em">

**原生文档**：[torch.DoubleStorage](https://pytorch.org/docs/2.7/storage.html#torch.DoubleStorage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.DoubleStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.DoubleStorage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.HalfStorage

<div style="margin-left: 2em">

**原生文档**：[torch.HalfStorage](https://pytorch.org/docs/2.7/storage.html#torch.HalfStorage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.HalfStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.HalfStorage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.LongStorage

<div style="margin-left: 2em">

**原生文档**：[torch.LongStorage](https://pytorch.org/docs/2.7/storage.html#torch.LongStorage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.LongStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.LongStorage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.ShortStorage

<div style="margin-left: 2em">

**原生文档**：[torch.ShortStorage](https://pytorch.org/docs/2.7/storage.html#torch.ShortStorage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ShortStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.ShortStorage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.CharStorage

<div style="margin-left: 2em">

**原生文档**：[torch.CharStorage](https://pytorch.org/docs/2.7/storage.html#torch.CharStorage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.CharStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.CharStorage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.ByteStorage

<div style="margin-left: 2em">

**原生文档**：[torch.ByteStorage](https://pytorch.org/docs/2.7/storage.html#torch.ByteStorage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ByteStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.ByteStorage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.BoolStorage

<div style="margin-left: 2em">

**原生文档**：[torch.BoolStorage](https://pytorch.org/docs/2.7/storage.html#torch.BoolStorage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.BoolStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.BoolStorage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.BFloat16Storage

<div style="margin-left: 2em">

**原生文档**：[torch.BFloat16Storage](https://pytorch.org/docs/2.7/storage.html#torch.BFloat16Storage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.BFloat16Storage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.BFloat16Storage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.ComplexDoubleStorage

<div style="margin-left: 2em">

**原生文档**：[torch.ComplexDoubleStorage](https://pytorch.org/docs/2.7/storage.html#torch.ComplexDoubleStorage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ComplexDoubleStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.ComplexDoubleStorage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.ComplexFloatStorage

<div style="margin-left: 2em">

**原生文档**：[torch.ComplexFloatStorage](https://pytorch.org/docs/2.7/storage.html#torch.ComplexFloatStorage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ComplexFloatStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.ComplexFloatStorage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.QUInt8Storage

<div style="margin-left: 2em">

**原生文档**：[torch.QUInt8Storage](https://pytorch.org/docs/2.7/storage.html#torch.QUInt8Storage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.QUInt8Storage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.QUInt8Storage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

**限制与说明**： 支持uint8

</div>

### <code><i>class</i></code> torch.QInt8Storage

<div style="margin-left: 2em">

**原生文档**：[torch.QInt8Storage](https://pytorch.org/docs/2.7/storage.html#torch.QInt8Storage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.QInt8Storage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.QInt8Storage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `self`仅支持int8

</div>

</div>

### <code><i>class</i></code> torch.QInt32Storage

<div style="margin-left: 2em">

**原生文档**：[torch.QInt32Storage](https://pytorch.org/docs/2.7/storage.html#torch.QInt32Storage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.QInt32Storage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.QInt32Storage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `self`仅支持int32

</div>

</div>

### <code><i>class</i></code> torch.QUInt4x2Storage

<div style="margin-left: 2em">

**原生文档**：[torch.QUInt4x2Storage](https://pytorch.org/docs/2.7/storage.html#torch.QUInt4x2Storage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.QUInt4x2Storage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.QUInt4x2Storage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `self`仅支持uint8

</div>

</div>

### <code><i>class</i></code> torch.QUInt2x4Storage

<div style="margin-left: 2em">

**原生文档**：[torch.QUInt2x4Storage](https://pytorch.org/docs/2.7/storage.html#torch.QUInt2x4Storage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.QUInt2x4Storage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.QUInt2x4Storage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `self`仅支持uint8

</div>

</div>

### <code><i>class</i></code> torch.TypedStorage

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">bfloat16()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.bfloat16](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.bfloat16)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">bool()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.bool](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.bool)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">byte()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.byte](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.byte)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">char()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.char](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.char)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">clone()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.clone](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.clone)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">complex_double()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.complex_double](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.complex_double)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">complex_float()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.complex_float](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.complex_float)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">copy_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.copy_](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.copy_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">cpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.cpu](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.cpu)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">cuda()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.cuda](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.cuda)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">data_ptr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.data_ptr](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.data_ptr)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">device()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.device](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.device)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">double()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.double](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.double)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">element_size()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.element_size](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.element_size)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">filename()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.filename](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.filename)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">fill_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.fill_](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.fill_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">float()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.float](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.float)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">float8_e4m3fn()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.float8_e4m3fn](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.float8_e4m3fn)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">float8_e5m2()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.float8_e5m2](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.float8_e5m2)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">from_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.from_buffer](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.from_buffer)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">from_file()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.from_file](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.from_file)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">get_device()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.get_device](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.get_device)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">half()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.half](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.half)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">hpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.hpu](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.hpu)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">int()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.int](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.int)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">is_cuda()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.is_cuda](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.is_cuda)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">is_hpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.is_hpu](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.is_hpu)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">is_pinned()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.is_pinned](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.is_pinned)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">is_shared()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.is_shared](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.is_shared)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">is_sparse()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.is_sparse](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.is_sparse)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">long()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.long](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.long)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">nbytes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.nbytes](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.nbytes)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">pickle_storage_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.pickle_storage_type](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.pickle_storage_type)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">pin_memory()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.pin_memory](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.pin_memory)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">resize_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.resize_](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.resize_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">share_memory_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.share_memory_](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.share_memory_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">short()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.short](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.short)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">size()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.size](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.size)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">tolist()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.tolist](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.tolist)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.type](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.type)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">untyped()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.TypedStorage.untyped](https://pytorch.org/docs/2.7/storage.html#torch.TypedStorage.untyped)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.FloatStorage

<div style="margin-left: 2em">

**原生文档**：[torch.FloatStorage](https://pytorch.org/docs/2.7/storage.html#torch.FloatStorage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.FloatStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.FloatStorage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.IntStorage

<div style="margin-left: 2em">

**原生文档**：[torch.IntStorage](https://pytorch.org/docs/2.7/storage.html#torch.IntStorage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">dtype()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.IntStorage.dtype](https://pytorch.org/docs/2.7/storage.html#torch.IntStorage.dtype)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>
