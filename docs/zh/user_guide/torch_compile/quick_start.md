# 快速入门

阅读本节之前，请确保已阅读[torch.compiler](./_menu_torch_compile.md)。

先来看一个简单的`torch.compile`示例，了解如何使用`torch.compile`进行推理。此示例使用`torch.cos()`和`torch.sin()`，它们都是逐点算子，会逐个元素地处理向量。该示例可能不会带来显著的性能提升，但可以帮助您直观理解如何在自己的程序中使用`torch.compile`。

> [!NOTE]
>
> 要运行此脚本，计算机上至少需要有一个NPU。如果没有NPU，可以删除以下代码片段中的`.to(device="npu:0")`，使其在CPU上运行。

```python
import torch
def fn(x):
   a = torch.cos(x)
   b = torch.sin(a)
   return b
new_fn = torch.compile(fn, backend="inductor")
input_tensor = torch.randn(10000).to(device="npu:0")
a = new_fn(input_tensor)
```

另一个更常用的逐点算子是`torch.relu()`。在eager模式下，逐点算子的性能并非最优，因为每个算子都需要从内存中读取张量、进行修改，再将修改后的内容写回内存。Inductor执行的最重要优化就是融合。在上述示例中，可以将2次读取（`x`、`a`）和2次写入（`a`、`b`）减少为1次读取（`x`）和1次写入（`b`）。这一点对较新的NPU尤为重要，因为其瓶颈通常是内存带宽（向NPU发送数据的速度），而不是计算能力（NPU执行浮点运算的速度）。

Inductor提供的另一项重要优化是自动支持NPU Graph。NPU Graph有助于消除Python程序逐个启动kernel的开销，这一点对较新的NPU尤为重要。

TorchDynamo支持许多不同的后端，而TorchInductor通过生成[Triton](https://github.com/triton-lang/triton) kernel工作。将上面的示例保存到名为`example.py`的文件中。运行`TORCH_COMPILE_DEBUG=1 python example.py`可以检查生成的Triton kernel代码。脚本执行时，终端中会输出`DEBUG`消息。在日志末尾附近，可以看到一个目录路径，其中包含`torchinductor_<your_username>`文件夹。在该文件夹中，可以找到`output_code.py`文件，其中包含类似如下内容的已生成kernel代码：

```python
@pointwise(size_hints=[16384], filename=__file__, triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
   xnumel = 10000
   xoffset = tl.program_id(0) * XBLOCK
   xindex = xoffset + tl.arange(0, XBLOCK)[:]
   xmask = xindex < xnumel
   x0 = xindex
   tmp0 = tl.load(in_ptr0 + (x0), xmask, other=0.0)
   tmp1 = tl.cos(tmp0)
   tmp2 = tl.sin(tmp1)
   tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
```

> [!NOTE]
>
> 以上代码片段仅为示例。根据硬件的不同，实际生成的代码可能有所差异。

可以看到，`cos`和`sin`操作位于同一个Triton kernel中，并且临时变量保存在访问速度极快的寄存器中，由此可以确认`cos`和`sin`确实发生了融合。

有关Triton性能的更多信息，请参考[此处](https://triton-lang.org/)。由于代码使用Python编写，因此相对容易理解。

接下来，尝试使用PyTorch Hub中的ResNet-50等实际模型。

```python
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
opt_model = torch.compile(model, backend="inductor")
opt_model(torch.randn(1,3,64,64))
```

可用后端并不只有这一种。可以在REPL中运行`torch.compiler.list_backends()`查看所有可用后端。接下来可以尝试`npugraphs`。

## 使用预训练模型

PyTorch用户经常使用来自[Transformers](https://github.com/huggingface/transformers)或[TIMM](https://github.com/rwightman/pytorch-image-models)的预训练模型。TorchDynamo和TorchInductor的设计目标之一，就是让用户编写的各种模型都能开箱即用。

下面直接从Hugging Face Hub下载一个预训练模型并对其进行优化：

```python
import torch
from transformers import BertTokenizer, BertModel
# 复制自https://huggingface.co/bert-base-uncased
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").to(device="npu:0")
model = torch.compile(model, backend="inductor") # 这是唯一修改的一行代码
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt').to(device="npu:0")
output = model(**encoded_input)
```

如果从模型和`encoded_input`中删除`to(device="npu:0")`，Triton将生成经过优化、可在CPU上运行的C++ kernel。您可以检查为BERT生成的Triton或C++ kernel。它们比前面的三角函数示例更复杂，但同样可以快速浏览这些代码，看看能否理解PyTorch的工作方式。

类似地，下面尝试一个TIMM示例：

```python
import timm
import torch
model = timm.create_model('resnext101_32x8d', pretrained=True, num_classes=2)
opt_model = torch.compile(model, backend="inductor")
opt_model(torch.randn(64,3,7,7))
```

## 后续步骤

本节介绍了几个推理示例，并帮助您初步了解`torch.compile`的工作方式。接下来可以参考以下内容：

- [用于训练的torch.compile教程](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [torch.compiler API参考](https://docs.pytorch.org/docs/2.13/torch.compiler_api.html)
- [用于细粒度追踪的TorchDynamo API](https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_fine_grain_apis.html)
