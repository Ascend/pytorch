# 模型性能分析
Inductor的性能调优方法可参考pytorch官方文档， 《[Profiling to understand torch.compile performance](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html)》，其中`torch.profiler.profile()`需要按照`torch_npu`相关接口进行适配，适配方法参考《[Ascend PyTorch调优工具](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/Profiling/atlasprofiling_16_0033.html)》。

例如，Torch社区中的`ModelWithBreaks`用例可参考如下进行改造，改造后的代码如下：
``` python

import torch
import torch._dynamo
import torch_npu
# user can switch between cuda and npu
device = 'npu'

class ModelWithBreaks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def create_sequential():
            return torch.nn.Sequential(
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
            )
        self.mod1 = create_sequential()
        self.mod2 = create_sequential()
        self.mod3 = create_sequential()
        self.mod4 = create_sequential()

    def forward(self, inp):
        mod1 = self.mod1(inp)
        torch._dynamo.graph_break()
        mod2 = self.mod2(mod1)
        torch._dynamo.graph_break()
        mod3 = self.mod3(mod2)
        torch._dynamo.graph_break()
        mod4 = self.mod4(mod3)
        return mod4

model = ModelWithBreaks().to(device)
inputs = [torch.randn((128, 128), device=device) for _ in range(10)]

model_c = torch.compile(model)

def fwd(inp):
    out = model_c(inp)
    return out

# warm up
fwd(inputs[0])

print("start profiling...")
with torch_npu.profiler.profile() as prof:
    for i in range(1, 4):
        fwd(inputs[i])
        prof.step()

prof.export_chrome_trace("trace_break.json")
```
Profiling中可观察到官方示例中对应的Torch-Compiled Region与CompiledFunction。
![](./trace.drawio.svg)
