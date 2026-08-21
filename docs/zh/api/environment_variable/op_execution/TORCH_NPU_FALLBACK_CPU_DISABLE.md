# TORCH_NPU_FALLBACK_CPU_DISABLE

## 功能描述

通过此环境变量可控制TorchNPU已纳管的隐式CPU fallback路径是否允许执行。当NPU侧没有对应算子实现，或者算子的已纳管兼容路径需要在CPU上完成主要数值计算时，可通过该环境变量选择保持兼容执行或直接报错。

- 配置为“0”时，允许CPU fallback。未配置该环境变量时默认按“0”处理，以保持现有兼容行为。CPU fallback路径可将输入从NPU复制到CPU完成计算，再将结果复制回NPU；已配置告警的路径会打印fallback告警。
- 配置为“1”时，禁止CPU fallback。算子在进入CPU主要计算前直接报错，错误信息中会包含算子名称和`TORCH_NPU_FALLBACK_CPU_DISABLE=1`，且不会打印“即将fallback到CPU”的告警。

## 配置示例

禁止已纳管算子的CPU fallback：

```bash
export TORCH_NPU_FALLBACK_CPU_DISABLE=1
```

允许已纳管算子的CPU fallback：

```bash
export TORCH_NPU_FALLBACK_CPU_DISABLE=0
```

## 使用约束

- 该环境变量仅支持配置为“0”或“1”，默认值为“0”。
- 必须在启动Python进程前配置该环境变量。TorchNPU在进程内首次读取后会缓存配置值，运行过程中修改`os.environ`不会改变当前进程的行为；如需切换配置，请启动新的Python进程。
- 该环境变量只控制TorchNPU已接入门控的CPU fallback路径，包括未注册NPU kernel时命中的Dispatcher fallback，以及已接入门控的显式CPU kernel。ACL Op/aclops路径、AclNN兼容路径、控制类CPU计算、Host staging、Host权重预处理、用户显式调用`.cpu()`或显式创建CPU Tensor等场景不属于该环境变量的控制范围。
- 该环境变量不会改变算子注册关系，也不会为算子增加NPU实现。配置为“1”后，缺少NPU实现且命中已纳管CPU fallback路径的算子将直接报错，可能影响现有模型的兼容性；建议在算子支持度检查、性能排查或要求严格NPU执行的场景中使用。
- `NPU_INDUCTOR_FALLBACK_LIST`控制Inductor是否将指定算子回退到PyTorch原生Aten调用，本环境变量控制Aten运行时是否允许进一步进入已纳管的CPU fallback。Inductor回退到Aten不等同于CPU fallback；只有后续实际命中已纳管CPU fallback路径时，本环境变量才会生效。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 800I A2 推理产品</term>
- <term>Atlas 推理系列产品</term>
- <term>Ascend 950DT</term>
