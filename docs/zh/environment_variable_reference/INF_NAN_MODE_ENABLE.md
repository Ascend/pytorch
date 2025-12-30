# INF\_NAN\_MODE\_ENABLE

## 功能描述

通过此环境变量可控制AI处理器对输入数据为Inf/NaN的处理能力，即控制AI处理器使用饱和模式还是INF\_NAN模式。默认值为“1”。

-   0：饱和模式，计算出现溢出时（Inf），计算结果会饱和为浮点数极值（+-MAX）；出现无法计算的数值时（NaN），计算结果会变为0值。
-   1：INF\_NAN模式，根据定义输出Inf/NaN的计算结果。

针对<term>Atlas 训练系列产品</term>/<term>Atlas 推理系列产品</term>/<term>Atlas 200I/500 A2 推理产品</term>，仅支持饱和模式，该环境变量不生效。

针对<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>，默认值为“1”INF\_NAN模式，支持配置为“0”饱和模式。

> [!NOTICE]  
> <term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>，若需和<term>Atlas 训练系列产品</term>精度对齐，可配置为“0”饱和模式。饱和模式在计算过程中会将Inf和NaN转换成对应数据类型的最大值和0值，导致后续运算结果出现差异，非特殊情况不建议配置。Atlas <term>A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>对饱和模式配置进行了拦截，如需强制开启饱和模式，需配置[INF\_NAN\_MODE\_FORCE\_DISABLE](INF_NAN_MODE_FORCE_DISABLE.md)=1。

饱和模式：Inf置为max，NaN置为0。

Inf示例

```
torch.exp(torch.tensor([12], dtype=torch.float16).npu()) 
# tensor([65504.], device='npu:0', dtype=torch.float16)
```

NaN示例

```
torch.sqrt(torch.tensor([-1.0], dtype=torch.float16).npu()) 
# tensor([0.], device='npu:0', dtype=torch.float16)
```

INF\_NAN模式：IEEE 754标准模式。

Inf示例

```
torch.exp(torch.tensor([12], dtype=torch.float16).npu()) 
# tensor([inf], device='npu:0', dtype=torch.float16)
```

NaN示例

```
torch.sqrt(torch.tensor([-1.0], dtype=torch.float16).npu())
# tensor([nan], device='npu:0', dtype=torch.float16)
```

## 配置示例

```
export INF_NAN_MODE_ENABLE=1
```

## 使用约束

无

## 支持的型号

-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>

