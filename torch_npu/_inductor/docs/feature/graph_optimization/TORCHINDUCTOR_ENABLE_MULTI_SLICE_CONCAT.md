# TORCHINDUCTOR_ENABLE_MULTI_SLICE_CONCAT

## 功能描述

TORCHINDUCTOR_ENABLE_MULTI_SLICE_CONCAT=1

控制 multi_slice_concat_pass 的开关。该pass把汇聚到同一个cat的一串固定宽度、常量偏移的列切片，折叠成单个 npu_ext::multi_slice_concat。

aclnnCat要求输入连续，因此每个非连续切片都要付出一次Slice拷贝、每个带mask的切片还要额外一次where。改写后同样的数据只需一次算子下发完成搬移，且用的是固定宽度拷贝而非间址寻址。

1. 环境变量默认不设置，等价于关闭，此时pass不会被注册，推理和训练两条路径都不会执行它。
2. 取值为 1、true、yes、on、y（不区分大小写）时使能，其余取值均按关闭处理。
3. 该pass注册在POST阶段（pass_type=PassType.POST，fx_pass_level=FxPassLevel.LEVEL1）。
4. 该开关同时决定是否注册 multi_slice_concat 的inductor lowering，关闭时对应的lowering也不会注册。
5. 使能后该pass仍可通过 SHUT_DOWN_FX_PASS_LIST 关闭，两个开关取并集。

## 配置示例

观察下列日志需要同时打开inductor的DEBUG日志：

```bash
export INDUCTOR_ASCEND_LOG_LEVEL=DEBUG
```

### 使能multi_slice_concat_pass

```bash
export TORCHINDUCTOR_ENABLE_MULTI_SLICE_CONCAT=1
```

```text
# 验证：观察日志multi_slice_concat_pass已注册
DEBUG - Registering function multi_slice_concat_pass from module torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass with pass_type=PassType.POST, fx_pass_level=FxPassLevel.LEVEL1
```

### 保持关闭（默认）

```bash
unset TORCHINDUCTOR_ENABLE_MULTI_SLICE_CONCAT
```

```text
# 验证：观察日志multi_slice_concat_pass未注册
DEBUG - Ignoring registration of multi_slice_concat_pass
```

## 使用约束

- 该pass只做数据搬移，不改变数值结果。
- 匹配条件是严格的，不满足的形态一律交回 aten.cat 处理。
- 规模上限（超出即不改写，仅用于兜住病态图）：单串切片数不超过64（每个段都会在kernel体内展开）；不同源tensor数不超过32（每个源占一个kernel指针入参）；跨段同时存活的行mask不超过4个，实际模型中一个cat通常只用到两三个。

## 支持的型号

- <term>Ascend 950DT 系列产品</term>
