# Quantization

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.13/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://docs.pytorch.org/docs/2.13/quantization-support.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [torch.ao.quantization](#torchaoquantization)
- [torch.ao.quantization.qconfig_mapping](#torchaoquantizationqconfig_mapping)
- [torch.ao.quantization.backend_config](#torchaoquantizationbackend_config)
- [torch.ao.quantization.fx.custom_config](#torchaoquantizationfxcustom_config)
- [torch.ao.quantization.observer](#torchaoquantizationobserver)
- [torch.ao.quantization.fake_quantize](#torchaoquantizationfake_quantize)
- [torch.ao.quantization.qconfig](#torchaoquantizationqconfig)
- [torch.ao.nn.intrinsic](#torchaonnintrinsic)
- [torch.ao.nn.intrinsic.qat](#torchaonnintrinsicqat)
- [torch.ao.nn.qat](#torchaonnqat)
- [torch.ao.nn.quantizable](#torchaonnquantizable)
- [torch.ao.nn.quantized.dynamic](#torchaonnquantizeddynamic)

</div>

<div style="display:none;">

## &#8203;Quantization

</div>

### torch.ao.ns.fx.utils.compute_sqnr

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns.fx.utils.compute_sqnr](https://pytorch.org/docs/2.13/quantization.html#torch.ao.ns.fx.utils.compute_sqnr)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

## torch.ao.quantization

### torch.ao.quantization.prepare_qat

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.prepare_qat](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.prepare_qat.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.ao.quantization.convert

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.convert](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.convert.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.ao.quantization.QuantStub

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.QuantStub](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.QuantStub.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### <code><i>class</i></code> torch.ao.quantization.DeQuantStub

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.DeQuantStub](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.DeQuantStub.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### <code><i>class</i></code> torch.ao.quantization.QuantWrapper

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.QuantWrapper](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.QuantWrapper.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

## torch.ao.quantization.qconfig_mapping

### <code><i>class</i></code> torch.ao.quantization.qconfig_mapping.QConfigMapping

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.from_dict](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.from_dict)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">set_global()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_global](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_global)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">set_module_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_module_name_object_type_order()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_object_type_order](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_object_type_order)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">set_module_name_regex()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_regex](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_regex)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_object_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_object_type](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_object_type)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.to_dict](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.to_dict)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

</div>

### torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

## torch.ao.quantization.backend_config

### <code><i>class</i></code> torch.ao.quantization.backend_config.BackendConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendConfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

> <font size="3">configs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.configs](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.configs)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.from_dict](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.from_dict)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_backend_pattern_config()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_config](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_config)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_backend_pattern_configs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_configs](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_configs)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.set_name](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_name)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.to_dict](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.to_dict)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.backend_config.BackendPatternConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

> <font size="3">add_dtype_config()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.add_dtype_config](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.add_dtype_config)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.from_dict](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.from_dict)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_dtype_configs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_dtype_configs](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_dtype_configs)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">set_fused_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_fused_module](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_fused_module)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">set_fuser_method()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_fuser_method](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_fuser_method)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_observation_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_observation_type](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_observation_type)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">set_pattern()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_pattern](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_pattern)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_qat_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_qat_module](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_qat_module)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">set_reference_quantized_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_reference_quantized_module](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_reference_quantized_module)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">set_root_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.to_dict](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.to_dict)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.backend_config.DTypeConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.DTypeConfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.DTypeConfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.DTypeConfig.from_dict](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.DTypeConfig.html#torch.ao.quantization.backend_config.DTypeConfig.from_dict)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.DTypeConfig.to_dict](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.DTypeConfig.html#torch.ao.quantization.backend_config.DTypeConfig.to_dict)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.backend_config.DTypeWithConstraints

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.DTypeWithConstraints](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.DTypeWithConstraints.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### <code><i>class</i></code> torch.ao.quantization.backend_config.ObservationType

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.ObservationType](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.ObservationType.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">INPUT_OUTPUT_NOT_OBSERVED()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.ObservationType.INPUT_OUTPUT_NOT_OBSERVED](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.INPUT_OUTPUT_NOT_OBSERVED)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">OUTPUT_SHARE_OBSERVER_WITH_INPUT()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

</div>

## torch.ao.quantization.fx.custom_config

### <code><i>class</i></code> torch.ao.quantization.fx.custom_config.FuseCustomConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.FuseCustomConfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.FuseCustomConfig.from_dict](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.from_dict)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_preserved_attributes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.FuseCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.set_preserved_attributes)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.FuseCustomConfig.to_dict](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.to_dict)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.fx.custom_config.PrepareCustomConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">set_float_to_observed_mapping()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_float_to_observed_mapping](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_float_to_observed_mapping)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_input_quantized_indexes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_input_quantized_indexes](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_input_quantized_indexes)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">set_non_traceable_module_classes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_classes](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_classes)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_non_traceable_module_names()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_names](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_names)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">set_output_quantized_indexes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_output_quantized_indexes](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_output_quantized_indexes)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_preserved_attributes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_preserved_attributes)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_standalone_module_class()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_class](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_class)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">set_standalone_module_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_name](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_name)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.to_dict](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.to_dict)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.fx.custom_config.ConvertCustomConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">set_observed_to_quantized_mapping()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_observed_to_quantized_mapping](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_observed_to_quantized_mapping)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_preserved_attributes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_preserved_attributes)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.to_dict](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.to_dict)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

## torch.ao.quantization.observer

### <code><i>class</i></code> torch.ao.quantization.observer.ObserverBase

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.ObserverBase](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.ObserverBase.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

> <font size="3">with_args()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.ObserverBase.with_args](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.ObserverBase.html#torch.ao.quantization.observer.ObserverBase.with_args)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">with_callable_args()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.ObserverBase.with_callable_args](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.ObserverBase.html#torch.ao.quantization.observer.ObserverBase.with_callable_args)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.MinMaxObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MinMaxObserver](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.MinMaxObserver.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

> <font size="3">calculate_qparams()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MinMaxObserver.calculate_qparams](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.calculate_qparams)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MinMaxObserver.forward](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.forward)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">reset_min_max_vals()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.MovingAverageMinMaxObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MovingAverageMinMaxObserver](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.MovingAverageMinMaxObserver.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.PerChannelMinMaxObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.PerChannelMinMaxObserver](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.PerChannelMinMaxObserver.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

> <font size="3">reset_min_max_vals()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.PerChannelMinMaxObserver.html#torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.HistogramObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.HistogramObserver](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.HistogramObserver.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.PlaceholderObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.PlaceholderObserver](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.PlaceholderObserver.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.RecordingObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.RecordingObserver](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.RecordingObserver.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：可能回退至CPU执行

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.NoopObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.NoopObserver](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.NoopObserver.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.ao.quantization.observer.get_observer_state_dict

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.get_observer_state_dict](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.get_observer_state_dict.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.ao.quantization.observer.load_observer_state_dict

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.load_observer_state_dict](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.load_observer_state_dict.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.ao.quantization.observer.default_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_observer](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.default_observer.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.ao.quantization.observer.default_placeholder_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_placeholder_observer](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.default_placeholder_observer.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.ao.quantization.observer.default_debug_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_debug_observer](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.default_debug_observer.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.ao.quantization.observer.default_weight_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_weight_observer](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.default_weight_observer.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.ao.quantization.observer.default_histogram_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_histogram_observer](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.default_histogram_observer.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.ao.quantization.observer.default_per_channel_weight_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_per_channel_weight_observer](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.default_per_channel_weight_observer.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.ao.quantization.observer.default_dynamic_quant_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_dynamic_quant_observer](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.default_dynamic_quant_observer.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.ao.quantization.observer.default_float_qparams_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_float_qparams_observer](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.observer.default_float_qparams_observer.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

## torch.ao.quantization.fake_quantize

### <code><i>class</i></code> torch.ao.quantization.fake_quantize.FakeQuantize

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.FakeQuantize](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：可能回退至CPU执行

</div>

### <code><i>class</i></code> torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：可能回退至CPU执行

</div>

### torch.ao.quantization.fake_quantize.disable_fake_quant

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.disable_fake_quant](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fake_quantize.disable_fake_quant.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.ao.quantization.fake_quantize.enable_fake_quant

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.enable_fake_quant](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fake_quantize.enable_fake_quant.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.ao.quantization.fake_quantize.disable_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.disable_observer](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fake_quantize.disable_observer.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.ao.quantization.fake_quantize.enable_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.enable_observer](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.fake_quantize.enable_observer.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

## torch.ao.quantization.qconfig

### <code><i>class</i></code> torch.ao.quantization.qconfig.QConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.QConfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig.QConfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.ao.quantization.qconfig.default_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_qconfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig.default_qconfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.ao.quantization.qconfig.default_debug_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_debug_qconfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig.default_debug_qconfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.ao.quantization.qconfig.default_per_channel_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_per_channel_qconfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig.default_per_channel_qconfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.ao.quantization.qconfig.default_dynamic_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_dynamic_qconfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig.default_dynamic_qconfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.ao.quantization.qconfig.float16_dynamic_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.float16_dynamic_qconfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig.float16_dynamic_qconfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.ao.quantization.qconfig.float16_static_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.float16_static_qconfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig.float16_static_qconfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.ao.quantization.qconfig.per_channel_dynamic_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.per_channel_dynamic_qconfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig.per_channel_dynamic_qconfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.ao.quantization.qconfig.default_qat_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_qat_qconfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig.default_qat_qconfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.ao.quantization.qconfig.default_weight_only_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_weight_only_qconfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig.default_weight_only_qconfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.ao.quantization.qconfig.default_activation_only_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_activation_only_qconfig](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig.default_activation_only_qconfig.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.ao.quantization.qconfig.default_qat_qconfig_v2

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_qat_qconfig_v2](https://pytorch.org/docs/2.13/generated/torch.ao.quantization.qconfig.default_qat_qconfig_v2.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

## torch.ao.nn.intrinsic

### <code><i>class</i></code> torch.ao.nn.intrinsic.LinearReLU

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.LinearReLU](https://pytorch.org/docs/2.13/generated/torch.ao.nn.intrinsic.LinearReLU.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

## torch.ao.nn.intrinsic.qat

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.LinearReLU

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.LinearReLU](https://pytorch.org/docs/2.13/generated/torch.ao.nn.intrinsic.qat.LinearReLU.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBn1d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.ConvBn1d](https://pytorch.org/docs/2.13/generated/torch.ao.nn.intrinsic.qat.ConvBn1d.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBnReLU1d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.ConvBnReLU1d](https://pytorch.org/docs/2.13/generated/torch.ao.nn.intrinsic.qat.ConvBnReLU1d.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBnReLU2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.ConvBnReLU2d](https://pytorch.org/docs/2.13/generated/torch.ao.nn.intrinsic.qat.ConvBnReLU2d.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.ao.nn.intrinsic.qat.update_bn_stats

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.update_bn_stats](https://pytorch.org/docs/2.13/generated/torch.ao.nn.intrinsic.qat.update_bn_stats.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.ao.nn.intrinsic.qat.freeze_bn_stats

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.freeze_bn_stats](https://pytorch.org/docs/2.13/generated/torch.ao.nn.intrinsic.qat.freeze_bn_stats.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：可能回退至CPU执行

</div>

## torch.ao.nn.qat

### <code><i>class</i></code> torch.ao.nn.qat.Linear

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.qat.Linear](https://pytorch.org/docs/2.13/generated/torch.ao.nn.qat.Linear.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

## torch.ao.nn.quantizable

### <code><i>class</i></code> torch.ao.nn.quantizable.LSTM

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantizable.LSTM](https://pytorch.org/docs/2.13/generated/torch.ao.nn.quantizable.LSTM.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

## torch.ao.nn.quantized.dynamic

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.Linear

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.Linear](https://pytorch.org/docs/2.13/generated/torch.ao.nn.quantized.dynamic.Linear.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.LSTM

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.LSTM](https://pytorch.org/docs/2.13/generated/torch.ao.nn.quantized.dynamic.LSTM.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.GRU

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.GRU](https://pytorch.org/docs/2.13/generated/torch.ao.nn.quantized.dynamic.GRU.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.RNNCell

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.RNNCell](https://pytorch.org/docs/2.13/generated/torch.ao.nn.quantized.dynamic.RNNCell.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.LSTMCell

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.LSTMCell](https://pytorch.org/docs/2.13/generated/torch.ao.nn.quantized.dynamic.LSTMCell.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.GRUCell

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.GRUCell](https://pytorch.org/docs/2.13/generated/torch.ao.nn.quantized.dynamic.GRUCell.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**： `input`仅支持fp32

</div>
