# Quantization

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://docs.pytorch.org/docs/2.7/quantization-support.html)。
> - 注：在[Quantization Accuracy Debugging](https://docs.pytorch.org/docs/2.7/quantization-accuracy-debugging.html)中可以找到[Numerical Debugging Tooling](#numerical-debugging-tooling)下API的具体用法。

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
- [Numerical Debugging Tooling](#numerical-debugging-tooling)

</div>

<div style="display:none;">

## &#8203;Quantization

</div>

## torch.ao.quantization

### torch.ao.quantization.prepare_qat

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.prepare_qat](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.prepare_qat.html)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id3 -->

</div>

### torch.ao.quantization.convert

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.convert](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.convert.html)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id6 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.QuantWrapper

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.QuantWrapper](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.QuantWrapper.html)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id9 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.QuantStub

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.QuantStub](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.QuantStub.html)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id12 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.DeQuantStub

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.DeQuantStub](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.DeQuantStub.html)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id15 -->

</div>

## torch.ao.quantization.qconfig_mapping

### <code><i>class</i></code> torch.ao.quantization.qconfig_mapping.QConfigMapping

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id18 -->

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.from_dict)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id21 -->

</div>

> <font size="3">set_global()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_global](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_global)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id24 -->

</div>

> <font size="3">set_module_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id27 -->

</div>

> <font size="3">set_module_name_object_type_order()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_object_type_order](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_object_type_order)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id30 -->

</div>

> <font size="3">set_module_name_regex()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_regex](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_regex)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id33 -->

</div>

> <font size="3">set_object_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_object_type](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_object_type)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id36 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.to_dict)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id39 -->

</div>

</div>

### torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping.html)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id42 -->

</div>

### torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping.html)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id45 -->

</div>

## torch.ao.quantization.backend_config

### <code><i>class</i></code> torch.ao.quantization.backend_config.BackendConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id48 -->

> <font size="3">configs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.configs](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.configs)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id51 -->

</div>

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.from_dict)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id54 -->

</div>

> <font size="3">set_backend_pattern_config()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_config](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_config)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id57 -->

</div>

> <font size="3">set_backend_pattern_configs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_configs](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_configs)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id60 -->

</div>

> <font size="3">set_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.set_name](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_name)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id63 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.to_dict)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id66 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.backend_config.BackendPatternConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id69 -->

> <font size="3">add_dtype_config()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.add_dtype_config](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.add_dtype_config)

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id72 -->

</div>

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.from_dict)

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id75 -->

</div>

> <font size="3">set_dtype_configs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_dtype_configs](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_dtype_configs)

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id78 -->

</div>

> <font size="3">set_fused_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_fused_module](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_fused_module)

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id81 -->

</div>

> <font size="3">set_fuser_method()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_fuser_method](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_fuser_method)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id84 -->

</div>

> <font size="3">set_observation_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_observation_type](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_observation_type)

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id87 -->

</div>

> <font size="3">set_pattern()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_pattern](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_pattern)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id90 -->

</div>

> <font size="3">set_qat_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_qat_module](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_qat_module)

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id93 -->

</div>

> <font size="3">set_reference_quantized_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_reference_quantized_module](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_reference_quantized_module)

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id96 -->

</div>

> <font size="3">set_root_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module)

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id99 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.to_dict)

**产品支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id102 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.backend_config.DTypeConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.DTypeConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.DTypeConfig.html)

**产品支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id105 -->

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.DTypeConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.DTypeConfig.html#torch.ao.quantization.backend_config.DTypeConfig.from_dict)

**产品支持情况**：

<!-- npu="910b" id106 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id108 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.DTypeConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.DTypeConfig.html#torch.ao.quantization.backend_config.DTypeConfig.to_dict)

**产品支持情况**：

<!-- npu="910b" id109 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id111 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.backend_config.DTypeWithConstraints

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.DTypeWithConstraints](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.DTypeWithConstraints.html)

**产品支持情况**：

<!-- npu="910b" id112 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id114 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.backend_config.ObservationType

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.ObservationType](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.ObservationType.html)

**产品支持情况**：

<!-- npu="910b" id115 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id117 -->

> <font size="3">INPUT_OUTPUT_NOT_OBSERVED()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.ObservationType.INPUT_OUTPUT_NOT_OBSERVED](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.INPUT_OUTPUT_NOT_OBSERVED)

**产品支持情况**：

<!-- npu="910b" id118 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id120 -->

</div>

> <font size="3">OUTPUT_SHARE_OBSERVER_WITH_INPUT()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)

**产品支持情况**：

<!-- npu="910b" id121 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id123 -->

</div>

> <font size="3">OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)

**产品支持情况**：

<!-- npu="910b" id124 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id126 -->

</div>

</div>

## torch.ao.quantization.fx.custom_config

### <code><i>class</i></code> torch.ao.quantization.fx.custom_config.FuseCustomConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.FuseCustomConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html)

**产品支持情况**：

<!-- npu="910b" id127 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id129 -->

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.FuseCustomConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.from_dict)

**产品支持情况**：

<!-- npu="910b" id130 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id132 -->

</div>

> <font size="3">set_preserved_attributes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.FuseCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.set_preserved_attributes)

**产品支持情况**：

<!-- npu="910b" id133 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id135 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.FuseCustomConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.to_dict)

**产品支持情况**：

<!-- npu="910b" id136 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id138 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.fx.custom_config.PrepareCustomConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html)

**产品支持情况**：

<!-- npu="910b" id139 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id141 -->

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict)

**产品支持情况**：

<!-- npu="910b" id142 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id144 -->

</div>

> <font size="3">set_float_to_observed_mapping()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_float_to_observed_mapping](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_float_to_observed_mapping)

**产品支持情况**：

<!-- npu="910b" id145 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id147 -->

</div>

> <font size="3">set_input_quantized_indexes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_input_quantized_indexes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_input_quantized_indexes)

**产品支持情况**：

<!-- npu="910b" id148 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id150 -->

</div>

> <font size="3">set_non_traceable_module_classes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_classes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_classes)

**产品支持情况**：

<!-- npu="910b" id151 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id153 -->

</div>

> <font size="3">set_non_traceable_module_names()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_names](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_names)

**产品支持情况**：

<!-- npu="910b" id154 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id156 -->

</div>

> <font size="3">set_output_quantized_indexes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_output_quantized_indexes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_output_quantized_indexes)

**产品支持情况**：

<!-- npu="910b" id157 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id159 -->

</div>

> <font size="3">set_preserved_attributes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_preserved_attributes)

**产品支持情况**：

<!-- npu="910b" id160 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id162 -->

</div>

> <font size="3">set_standalone_module_class()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_class](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_class)

**产品支持情况**：

<!-- npu="910b" id163 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id165 -->

</div>

> <font size="3">set_standalone_module_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_name](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_name)

**产品支持情况**：

<!-- npu="910b" id166 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id168 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.to_dict)

**产品支持情况**：

<!-- npu="910b" id169 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id171 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.fx.custom_config.ConvertCustomConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html)

**产品支持情况**：

<!-- npu="910b" id172 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id174 -->

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict)

**产品支持情况**：

<!-- npu="910b" id175 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id177 -->

</div>

> <font size="3">set_observed_to_quantized_mapping()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_observed_to_quantized_mapping](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_observed_to_quantized_mapping)

**产品支持情况**：

<!-- npu="910b" id178 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id180 -->

</div>

> <font size="3">set_preserved_attributes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_preserved_attributes)

**产品支持情况**：

<!-- npu="910b" id181 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id183 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.to_dict)

**产品支持情况**：

<!-- npu="910b" id184 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id186 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry.html)

**产品支持情况**：

<!-- npu="910b" id187 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id189 -->

</div>

## torch.ao.quantization.observer

### <code><i>class</i></code> torch.ao.quantization.observer.ObserverBase

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.ObserverBase](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.ObserverBase.html)

**产品支持情况**：

<!-- npu="910b" id190 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id192 -->

> <font size="3">with_args()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.ObserverBase.with_args](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.ObserverBase.html#torch.ao.quantization.observer.ObserverBase.with_args)

**产品支持情况**：

<!-- npu="910b" id193 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id195 -->

</div>

> <font size="3">with_callable_args()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.ObserverBase.with_callable_args](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.ObserverBase.html#torch.ao.quantization.observer.ObserverBase.with_callable_args)

**产品支持情况**：

<!-- npu="910b" id196 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id198 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.MinMaxObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MinMaxObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MinMaxObserver.html)

**产品支持情况**：

<!-- npu="910b" id199 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id201 -->

> <font size="3">calculate_qparams()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MinMaxObserver.calculate_qparams](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.calculate_qparams)

**产品支持情况**：

<!-- npu="910b" id202 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id204 -->

</div>

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MinMaxObserver.forward](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.forward)

**产品支持情况**：

<!-- npu="910b" id205 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id207 -->

</div>

> <font size="3">reset_min_max_vals()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals)

**产品支持情况**：

<!-- npu="910b" id208 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id210 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.MovingAverageMinMaxObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MovingAverageMinMaxObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MovingAverageMinMaxObserver.html)

**产品支持情况**：

<!-- npu="910b" id211 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id213 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.PerChannelMinMaxObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.PerChannelMinMaxObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.PerChannelMinMaxObserver.html)

**产品支持情况**：

<!-- npu="910b" id214 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id215 -->
<!-- npu="950" id216 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id216 -->

> <font size="3">reset_min_max_vals()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.PerChannelMinMaxObserver.html#torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals)

**产品支持情况**：

<!-- npu="910b" id217 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id219 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver.html)

**产品支持情况**：

<!-- npu="910b" id220 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id222 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.HistogramObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.HistogramObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.HistogramObserver.html)

**产品支持情况**：

<!-- npu="910b" id223 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id223 -->
<!-- npu="A3" id224 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id224 -->
<!-- npu="950" id225 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id225 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.PlaceholderObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.PlaceholderObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.PlaceholderObserver.html)

**产品支持情况**：

<!-- npu="910b" id226 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id226 -->
<!-- npu="A3" id227 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id227 -->
<!-- npu="950" id228 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id228 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.RecordingObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.RecordingObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.RecordingObserver.html)

**产品支持情况**：

<!-- npu="910b" id229 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id229 -->
<!-- npu="A3" id230 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id230 -->
<!-- npu="950" id231 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id231 -->

**限制与说明**：可能回退至CPU执行

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.NoopObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.NoopObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.NoopObserver.html)

**产品支持情况**：

<!-- npu="910b" id232 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id232 -->
<!-- npu="A3" id233 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id233 -->
<!-- npu="950" id234 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id234 -->

</div>

### torch.ao.quantization.observer.get_observer_state_dict

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.get_observer_state_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.get_observer_state_dict.html)

**产品支持情况**：

<!-- npu="910b" id235 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id235 -->
<!-- npu="A3" id236 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id236 -->
<!-- npu="950" id237 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id237 -->

</div>

### torch.ao.quantization.observer.load_observer_state_dict

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.load_observer_state_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.load_observer_state_dict.html)

**产品支持情况**：

<!-- npu="910b" id238 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id238 -->
<!-- npu="A3" id239 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id239 -->
<!-- npu="950" id240 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id240 -->

</div>

### torch.ao.quantization.observer.default_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_observer.html)

**产品支持情况**：

<!-- npu="910b" id241 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id241 -->
<!-- npu="A3" id242 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id242 -->
<!-- npu="950" id243 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id243 -->

</div>

### torch.ao.quantization.observer.default_placeholder_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_placeholder_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_placeholder_observer.html)

**产品支持情况**：

<!-- npu="910b" id244 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id244 -->
<!-- npu="A3" id245 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id245 -->
<!-- npu="950" id246 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id246 -->

</div>

### torch.ao.quantization.observer.default_debug_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_debug_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_debug_observer.html)

**产品支持情况**：

<!-- npu="910b" id247 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id247 -->
<!-- npu="A3" id248 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id248 -->
<!-- npu="950" id249 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id249 -->

</div>

### torch.ao.quantization.observer.default_weight_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_weight_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_weight_observer.html)

**产品支持情况**：

<!-- npu="910b" id250 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id250 -->
<!-- npu="A3" id251 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id251 -->
<!-- npu="950" id252 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id252 -->

</div>

### torch.ao.quantization.observer.default_histogram_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_histogram_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_histogram_observer.html)

**产品支持情况**：

<!-- npu="910b" id253 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id253 -->
<!-- npu="A3" id254 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id254 -->
<!-- npu="950" id255 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id255 -->

</div>

### torch.ao.quantization.observer.default_per_channel_weight_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_per_channel_weight_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_per_channel_weight_observer.html)

**产品支持情况**：

<!-- npu="910b" id256 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id256 -->
<!-- npu="A3" id257 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id257 -->
<!-- npu="950" id258 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id258 -->

</div>

### torch.ao.quantization.observer.default_dynamic_quant_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_dynamic_quant_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_dynamic_quant_observer.html)

**产品支持情况**：

<!-- npu="910b" id259 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id259 -->
<!-- npu="A3" id260 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id260 -->
<!-- npu="950" id261 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id261 -->

</div>

### torch.ao.quantization.observer.default_float_qparams_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_float_qparams_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_float_qparams_observer.html)

**产品支持情况**：

<!-- npu="910b" id262 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id262 -->
<!-- npu="A3" id263 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id263 -->
<!-- npu="950" id264 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id264 -->

</div>

## torch.ao.quantization.fake_quantize

### <code><i>class</i></code> torch.ao.quantization.fake_quantize.FakeQuantize

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.FakeQuantize](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)

**产品支持情况**：

<!-- npu="910b" id265 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id265 -->
<!-- npu="A3" id266 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id266 -->
<!-- npu="950" id267 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id267 -->

**限制与说明**：可能回退至CPU执行

</div>

### <code><i>class</i></code> torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize.html)

**产品支持情况**：

<!-- npu="910b" id268 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id268 -->
<!-- npu="A3" id269 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id269 -->
<!-- npu="950" id270 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id270 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize.html)

**产品支持情况**：

<!-- npu="910b" id271 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id271 -->
<!-- npu="A3" id272 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id272 -->
<!-- npu="950" id273 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id273 -->

**限制与说明**：可能回退至CPU执行

</div>

### torch.ao.quantization.fake_quantize.disable_fake_quant

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.disable_fake_quant](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.disable_fake_quant.html)

**产品支持情况**：

<!-- npu="910b" id274 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id274 -->
<!-- npu="A3" id275 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id275 -->
<!-- npu="950" id276 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id276 -->

</div>

### torch.ao.quantization.fake_quantize.enable_fake_quant

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.enable_fake_quant](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.enable_fake_quant.html)

**产品支持情况**：

<!-- npu="910b" id277 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id277 -->
<!-- npu="A3" id278 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id278 -->
<!-- npu="950" id279 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id279 -->

</div>

### torch.ao.quantization.fake_quantize.disable_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.disable_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.disable_observer.html)

**产品支持情况**：

<!-- npu="910b" id280 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id280 -->
<!-- npu="A3" id281 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id281 -->
<!-- npu="950" id282 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id282 -->

</div>

### torch.ao.quantization.fake_quantize.enable_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.enable_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.enable_observer.html)

**产品支持情况**：

<!-- npu="910b" id283 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id283 -->
<!-- npu="A3" id284 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id284 -->
<!-- npu="950" id285 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id285 -->

</div>

## torch.ao.quantization.qconfig

### <code><i>class</i></code> torch.ao.quantization.qconfig.QConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.QConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.QConfig.html)

**产品支持情况**：

<!-- npu="910b" id286 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id286 -->
<!-- npu="A3" id287 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id287 -->
<!-- npu="950" id288 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id288 -->

</div>

### torch.ao.quantization.qconfig.default_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id289 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id289 -->
<!-- npu="A3" id290 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id290 -->
<!-- npu="950" id291 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id291 -->

</div>

### torch.ao.quantization.qconfig.default_debug_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_debug_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_debug_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id292 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id292 -->
<!-- npu="A3" id293 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id293 -->
<!-- npu="950" id294 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id294 -->

</div>

### torch.ao.quantization.qconfig.default_per_channel_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_per_channel_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_per_channel_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id295 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id295 -->
<!-- npu="A3" id296 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id296 -->
<!-- npu="950" id297 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id297 -->

</div>

### torch.ao.quantization.qconfig.default_dynamic_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_dynamic_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_dynamic_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id298 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id298 -->
<!-- npu="A3" id299 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id299 -->
<!-- npu="950" id300 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id300 -->

</div>

### torch.ao.quantization.qconfig.float16_dynamic_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.float16_dynamic_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.float16_dynamic_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id301 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id301 -->
<!-- npu="A3" id302 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id302 -->
<!-- npu="950" id303 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id303 -->

</div>

### torch.ao.quantization.qconfig.float16_static_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.float16_static_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.float16_static_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id304 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id304 -->
<!-- npu="A3" id305 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id305 -->
<!-- npu="950" id306 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id306 -->

</div>

### torch.ao.quantization.qconfig.per_channel_dynamic_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.per_channel_dynamic_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.per_channel_dynamic_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id307 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id307 -->
<!-- npu="A3" id308 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id308 -->
<!-- npu="950" id309 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id309 -->

</div>

### torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id310 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id310 -->
<!-- npu="A3" id311 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id311 -->
<!-- npu="950" id312 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id312 -->

</div>

### torch.ao.quantization.qconfig.default_qat_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_qat_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_qat_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id313 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id313 -->
<!-- npu="A3" id314 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id314 -->
<!-- npu="950" id315 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id315 -->

</div>

### torch.ao.quantization.qconfig.default_weight_only_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_weight_only_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_weight_only_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id316 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id316 -->
<!-- npu="A3" id317 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id317 -->
<!-- npu="950" id318 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id318 -->

</div>

### torch.ao.quantization.qconfig.default_activation_only_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_activation_only_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_activation_only_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id319 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id319 -->
<!-- npu="A3" id320 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id320 -->
<!-- npu="950" id321 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id321 -->

</div>

### torch.ao.quantization.qconfig.default_qat_qconfig_v2

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_qat_qconfig_v2](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_qat_qconfig_v2.html)

**产品支持情况**：

<!-- npu="910b" id322 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id322 -->
<!-- npu="A3" id323 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id323 -->
<!-- npu="950" id324 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id324 -->

</div>

## torch.ao.nn.intrinsic

### <code><i>class</i></code> torch.ao.nn.intrinsic.LinearReLU

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.LinearReLU](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.LinearReLU.html)

**产品支持情况**：

<!-- npu="910b" id325 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id325 -->
<!-- npu="A3" id326 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id326 -->
<!-- npu="950" id327 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id327 -->

</div>

## torch.ao.nn.intrinsic.qat

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.LinearReLU

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.LinearReLU](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.LinearReLU.html)

**产品支持情况**：

<!-- npu="910b" id328 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id328 -->
<!-- npu="A3" id329 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id329 -->
<!-- npu="950" id330 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id330 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBn1d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.ConvBn1d](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.ConvBn1d.html)

**产品支持情况**：

<!-- npu="910b" id331 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id331 -->
<!-- npu="A3" id332 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id332 -->
<!-- npu="950" id333 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id333 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBnReLU1d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.ConvBnReLU1d](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.ConvBnReLU1d.html)

**产品支持情况**：

<!-- npu="910b" id334 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id334 -->
<!-- npu="A3" id335 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id335 -->
<!-- npu="950" id336 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id336 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBnReLU2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.ConvBnReLU2d](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.ConvBnReLU2d.html)

**产品支持情况**：

<!-- npu="910b" id337 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id337 -->
<!-- npu="A3" id338 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id338 -->
<!-- npu="950" id339 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id339 -->

</div>

### torch.ao.nn.intrinsic.qat.update_bn_stats

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.update_bn_stats](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.update_bn_stats.html)

**产品支持情况**：

<!-- npu="910b" id340 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id340 -->
<!-- npu="A3" id341 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id341 -->
<!-- npu="950" id342 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id342 -->

</div>

### torch.ao.nn.intrinsic.qat.freeze_bn_stats

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.freeze_bn_stats](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.freeze_bn_stats.html)

**产品支持情况**：

<!-- npu="910b" id343 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id343 -->
<!-- npu="A3" id344 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id344 -->
<!-- npu="950" id345 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id345 -->

**限制与说明**：可能回退至CPU执行

</div>

## torch.ao.nn.qat

### <code><i>class</i></code> torch.ao.nn.qat.Linear

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.qat.Linear](https://pytorch.org/docs/2.7/generated/torch.ao.nn.qat.Linear.html)

**产品支持情况**：

<!-- npu="910b" id346 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id346 -->
<!-- npu="A3" id347 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id347 -->
<!-- npu="950" id348 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id348 -->

</div>

## torch.ao.nn.quantizable

### <code><i>class</i></code> torch.ao.nn.quantizable.LSTM

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantizable.LSTM](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantizable.LSTM.html)

**产品支持情况**：

<!-- npu="910b" id349 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id349 -->
<!-- npu="A3" id350 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id350 -->
<!-- npu="950" id351 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id351 -->

</div>

## torch.ao.nn.quantized.dynamic

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.Linear

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.Linear](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.Linear.html)

**产品支持情况**：

<!-- npu="910b" id352 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id352 -->
<!-- npu="A3" id353 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id353 -->
<!-- npu="950" id354 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id354 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.LSTM

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.LSTM](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.LSTM.html)

**产品支持情况**：

<!-- npu="910b" id355 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id355 -->
<!-- npu="A3" id356 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id356 -->
<!-- npu="950" id357 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id357 -->

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.GRU

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.GRU](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.GRU.html)

**产品支持情况**：

<!-- npu="910b" id358 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id358 -->
<!-- npu="A3" id359 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id359 -->
<!-- npu="950" id360 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id360 -->

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.RNNCell

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.RNNCell](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.RNNCell.html)

**产品支持情况**：

<!-- npu="910b" id361 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id361 -->
<!-- npu="A3" id362 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id362 -->
<!-- npu="950" id363 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id363 -->

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.LSTMCell

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.LSTMCell](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.LSTMCell.html)

**产品支持情况**：

<!-- npu="910b" id364 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id364 -->
<!-- npu="A3" id365 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id365 -->
<!-- npu="950" id366 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id366 -->

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.GRUCell

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.GRUCell](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.GRUCell.html)

**产品支持情况**：

<!-- npu="910b" id367 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id367 -->
<!-- npu="A3" id368 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id368 -->
<!-- npu="950" id369 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id369 -->

**限制与说明**： `input`仅支持fp32

</div>

## Numerical Debugging Tooling

### torch.ao.ns._numeric_suite.compare_weights

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.compare_weights](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.compare_weights)

**产品支持情况**：

<!-- npu="910b" id370 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id370 -->
<!-- npu="A3" id371 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id371 -->
<!-- npu="950" id372 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id372 -->

</div>

### torch.ao.ns._numeric_suite.get_logger_dict

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.get_logger_dict](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.get_logger_dict)

**产品支持情况**：

<!-- npu="910b" id373 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id373 -->
<!-- npu="A3" id374 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id374 -->
<!-- npu="950" id375 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id375 -->

</div>

### <code><i>class</i></code> torch.ao.ns._numeric_suite.Logger

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.Logger](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Logger)

**产品支持情况**：

<!-- npu="910b" id376 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id376 -->
<!-- npu="A3" id377 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id377 -->
<!-- npu="950" id378 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id378 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.Logger.forward](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Logger.forward)

**产品支持情况**：

<!-- npu="910b" id379 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id379 -->
<!-- npu="A3" id380 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id380 -->
<!-- npu="950" id381 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id381 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.ns._numeric_suite.ShadowLogger

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.ShadowLogger](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.ShadowLogger)

**产品支持情况**：

<!-- npu="910b" id382 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id382 -->
<!-- npu="A3" id383 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id383 -->
<!-- npu="950" id384 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id384 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.ShadowLogger.forward](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.ShadowLogger.forward)

**产品支持情况**：

<!-- npu="910b" id385 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id385 -->
<!-- npu="A3" id386 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id386 -->
<!-- npu="950" id387 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id387 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.ns._numeric_suite.OutputLogger

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.OutputLogger](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.OutputLogger)

**产品支持情况**：

<!-- npu="910b" id388 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id388 -->
<!-- npu="A3" id389 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id389 -->
<!-- npu="950" id390 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id390 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.OutputLogger.forward](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.OutputLogger.forward)

**产品支持情况**：

<!-- npu="910b" id391 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id391 -->
<!-- npu="A3" id392 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id392 -->
<!-- npu="950" id393 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id393 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.ns._numeric_suite.Shadow

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.Shadow](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Shadow)

**产品支持情况**：

<!-- npu="910b" id394 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id394 -->
<!-- npu="A3" id395 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id395 -->
<!-- npu="950" id396 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id396 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.Shadow.forward](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Shadow.forward)

**产品支持情况**：

<!-- npu="910b" id397 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id397 -->
<!-- npu="A3" id398 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id398 -->
<!-- npu="950" id399 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id399 -->

</div>

> <font size="3">add()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.Shadow.add](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Shadow.add)

**产品支持情况**：

<!-- npu="910b" id400 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id400 -->
<!-- npu="A3" id401 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id401 -->
<!-- npu="950" id402 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id402 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_scalar()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.Shadow.add_scalar](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Shadow.add_scalar)

**产品支持情况**：

<!-- npu="910b" id403 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id403 -->
<!-- npu="A3" id404 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id404 -->
<!-- npu="950" id405 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id405 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">mul()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.Shadow.mul](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Shadow.mul)

**产品支持情况**：

<!-- npu="910b" id406 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id406 -->
<!-- npu="A3" id407 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id407 -->
<!-- npu="950" id408 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id408 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">mul_scalar()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.Shadow.mul_scalar](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Shadow.mul_scalar)

**产品支持情况**：

<!-- npu="910b" id409 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id409 -->
<!-- npu="A3" id410 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id410 -->
<!-- npu="950" id411 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id411 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">cat()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.Shadow.cat](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Shadow.cat)

**产品支持情况**：

<!-- npu="910b" id412 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id412 -->
<!-- npu="A3" id413 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id413 -->
<!-- npu="950" id414 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id414 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_relu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.Shadow.add_relu](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Shadow.add_relu)

**产品支持情况**：

<!-- npu="910b" id415 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id415 -->
<!-- npu="A3" id416 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id416 -->
<!-- npu="950" id417 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id417 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int32，int64

</div>

</div>

### torch.ao.ns._numeric_suite.prepare_model_with_stubs

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.prepare_model_with_stubs](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.prepare_model_with_stubs)

**产品支持情况**：

<!-- npu="910b" id418 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id418 -->
<!-- npu="A3" id419 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id419 -->
<!-- npu="950" id420 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id420 -->

</div>

### torch.ao.ns._numeric_suite.compare_model_stub

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.compare_model_stub](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.compare_model_stub)

**产品支持情况**：

<!-- npu="910b" id421 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id421 -->
<!-- npu="A3" id422 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id422 -->
<!-- npu="950" id423 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id423 -->

</div>

### torch.ao.ns._numeric_suite.get_matching_activations

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.get_matching_activations](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.get_matching_activations)

**产品支持情况**：

<!-- npu="910b" id424 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id424 -->
<!-- npu="A3" id425 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id425 -->
<!-- npu="950" id426 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id426 -->

</div>

### torch.ao.ns._numeric_suite.prepare_model_outputs

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.prepare_model_outputs](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.prepare_model_outputs)

**产品支持情况**：

<!-- npu="910b" id427 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id427 -->
<!-- npu="A3" id428 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id428 -->
<!-- npu="950" id429 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id429 -->

</div>

### torch.ao.ns._numeric_suite.compare_model_outputs

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite.compare_model_outputs](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.compare_model_outputs)

**产品支持情况**：

<!-- npu="910b" id430 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id430 -->
<!-- npu="A3" id431 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id431 -->
<!-- npu="950" id432 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id432 -->

</div>

### <code><i>class</i></code> torch.ao.ns._numeric_suite_fx.OutputLogger

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.OutputLogger](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.OutputLogger)

**产品支持情况**：

<!-- npu="910b" id433 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id433 -->
<!-- npu="A3" id434 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id434 -->
<!-- npu="950" id435 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id435 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.OutputLogger.forward](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.OutputLogger.forward)

**产品支持情况**：

<!-- npu="910b" id436 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id436 -->
<!-- npu="A3" id437 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id437 -->
<!-- npu="950" id438 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id438 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.ns._numeric_suite_fx.OutputComparisonLogger

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.OutputComparisonLogger](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.OutputComparisonLogger)

**产品支持情况**：

<!-- npu="910b" id439 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id439 -->
<!-- npu="A3" id440 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id440 -->
<!-- npu="950" id441 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id441 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.OutputComparisonLogger.forward](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.OutputComparisonLogger.forward)

**产品支持情况**：

<!-- npu="910b" id442 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id442 -->
<!-- npu="A3" id443 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id443 -->
<!-- npu="950" id444 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id444 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.ns._numeric_suite_fx.NSTracer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.NSTracer](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.NSTracer)

**产品支持情况**：

<!-- npu="910b" id445 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id445 -->
<!-- npu="A3" id446 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id446 -->
<!-- npu="950" id447 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id447 -->

> <font size="3">is_leaf_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.NSTracer.is_leaf_module](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.NSTracer.is_leaf_module)

**产品支持情况**：

<!-- npu="910b" id448 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id448 -->
<!-- npu="A3" id449 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id449 -->
<!-- npu="950" id450 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id450 -->

</div>

</div>

### torch.ao.ns._numeric_suite_fx.extract_weights

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.extract_weights](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.extract_weights)

**产品支持情况**：

<!-- npu="910b" id451 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id451 -->
<!-- npu="A3" id452 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id452 -->
<!-- npu="950" id453 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id453 -->

</div>

### torch.ao.ns._numeric_suite_fx.add_loggers

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.add_loggers](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.add_loggers)

**产品支持情况**：

<!-- npu="910b" id454 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id454 -->
<!-- npu="A3" id455 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id455 -->
<!-- npu="950" id456 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id456 -->

</div>

### torch.ao.ns._numeric_suite_fx.extract_logger_info

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.extract_logger_info](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.extract_logger_info)

**产品支持情况**：

<!-- npu="910b" id457 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id457 -->
<!-- npu="A3" id458 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id458 -->
<!-- npu="950" id459 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id459 -->

</div>

### torch.ao.ns._numeric_suite_fx.add_shadow_loggers

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.add_shadow_loggers](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.add_shadow_loggers)

**产品支持情况**：

<!-- npu="910b" id460 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id460 -->
<!-- npu="A3" id461 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id461 -->
<!-- npu="950" id462 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id462 -->

</div>

### torch.ao.ns._numeric_suite_fx.extract_shadow_logger_info

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.extract_shadow_logger_info](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.extract_shadow_logger_info)

**产品支持情况**：

<!-- npu="910b" id463 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id463 -->
<!-- npu="A3" id464 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id464 -->
<!-- npu="950" id465 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id465 -->

</div>

### torch.ao.ns._numeric_suite_fx.extend_logger_results_with_comparison

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.extend_logger_results_with_comparison](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.extend_logger_results_with_comparison)

**产品支持情况**：

<!-- npu="910b" id466 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id466 -->
<!-- npu="A3" id467 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id467 -->
<!-- npu="950" id468 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id468 -->

</div>

### torch.ao.ns._numeric_suite_fx.prepare_n_shadows_model

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.prepare_n_shadows_model](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.prepare_n_shadows_model)

**产品支持情况**：

<!-- npu="910b" id469 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id469 -->
<!-- npu="A3" id470 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id470 -->
<!-- npu="950" id471 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id471 -->

</div>

### torch.ao.ns._numeric_suite_fx.loggers_set_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.loggers_set_enabled](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.loggers_set_enabled)

**产品支持情况**：

<!-- npu="910b" id472 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id472 -->
<!-- npu="A3" id473 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id473 -->
<!-- npu="950" id474 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id474 -->

</div>

### torch.ao.ns._numeric_suite_fx.loggers_set_save_activations

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.loggers_set_save_activations](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.loggers_set_save_activations)

**产品支持情况**：

<!-- npu="910b" id475 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id475 -->
<!-- npu="A3" id476 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id476 -->
<!-- npu="950" id477 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id477 -->

</div>

### torch.ao.ns._numeric_suite_fx.convert_n_shadows_model

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.convert_n_shadows_model](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.convert_n_shadows_model)

**产品支持情况**：

<!-- npu="910b" id478 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id478 -->
<!-- npu="A3" id479 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id479 -->
<!-- npu="950" id480 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id480 -->

</div>

### torch.ao.ns._numeric_suite_fx.extract_results_n_shadows_model

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.extract_results_n_shadows_model](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.extract_results_n_shadows_model)

**产品支持情况**：

<!-- npu="910b" id481 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id481 -->
<!-- npu="A3" id482 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id482 -->
<!-- npu="950" id483 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id483 -->

</div>

### torch.ao.ns._numeric_suite_fx.print_comparisons_n_shadows_model

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns._numeric_suite_fx.print_comparisons_n_shadows_model](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.print_comparisons_n_shadows_model)

**产品支持情况**：

<!-- npu="910b" id484 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id484 -->
<!-- npu="A3" id485 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id485 -->
<!-- npu="950" id486 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id486 -->

</div>

### torch.ao.ns.fx.utils.compute_sqnr

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns.fx.utils.compute_sqnr](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns.fx.utils.compute_sqnr)

**产品支持情况**：

<!-- npu="910b" id487 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id487 -->
<!-- npu="A3" id488 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id488 -->
<!-- npu="950" id489 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id489 -->

</div>
