# Quantization

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 本支持清单的范围以[PyTorch社区文档](https://pytorch.org/docs/2.9/)为准，社区文档中未列出的接口不在本清单范围内；已在社区文档中存在但未收录于本清单的API尚未验证，请谨慎使用，我们将在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://docs.pytorch.org/docs/2.9/quantization-support.html)。
> - 注：在[Quantization Accuracy Debugging](https://docs.pytorch.org/docs/2.9/quantization.html)中可以找到[Numerical Debugging Tooling](#numerical-debugging-tooling)下API的具体用法。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [torch.ao.quantization](#torchaoquantization)

- [torch.ao.quantization.quantize_fx](#torchaoquantizationquantize_fx)
- [torch.ao.quantization.qconfig_mapping](#torchaoquantizationqconfig_mapping)
- [torch.ao.quantization.backend_config](#torchaoquantizationbackend_config)
- [torch.ao.quantization.fx.custom_config](#torchaoquantizationfxcustom_config)

- [PT2 Export (pt2e) Numeric Debugger](#pt2-export-pt2e-numeric-debugger)
- [torch.ao.quantization.observer](#torchaoquantizationobserver)
- [torch.ao.quantization.fake_quantize](#torchaoquantizationfake_quantize)
- [torch.ao.quantization.qconfig](#torchaoquantizationqconfig)
- [torch.ao.nn.intrinsic](#torchaonnintrinsic)
- [torch.ao.nn.intrinsic.qat](#torchaonnintrinsicqat)

- [torch.ao.nn.intrinsic.quantized](#torchaonnintrinsicquantized)

- [torch.ao.nn.qat](#torchaonnqat)
- [torch.ao.nn.quantized](#torchaonnquantized)

- [torch.ao.nn.quantized.functional](#torchaonnquantizedfunctional)
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

**原生文档**：[torch.ao.quantization.prepare_qat](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.prepare_qat.html)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id3 -->

</div>

<!-- npu="910b,A3" id373 -->
### torch.ao.quantization.convert

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.convert](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.convert.html)

**产品支持情况**：
<!-- end id373 -->

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->

<!-- npu="910b,A3" id374 -->
</div>
<!-- end id374 -->

### <code><i>class</i></code> torch.ao.quantization.fuse\_modules.fuse\_modules

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fuse\_modules.fuse\_modules](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.fuse_modules.fuse_modules.html#torch.ao.quantization.fuse_modules.fuse_modules)

**产品支持情况**：

<!-- npu="910b" id425 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id425 -->
<!-- npu="A3" id426 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id426 -->
<!-- npu="950" id427 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id427 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.QuantStub

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.QuantStub](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.QuantStub.html)

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

**原生文档**：[torch.ao.quantization.DeQuantStub](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.DeQuantStub.html)

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

### <code><i>class</i></code> torch.ao.quantization.QuantWrapper

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.QuantWrapper](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.QuantWrapper.html)

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

### <code><i>class</i></code> torch.ao.quantization.add\_quant\_dequant

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.add\_quant\_dequant](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.add_quant_dequant.html#torch.ao.quantization.add_quant_dequant)

**产品支持情况**：

<!-- npu="910b" id428 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id428 -->
<!-- npu="A3" id429 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id429 -->
<!-- npu="950" id430 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id430 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.swap\_module

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.swap\_module](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.swap_module.html#torch.ao.quantization.swap_module)

**产品支持情况**：

<!-- npu="910b" id431 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id431 -->
<!-- npu="A3" id432 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id432 -->
<!-- npu="950" id433 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id433 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.propagate\_qconfig\_

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.propagate\_qconfig\_](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.propagate_qconfig_.html#torch.ao.quantization.propagate_qconfig_)

**产品支持情况**：

<!-- npu="910b" id434 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id434 -->
<!-- npu="A3" id435 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id435 -->
<!-- npu="950" id436 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id436 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.default\_eval\_fn

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.default\_eval\_fn](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.default_eval_fn.html#torch.ao.quantization.default_eval_fn)

**产品支持情况**：

<!-- npu="910b" id437 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id437 -->
<!-- npu="A3" id438 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id438 -->
<!-- npu="950" id439 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id439 -->

</div>

## torch.ao.quantization.quantize_fx

<!-- npu="910b,A3" id452 -->
### <code><i>class</i></code> torch.ao.quantization.quantize\_fx.fuse\_fx

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.quantize\_fx.fuse\_fx](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.quantize_fx.fuse_fx.html#torch.ao.quantization.quantize_fx.fuse_fx)

**产品支持情况**：

<!-- npu="910b" id449 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id449 -->
<!-- npu="A3" id450 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id450 -->

</div>

<!-- end id452 -->

## torch.ao.quantization.qconfig_mapping

### <code><i>class</i></code> torch.ao.quantization.qconfig_mapping.QConfigMapping

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html)

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

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.from_dict](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.from_dict)

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

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_global](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_global)

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

<!-- npu="910b,A3" id27 -->
> <font size="3">set_module_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->

</div>
<!-- end id27 -->

> <font size="3">set_module_name_object_type_order()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_object_type_order](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_object_type_order)

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

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_regex](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_regex)

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

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_object_type](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_object_type)

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

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.to_dict](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.to_dict)

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

**原生文档**：[torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping.html)

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

**原生文档**：[torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping.html)

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

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendConfig.html)

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

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.configs](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.configs)

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

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.from_dict](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.from_dict)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id54 -->

</div>

> <font size="3">set_backend_pattern_config()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_config](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_config)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id57 -->

</div>

> <font size="3">set_backend_pattern_configs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_configs](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_configs)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id60 -->

</div>

> <font size="3">set_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.set_name](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_name)

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

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.to_dict](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.to_dict)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id66 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.backend_config.BackendPatternConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html)

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

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.add_dtype_config](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.add_dtype_config)

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

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.from_dict](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.from_dict)

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id75 -->

</div>

> <font size="3">set_dtype_configs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_dtype_configs](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_dtype_configs)

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

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_fused_module](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_fused_module)

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

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_fuser_method](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_fuser_method)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id84 -->

</div>

> <font size="3">set_observation_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_observation_type](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_observation_type)

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

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_pattern](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_pattern)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id90 -->

</div>

> <font size="3">set_qat_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_qat_module](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_qat_module)

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

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_reference_quantized_module](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_reference_quantized_module)

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

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module)

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

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.to_dict](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.to_dict)

**产品支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id102 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.backend_config.DTypeConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.DTypeConfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.DTypeConfig.html)

**产品支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id105 -->

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.DTypeConfig.from_dict](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.DTypeConfig.html#torch.ao.quantization.backend_config.DTypeConfig.from_dict)

**产品支持情况**：

<!-- npu="910b" id106 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id108 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.DTypeConfig.to_dict](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.DTypeConfig.html#torch.ao.quantization.backend_config.DTypeConfig.to_dict)

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

**原生文档**：[torch.ao.quantization.backend_config.DTypeWithConstraints](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.DTypeWithConstraints.html)

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

**原生文档**：[torch.ao.quantization.backend_config.ObservationType](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.ObservationType.html)

**产品支持情况**：

<!-- npu="910b" id115 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id117 -->

> <font size="3">INPUT_OUTPUT_NOT_OBSERVED()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.ObservationType.INPUT_OUTPUT_NOT_OBSERVED](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.INPUT_OUTPUT_NOT_OBSERVED)

**产品支持情况**：

<!-- npu="910b" id118 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id120 -->

</div>

> <font size="3">OUTPUT_SHARE_OBSERVER_WITH_INPUT()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)

**产品支持情况**：

<!-- npu="910b" id121 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id123 -->

</div>

> <font size="3">OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)

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

**原生文档**：[torch.ao.quantization.fx.custom_config.FuseCustomConfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html)

**产品支持情况**：

<!-- npu="910b" id127 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id129 -->

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.FuseCustomConfig.from_dict](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.from_dict)

**产品支持情况**：

<!-- npu="910b" id130 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id132 -->

</div>

> <font size="3">set_preserved_attributes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.FuseCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.set_preserved_attributes)

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

**原生文档**：[torch.ao.quantization.fx.custom_config.FuseCustomConfig.to_dict](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.to_dict)

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

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html)

**产品支持情况**：

<!-- npu="910b" id139 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id141 -->

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict)

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

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_float_to_observed_mapping](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_float_to_observed_mapping)

**产品支持情况**：

<!-- npu="910b" id145 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id147 -->

</div>

> <font size="3">set_input_quantized_indexes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_input_quantized_indexes](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_input_quantized_indexes)

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

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_classes](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_classes)

**产品支持情况**：

<!-- npu="910b" id151 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id153 -->

</div>

> <font size="3">set_non_traceable_module_names()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_names](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_names)

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

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_output_quantized_indexes](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_output_quantized_indexes)

**产品支持情况**：

<!-- npu="910b" id157 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id159 -->

</div>

> <font size="3">set_preserved_attributes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_preserved_attributes)

**产品支持情况**：

<!-- npu="910b" id160 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id162 -->

</div>

> <font size="3">set_standalone_module_class()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_class](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_class)

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

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_name](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_name)

**产品支持情况**：

<!-- npu="910b" id166 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id168 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.to_dict](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.to_dict)

**产品支持情况**：

<!-- npu="910b" id169 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id171 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.fx.custom_config.ConvertCustomConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html)

**产品支持情况**：

<!-- npu="910b" id172 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id174 -->

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict)

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

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_observed_to_quantized_mapping](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_observed_to_quantized_mapping)

**产品支持情况**：

<!-- npu="910b" id178 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id180 -->

</div>

> <font size="3">set_preserved_attributes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_preserved_attributes)

**产品支持情况**：

<!-- npu="910b" id181 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id183 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.to_dict](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.to_dict)

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

**原生文档**：[torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry.html)

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

## PT2 Export (pt2e) Numeric Debugger

### <code><i>class</i></code> torch.ao.quantization.prepare\_for\_propagation\_comparison

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.prepare\_for\_propagation\_comparison](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.prepare_for_propagation_comparison.html#torch.ao.quantization.prepare_for_propagation_comparison)

**产品支持情况**：

<!-- npu="910b" id453 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id453 -->
<!-- npu="A3" id454 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id454 -->
<!-- npu="950" id455 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id455 -->

</div>

<!-- npu="910b,A3" id459 -->
### <code><i>class</i></code> torch.ao.quantization.extract\_results\_from\_loggers

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.extract\_results\_from\_loggers](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.extract_results_from_loggers.html#torch.ao.quantization.extract_results_from_loggers)

**产品支持情况**：

<!-- npu="910b" id456 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id456 -->
<!-- npu="A3" id457 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id457 -->

</div>

<!-- end id459 -->

### <code><i>class</i></code> torch.ao.quantization.compare\_results

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.compare\_results](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.compare_results.html#torch.ao.quantization.compare_results)

**产品支持情况**：

<!-- npu="910b" id460 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id460 -->
<!-- npu="A3" id461 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id461 -->
<!-- npu="950" id462 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id462 -->

</div>

## torch.ao.quantization.observer

### <code><i>class</i></code> torch.ao.quantization.pt2e.export_utils.model_is_exported

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.pt2e.export_utils.model_is_exported](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.pt2e.export_utils.model_is_exported.html#torch.ao.quantization.pt2e.export_utils.model_is_exported)

**产品支持情况**：

<!-- npu="910b" id413 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id413 -->
<!-- npu="A3" id414 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id414 -->
<!-- npu="950" id415 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id415 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.generate_numeric_debug_handle

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.generate_numeric_debug_handle](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.generate_numeric_debug_handle.html#torch.ao.quantization.generate_numeric_debug_handle)

**产品支持情况**：

<!-- npu="910b" id401 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id401 -->
<!-- npu="A3" id402 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id402 -->
<!-- npu="950" id403 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id403 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.ObserverBase

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.ObserverBase](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.ObserverBase.html)

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

**原生文档**：[torch.ao.quantization.observer.ObserverBase.with_args](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.ObserverBase.html#torch.ao.quantization.observer.ObserverBase.with_args)

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

**原生文档**：[torch.ao.quantization.observer.ObserverBase.with_callable_args](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.ObserverBase.html#torch.ao.quantization.observer.ObserverBase.with_callable_args)

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

**原生文档**：[torch.ao.quantization.observer.MinMaxObserver](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.MinMaxObserver.html)

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

**原生文档**：[torch.ao.quantization.observer.MinMaxObserver.calculate_qparams](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.calculate_qparams)

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

**原生文档**：[torch.ao.quantization.observer.MinMaxObserver.forward](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.forward)

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

**原生文档**：[torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals)

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

**原生文档**：[torch.ao.quantization.observer.MovingAverageMinMaxObserver](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.MovingAverageMinMaxObserver.html)

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

**原生文档**：[torch.ao.quantization.observer.PerChannelMinMaxObserver](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.PerChannelMinMaxObserver.html)

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

**原生文档**：[torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.PerChannelMinMaxObserver.html#torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals)

**产品支持情况**：

<!-- npu="910b" id217 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id219 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver.html)

**产品支持情况**：

<!-- npu="910b" id220 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id222 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.HistogramObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.HistogramObserver](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.HistogramObserver.html)

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

**原生文档**：[torch.ao.quantization.observer.PlaceholderObserver](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.PlaceholderObserver.html)

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

**原生文档**：[torch.ao.quantization.observer.RecordingObserver](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.RecordingObserver.html)

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

**原生文档**：[torch.ao.quantization.observer.NoopObserver](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.NoopObserver.html)

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

**原生文档**：[torch.ao.quantization.observer.get_observer_state_dict](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.get_observer_state_dict.html)

**产品支持情况**：

<!-- npu="910b" id235 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id235 -->
<!-- npu="A3" id236 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id236 -->
<!-- npu="950" id237 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id237 -->

</div>

### torch.ao.quantization.observer.load_observer_state_dict

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.load_observer_state_dict](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.load_observer_state_dict.html)

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

**原生文档**：[torch.ao.quantization.observer.default_observer](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.default_observer.html)

**产品支持情况**：

<!-- npu="910b" id241 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id241 -->
<!-- npu="A3" id242 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id242 -->
<!-- npu="950" id243 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id243 -->

</div>

### torch.ao.quantization.observer.default_placeholder_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_placeholder_observer](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.default_placeholder_observer.html)

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

**原生文档**：[torch.ao.quantization.observer.default_debug_observer](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.default_debug_observer.html)

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

**原生文档**：[torch.ao.quantization.observer.default_weight_observer](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.default_weight_observer.html)

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

**原生文档**：[torch.ao.quantization.observer.default_histogram_observer](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.default_histogram_observer.html)

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

**原生文档**：[torch.ao.quantization.observer.default_per_channel_weight_observer](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.default_per_channel_weight_observer.html)

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

**原生文档**：[torch.ao.quantization.observer.default_dynamic_quant_observer](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.default_dynamic_quant_observer.html)

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

**原生文档**：[torch.ao.quantization.observer.default_float_qparams_observer](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.default_float_qparams_observer.html)

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

### <code><i>class</i></code> torch.ao.quantization.observer.Granularity

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.Granularity](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.Granularity.html#torch.ao.quantization.observer.Granularity)

**产品支持情况**：

<!-- npu="910b" id608 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id608 -->
<!-- npu="A3" id609 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id609 -->
<!-- npu="950" id610 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id610 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.PerAxis

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.PerAxis](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.PerAxis.html#torch.ao.quantization.observer.PerAxis)

**产品支持情况**：

<!-- npu="910b" id611 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id611 -->
<!-- npu="A3" id612 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id612 -->
<!-- npu="950" id613 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id613 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.PerBlock

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.PerBlock](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.PerBlock.html#torch.ao.quantization.observer.PerBlock)

**产品支持情况**：

<!-- npu="910b" id614 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id614 -->
<!-- npu="A3" id615 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id615 -->
<!-- npu="950" id616 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id616 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.PerRow

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.PerRow](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.PerRow.html#torch.ao.quantization.observer.PerRow)

**产品支持情况**：

<!-- npu="910b" id617 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id617 -->
<!-- npu="A3" id618 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id618 -->
<!-- npu="950" id619 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id619 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.PerTensor

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.PerTensor](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.PerTensor.html#torch.ao.quantization.observer.PerTensor)

**产品支持情况**：

<!-- npu="910b" id620 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id620 -->
<!-- npu="A3" id621 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id621 -->
<!-- npu="950" id622 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id622 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.PerToken

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.PerToken](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.PerToken.html#torch.ao.quantization.observer.PerToken)

**产品支持情况**：

<!-- npu="910b" id623 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id623 -->
<!-- npu="A3" id624 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id624 -->
<!-- npu="950" id625 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id625 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.TorchAODType

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.TorchAODType](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.TorchAODType.html#torch.ao.quantization.observer.TorchAODType)

**产品支持情况**：

<!-- npu="910b" id626 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id626 -->
<!-- npu="A3" id627 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id627 -->
<!-- npu="950" id628 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id628 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.get\_block\_size

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.get\_block\_size](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.get_block_size.html#torch.ao.quantization.observer.get_block_size)

**产品支持情况**：

<!-- npu="910b" id629 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id629 -->
<!-- npu="A3" id630 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id630 -->
<!-- npu="950" id631 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id631 -->

</div>

## torch.ao.quantization.fake_quantize

### <code><i>class</i></code> torch.ao.quantization.observer.AffineQuantizedObserverBase

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.AffineQuantizedObserverBase](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.AffineQuantizedObserverBase.html#torch.ao.quantization.observer.AffineQuantizedObserverBase)

**产品支持情况**：

<!-- npu="910b" id416 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id416 -->
<!-- npu="A3" id417 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id417 -->
<!-- npu="950" id418 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id418 -->

> <font size="3">calculate_qparams()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.AffineQuantizedObserverBase.calculate_qparams](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.AffineQuantizedObserverBase.html#torch.ao.quantization.observer.AffineQuantizedObserverBase.calculate_qparams)

**产品支持情况**：

<!-- npu="910b" id404 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id404 -->
<!-- npu="A3" id405 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id405 -->
<!-- npu="950" id406 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id406 -->

</div>

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.AffineQuantizedObserverBase.forward](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.AffineQuantizedObserverBase.html#torch.ao.quantization.observer.AffineQuantizedObserverBase.forward)

**产品支持情况**：

<!-- npu="910b" id407 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id407 -->
<!-- npu="A3" id408 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id408 -->
<!-- npu="950" id409 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id409 -->

</div>

> <font size="3">with_args()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.AffineQuantizedObserverBase.with\_args](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.AffineQuantizedObserverBase.html#torch.ao.quantization.observer.AffineQuantizedObserverBase.with_args)

**产品支持情况**：

<!-- npu="910b" id666 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id666 -->
<!-- npu="A3" id667 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id667 -->
<!-- npu="950" id668 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id668 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.MappingType

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MappingType](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.observer.MappingType.html#torch.ao.quantization.observer.MappingType)

**产品支持情况**：

<!-- npu="910b" id410 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id410 -->
<!-- npu="A3" id411 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id411 -->
<!-- npu="950" id412 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id412 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.fake\_quantize.FakeQuantizeBase

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake\_quantize.FakeQuantizeBase](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.fake_quantize.FakeQuantizeBase.html#torch.ao.quantization.fake_quantize.FakeQuantizeBase)

**产品支持情况**：

<!-- npu="910b" id605 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id605 -->
<!-- npu="A3" id606 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id606 -->
<!-- npu="950" id607 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id607 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.fake_quantize.FakeQuantize

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.FakeQuantize](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)

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

**原生文档**：[torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize.html)

**产品支持情况**：

<!-- npu="910b" id268 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id268 -->
<!-- npu="A3" id269 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id269 -->
<!-- npu="950" id270 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id270 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize.html)

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

**原生文档**：[torch.ao.quantization.fake_quantize.disable_fake_quant](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fake_quantize.disable_fake_quant.html)

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

**原生文档**：[torch.ao.quantization.fake_quantize.enable_fake_quant](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fake_quantize.enable_fake_quant.html)

**产品支持情况**：

<!-- npu="910b" id277 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id277 -->
<!-- npu="A3" id278 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id278 -->
<!-- npu="950" id279 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id279 -->

</div>

### torch.ao.quantization.fake_quantize.disable_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.disable_observer](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fake_quantize.disable_observer.html)

**产品支持情况**：

<!-- npu="910b" id280 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id280 -->
<!-- npu="A3" id281 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id281 -->
<!-- npu="950" id282 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id282 -->

</div>

### torch.ao.quantization.fake_quantize.enable_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.enable_observer](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.fake_quantize.enable_observer.html)

**产品支持情况**：

<!-- npu="910b" id283 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id283 -->
<!-- npu="A3" id284 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id284 -->
<!-- npu="950" id285 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id285 -->

</div>

### torch.ao.quantization.fake\_quantize.default\_fake\_quant

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake\_quantize.default\_fake\_quant](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.fake_quantize.default_fake_quant.html#torch.ao.quantization.fake_quantize.default_fake_quant)

**产品支持情况**：

<!-- npu="910b" id645 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id645 -->
<!-- npu="A3" id646 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id646 -->
<!-- npu="950" id647 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id647 -->

</div>

### torch.ao.quantization.fake\_quantize.default\_weight\_fake\_quant

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake\_quantize.default\_weight\_fake\_quant](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.fake_quantize.default_weight_fake_quant.html#torch.ao.quantization.fake_quantize.default_weight_fake_quant)

**产品支持情况**：

<!-- npu="910b" id663 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id663 -->
<!-- npu="A3" id664 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id664 -->
<!-- npu="950" id665 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id665 -->

</div>

### torch.ao.quantization.fake\_quantize.default\_per\_channel\_weight\_fake\_quant

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake\_quantize.default\_per\_channel\_weight\_fake\_quant](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.fake_quantize.default_per_channel_weight_fake_quant.html#torch.ao.quantization.fake_quantize.default_per_channel_weight_fake_quant)

**产品支持情况**：

<!-- npu="910b" id660 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id660 -->
<!-- npu="A3" id661 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id661 -->
<!-- npu="950" id662 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id662 -->

</div>

### torch.ao.quantization.fake\_quantize.default\_histogram\_fake\_quant

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake\_quantize.default\_histogram\_fake\_quant](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.fake_quantize.default_histogram_fake_quant.html#torch.ao.quantization.fake_quantize.default_histogram_fake_quant)

**产品支持情况**：

<!-- npu="910b" id657 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id657 -->
<!-- npu="A3" id658 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id658 -->
<!-- npu="950" id659 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id659 -->

</div>

### torch.ao.quantization.fake\_quantize.default\_fused\_act\_fake\_quant

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake\_quantize.default\_fused\_act\_fake\_quant](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.fake_quantize.default_fused_act_fake_quant.html#torch.ao.quantization.fake_quantize.default_fused_act_fake_quant)

**产品支持情况**：

<!-- npu="910b" id648 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id648 -->
<!-- npu="A3" id649 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id649 -->
<!-- npu="950" id650 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id650 -->

</div>

### torch.ao.quantization.fake\_quantize.default\_fused\_wt\_fake\_quant

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake\_quantize.default\_fused\_wt\_fake\_quant](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.fake_quantize.default_fused_wt_fake_quant.html#torch.ao.quantization.fake_quantize.default_fused_wt_fake_quant)

**产品支持情况**：

<!-- npu="910b" id654 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id654 -->
<!-- npu="A3" id655 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id655 -->
<!-- npu="950" id656 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id656 -->

</div>

### torch.ao.quantization.fake\_quantize.default\_fused\_per\_channel\_wt\_fake\_quant

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake\_quantize.default\_fused\_per\_channel\_wt\_fake\_quant](https://docs.pytorch.org/docs/2.9/generated/torch.ao.quantization.fake_quantize.default_fused_per_channel_wt_fake_quant.html#torch.ao.quantization.fake_quantize.default_fused_per_channel_wt_fake_quant)

**产品支持情况**：

<!-- npu="910b" id651 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id651 -->
<!-- npu="A3" id652 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id652 -->
<!-- npu="950" id653 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id653 -->

</div>

## torch.ao.quantization.qconfig

### <code><i>class</i></code> torch.ao.quantization.qconfig.QConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.QConfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig.QConfig.html)

**产品支持情况**：

<!-- npu="910b" id286 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id286 -->
<!-- npu="A3" id287 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id287 -->
<!-- npu="950" id288 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id288 -->

</div>

### torch.ao.quantization.qconfig.default_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_qconfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig.default_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id289 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id289 -->
<!-- npu="A3" id290 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id290 -->
<!-- npu="950" id291 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id291 -->

</div>

### torch.ao.quantization.qconfig.default_debug_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_debug_qconfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig.default_debug_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id292 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id292 -->
<!-- npu="A3" id293 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id293 -->
<!-- npu="950" id294 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id294 -->

</div>

### torch.ao.quantization.qconfig.default_per_channel_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_per_channel_qconfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig.default_per_channel_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id295 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id295 -->
<!-- npu="A3" id296 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id296 -->
<!-- npu="950" id297 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id297 -->

</div>

### torch.ao.quantization.qconfig.default_dynamic_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_dynamic_qconfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig.default_dynamic_qconfig.html)

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

**原生文档**：[torch.ao.quantization.qconfig.float16_dynamic_qconfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig.float16_dynamic_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id301 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id301 -->
<!-- npu="A3" id302 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id302 -->
<!-- npu="950" id303 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id303 -->

</div>

### torch.ao.quantization.qconfig.float16_static_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.float16_static_qconfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig.float16_static_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id304 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id304 -->
<!-- npu="A3" id305 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id305 -->
<!-- npu="950" id306 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id306 -->

</div>

### torch.ao.quantization.qconfig.per_channel_dynamic_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.per_channel_dynamic_qconfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig.per_channel_dynamic_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id307 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id307 -->
<!-- npu="A3" id308 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id308 -->
<!-- npu="950" id309 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id309 -->

</div>

### torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig.html)

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

**原生文档**：[torch.ao.quantization.qconfig.default_qat_qconfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig.default_qat_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id313 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id313 -->
<!-- npu="A3" id314 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id314 -->
<!-- npu="950" id315 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id315 -->

</div>

### torch.ao.quantization.qconfig.default_weight_only_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_weight_only_qconfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig.default_weight_only_qconfig.html)

**产品支持情况**：

<!-- npu="910b" id316 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id316 -->
<!-- npu="A3" id317 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id317 -->
<!-- npu="950" id318 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id318 -->

</div>

### torch.ao.quantization.qconfig.default_activation_only_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_activation_only_qconfig](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig.default_activation_only_qconfig.html)

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

**原生文档**：[torch.ao.quantization.qconfig.default_qat_qconfig_v2](https://pytorch.org/docs/2.9/generated/torch.ao.quantization.qconfig.default_qat_qconfig_v2.html)

**产品支持情况**：

<!-- npu="910b" id322 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id322 -->
<!-- npu="A3" id323 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id323 -->
<!-- npu="950" id324 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id324 -->

</div>

## torch.ao.nn.intrinsic

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvReLU1d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.ConvReLU1d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.ConvReLU1d.html#torch.ao.nn.intrinsic.ConvReLU1d)

**产品支持情况**：

<!-- npu="910b" id386 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id386 -->
<!-- npu="A3" id387 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id387 -->
<!-- npu="950" id388 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id388 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvReLU2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.ConvReLU2d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.ConvReLU2d.html#torch.ao.nn.intrinsic.ConvReLU2d)

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

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvReLU3d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.ConvReLU3d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.ConvReLU3d.html#torch.ao.nn.intrinsic.ConvReLU3d)

**产品支持情况**：

<!-- npu="910b" id466 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id466 -->
<!-- npu="A3" id467 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id467 -->
<!-- npu="950" id468 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id468 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.LinearReLU

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.LinearReLU](https://pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.LinearReLU.html)

**产品支持情况**：

<!-- npu="910b" id325 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id325 -->
<!-- npu="A3" id326 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id326 -->
<!-- npu="950" id327 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id327 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvBn1d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.ConvBn1d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.ConvBn1d.html#torch.ao.nn.intrinsic.ConvBn1d)

**产品支持情况**：

<!-- npu="910b" id469 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id469 -->
<!-- npu="A3" id470 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id470 -->
<!-- npu="950" id471 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id471 -->

<!-- npu="950" id673 -->
**限制与说明**： 在<term>Ascend 950DT系列产品</term>上，底层卷积计算的W维度不能超过1000000。
<!-- end id673 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvBn2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.ConvBn2d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.ConvBn2d.html#torch.ao.nn.intrinsic.ConvBn2d)

**产品支持情况**：

<!-- npu="910b" id472 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id472 -->
<!-- npu="A3" id473 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id473 -->
<!-- npu="950" id474 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id474 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvBn3d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.ConvBn3d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.ConvBn3d.html#torch.ao.nn.intrinsic.ConvBn3d)

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

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvBnReLU1d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.ConvBnReLU1d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.ConvBnReLU1d.html#torch.ao.nn.intrinsic.ConvBnReLU1d)

**产品支持情况**：

<!-- npu="910b" id478 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id478 -->
<!-- npu="A3" id479 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id479 -->
<!-- npu="950" id480 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id480 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvBnReLU2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.ConvBnReLU2d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.ConvBnReLU2d.html#torch.ao.nn.intrinsic.ConvBnReLU2d)

**产品支持情况**：

<!-- npu="910b" id481 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id481 -->
<!-- npu="A3" id482 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id482 -->
<!-- npu="950" id483 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id483 -->

</div>

## torch.ao.nn.intrinsic.qat

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvBnReLU3d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.ConvBnReLU3d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.ConvBnReLU3d.html#torch.ao.nn.intrinsic.ConvBnReLU3d)

**产品支持情况**：

<!-- npu="910b" id383 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id383 -->
<!-- npu="A3" id384 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id384 -->
<!-- npu="950" id385 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id385 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.BNReLU2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.BNReLU2d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.BNReLU2d.html#torch.ao.nn.intrinsic.BNReLU2d)

**产品支持情况**：

<!-- npu="910b" id377 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id377 -->
<!-- npu="A3" id378 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id378 -->
<!-- npu="950" id379 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id379 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.BNReLU3d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.BNReLU3d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.BNReLU3d.html#torch.ao.nn.intrinsic.BNReLU3d)

**产品支持情况**：

<!-- npu="910b" id380 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id380 -->
<!-- npu="A3" id381 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id381 -->
<!-- npu="950" id382 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id382 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.LinearReLU

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.LinearReLU](https://pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.qat.LinearReLU.html)

**产品支持情况**：

<!-- npu="910b" id328 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id328 -->
<!-- npu="A3" id329 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id329 -->
<!-- npu="950" id330 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id330 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBn1d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.ConvBn1d](https://pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.qat.ConvBn1d.html)

**产品支持情况**：

<!-- npu="910b" id331 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id331 -->
<!-- npu="A3" id332 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id332 -->
<!-- npu="950" id333 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id333 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBnReLU1d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.ConvBnReLU1d](https://pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.qat.ConvBnReLU1d.html)

**产品支持情况**：

<!-- npu="910b" id334 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id334 -->
<!-- npu="A3" id335 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id335 -->
<!-- npu="950" id336 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id336 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBn2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.ConvBn2d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.qat.ConvBn2d.html#torch.ao.nn.intrinsic.qat.ConvBn2d)

**产品支持情况**：

<!-- npu="910b" id484 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id484 -->
<!-- npu="A3" id485 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id485 -->
<!-- npu="950" id486 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id486 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBnReLU2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.ConvBnReLU2d](https://pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.qat.ConvBnReLU2d.html)

**产品支持情况**：

<!-- npu="910b" id337 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id337 -->
<!-- npu="A3" id338 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id338 -->
<!-- npu="950" id339 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id339 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvReLU2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.ConvReLU2d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.qat.ConvReLU2d.html#torch.ao.nn.intrinsic.qat.ConvReLU2d)

**产品支持情况**：

<!-- npu="910b" id487 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id487 -->
<!-- npu="A3" id488 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id488 -->
<!-- npu="950" id489 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id489 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBn3d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.ConvBn3d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.qat.ConvBn3d.html#torch.ao.nn.intrinsic.qat.ConvBn3d)

**产品支持情况**：

<!-- npu="910b" id490 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id490 -->
<!-- npu="A3" id491 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id491 -->
<!-- npu="950" id492 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id492 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBnReLU3d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.ConvBnReLU3d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.qat.ConvBnReLU3d.html#torch.ao.nn.intrinsic.qat.ConvBnReLU3d)

**产品支持情况**：

<!-- npu="910b" id493 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id493 -->
<!-- npu="A3" id494 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id494 -->
<!-- npu="950" id495 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id495 -->

</div>

### torch.ao.nn.intrinsic.qat.update_bn_stats

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.update_bn_stats](https://pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.qat.update_bn_stats.html)

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

**原生文档**：[torch.ao.nn.intrinsic.qat.freeze_bn_stats](https://pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.qat.freeze_bn_stats.html)

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

## torch.ao.nn.intrinsic.quantized

### <code><i>class</i></code> torch.ao.nn.intrinsic.quantized.BNReLU2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.quantized.BNReLU2d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.quantized.BNReLU2d.html#torch.ao.nn.intrinsic.quantized.BNReLU2d)

**产品支持情况**：

<!-- npu="910b" id496 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id496 -->
<!-- npu="A3" id497 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id497 -->
<!-- npu="950" id498 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id498 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.quantized.BNReLU3d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.quantized.BNReLU3d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.quantized.BNReLU3d.html#torch.ao.nn.intrinsic.quantized.BNReLU3d)

**产品支持情况**：

<!-- npu="910b" id500 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id500 -->
<!-- npu="A3" id501 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id501 -->
<!-- npu="950" id502 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id502 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.quantized.ConvReLU1d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.quantized.ConvReLU1d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.quantized.ConvReLU1d.html#torch.ao.nn.intrinsic.quantized.ConvReLU1d)

**产品支持情况**：

<!-- npu="910b" id504 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id504 -->
<!-- npu="A3" id505 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id505 -->
<!-- npu="950" id506 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id506 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.quantized.ConvReLU2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.quantized.ConvReLU2d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.intrinsic.quantized.ConvReLU2d.html#torch.ao.nn.intrinsic.quantized.ConvReLU2d)

**产品支持情况**：

<!-- npu="910b" id508 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id508 -->
<!-- npu="A3" id509 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id509 -->
<!-- npu="950" id510 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id510 -->

</div>

## torch.ao.nn.qat

### <code><i>class</i></code> torch.ao.nn.qat.Conv2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.qat.Conv2d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.qat.Conv2d.html#torch.ao.nn.qat.Conv2d)

**产品支持情况**：

<!-- npu="910b" id512 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id512 -->
<!-- npu="A3" id513 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id513 -->
<!-- npu="950" id514 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id514 -->

</div>

### <code><i>class</i></code> torch.ao.nn.qat.Conv3d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.qat.Conv3d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.qat.Conv3d.html#torch.ao.nn.qat.Conv3d)

**产品支持情况**：

<!-- npu="910b" id516 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id516 -->
<!-- npu="A3" id517 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id517 -->
<!-- npu="950" id518 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id518 -->

</div>

### <code><i>class</i></code> torch.ao.nn.qat.Linear

<!-- npu="910b,A3" id672 -->
<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.qat.Linear](https://pytorch.org/docs/2.9/generated/torch.ao.nn.qat.Linear.html)

**产品支持情况**：

<!-- npu="910b" id346 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id346 -->
<!-- npu="A3" id347 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id347 -->
<!-- npu="950" id348 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id348 -->

<!-- end id672 -->

> <font size="3">from_float()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.qat.Linear.from\_float](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.qat.Linear.html#torch.ao.nn.qat.Linear.from_float)

**产品支持情况**：

<!-- npu="910b" id632 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id632 -->
<!-- npu="A3" id633 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id633 -->
<!-- npu="950" id634 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id634 -->

</div>

<!-- npu="910b,A3" id635 -->
</div>

<!-- end id635 -->

## torch.ao.nn.quantized

### <code><i>class</i></code> torch.ao.nn.quantized.ReLU6

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.ReLU6](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.ReLU6.html#torch.ao.nn.quantized.ReLU6)

**产品支持情况**：

<!-- npu="910b" id520 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id520 -->
<!-- npu="A3" id521 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id521 -->
<!-- npu="950" id522 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id522 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.ELU

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.ELU](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.ELU.html#torch.ao.nn.quantized.ELU)

**产品支持情况**：

<!-- npu="910b" id523 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id523 -->
<!-- npu="A3" id524 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id524 -->
<!-- npu="950" id525 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id525 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.LeakyReLU

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.LeakyReLU](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.LeakyReLU.html#torch.ao.nn.quantized.LeakyReLU)

**产品支持情况**：

<!-- npu="910b" id526 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id526 -->
<!-- npu="A3" id527 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id527 -->
<!-- npu="950" id528 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id528 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.Sigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.Sigmoid](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.Sigmoid.html#torch.ao.nn.quantized.Sigmoid)

**产品支持情况**：

<!-- npu="910b" id529 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id529 -->
<!-- npu="A3" id530 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id530 -->
<!-- npu="950" id531 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id531 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.BatchNorm2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.BatchNorm2d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.BatchNorm2d.html#torch.ao.nn.quantized.BatchNorm2d)

**产品支持情况**：

<!-- npu="910b" id532 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id532 -->
<!-- npu="A3" id533 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id533 -->
<!-- npu="950" id534 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id534 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.BatchNorm3d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.BatchNorm3d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.BatchNorm3d.html#torch.ao.nn.quantized.BatchNorm3d)

**产品支持情况**：

<!-- npu="910b" id535 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id535 -->
<!-- npu="A3" id536 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id536 -->
<!-- npu="950" id537 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id537 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.Conv1d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.Conv1d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.Conv1d.html#torch.ao.nn.quantized.Conv1d)

**产品支持情况**：

<!-- npu="910b" id538 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id538 -->
<!-- npu="A3" id539 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id539 -->
<!-- npu="950" id540 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id540 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.Conv2d

<div style="margin-left: 2em">

> <font size="3">from_float()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.Conv2d.from\_float](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.Conv2d.html#torch.ao.nn.quantized.Conv2d.from_float)

**产品支持情况**：

<!-- npu="910b" id639 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id639 -->
<!-- npu="A3" id640 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id640 -->
<!-- npu="950" id641 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id641 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.Conv3d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.Conv3d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.Conv3d.html#torch.ao.nn.quantized.Conv3d)

**产品支持情况**：

<!-- npu="910b" id541 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id541 -->
<!-- npu="A3" id542 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id542 -->
<!-- npu="950" id543 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id543 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.ConvTranspose1d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.ConvTranspose1d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.ConvTranspose1d.html#torch.ao.nn.quantized.ConvTranspose1d)

**产品支持情况**：

<!-- npu="910b" id544 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id544 -->
<!-- npu="A3" id545 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id545 -->
<!-- npu="950" id546 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id546 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.ConvTranspose2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.ConvTranspose2d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.ConvTranspose2d.html#torch.ao.nn.quantized.ConvTranspose2d)

**产品支持情况**：

<!-- npu="910b" id547 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id547 -->
<!-- npu="A3" id548 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id548 -->
<!-- npu="950" id549 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id549 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.Embedding

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.Embedding](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.Embedding.html#torch.ao.nn.quantized.Embedding)

**产品支持情况**：

<!-- npu="910b" id550 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id550 -->
<!-- npu="A3" id551 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id551 -->
<!-- npu="950" id552 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id552 -->

> <font size="3">from_float()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.Embedding.from\_float](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.Embedding.html#torch.ao.nn.quantized.Embedding.from_float)

**产品支持情况**：

<!-- npu="910b" id642 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id642 -->
<!-- npu="A3" id643 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id643 -->
<!-- npu="950" id644 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id644 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.QFunctional

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.QFunctional](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.QFunctional.html#torch.ao.nn.quantized.QFunctional)

**产品支持情况**：

<!-- npu="910b" id553 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id553 -->
<!-- npu="A3" id554 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id554 -->
<!-- npu="950" id555 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id555 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.LayerNorm

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.LayerNorm](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.LayerNorm.html#torch.ao.nn.quantized.LayerNorm)

**产品支持情况**：

<!-- npu="910b" id556 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id556 -->
<!-- npu="A3" id557 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id557 -->
<!-- npu="950" id558 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id558 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.InstanceNorm1d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.InstanceNorm1d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.InstanceNorm1d.html#torch.ao.nn.quantized.InstanceNorm1d)

**产品支持情况**：

<!-- npu="910b" id559 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id559 -->
<!-- npu="A3" id560 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id560 -->
<!-- npu="950" id561 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id561 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.InstanceNorm2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.InstanceNorm2d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.InstanceNorm2d.html#torch.ao.nn.quantized.InstanceNorm2d)

**产品支持情况**：

<!-- npu="910b" id562 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id562 -->
<!-- npu="A3" id563 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id563 -->
<!-- npu="950" id564 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id564 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.InstanceNorm3d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.InstanceNorm3d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.InstanceNorm3d.html#torch.ao.nn.quantized.InstanceNorm3d)

**产品支持情况**：

<!-- npu="910b" id565 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id565 -->
<!-- npu="A3" id566 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id566 -->
<!-- npu="950" id567 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id567 -->

</div>

## torch.ao.nn.quantized.functional

### <code><i>class</i></code> torch.ao.nn.quantized.functional.avg\_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.functional.avg\_pool2d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.functional.avg_pool2d.html#torch.ao.nn.quantized.functional.avg_pool2d)

**产品支持情况**：

<!-- npu="910b" id568 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id568 -->
<!-- npu="A3" id569 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id569 -->
<!-- npu="950" id570 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id570 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.avg\_pool3d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.functional.avg\_pool3d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.functional.avg_pool3d.html#torch.ao.nn.quantized.functional.avg_pool3d)

**产品支持情况**：

<!-- npu="950" id571 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id571 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.conv3d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.functional.conv3d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.functional.conv3d.html#torch.ao.nn.quantized.functional.conv3d)

**产品支持情况**：

<!-- npu="910b" id572 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id572 -->
<!-- npu="A3" id573 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id573 -->
<!-- npu="950" id574 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id574 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.interpolate

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.functional.interpolate](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.functional.interpolate.html#torch.ao.nn.quantized.functional.interpolate)

**产品支持情况**：

<!-- npu="910b" id575 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id575 -->
<!-- npu="A3" id576 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id576 -->
<!-- npu="950" id577 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id577 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.linear

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.functional.linear](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.functional.linear.html#torch.ao.nn.quantized.functional.linear)

**产品支持情况**：

<!-- npu="910b" id578 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id578 -->
<!-- npu="A3" id579 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id579 -->
<!-- npu="950" id580 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id580 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.max\_pool1d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.functional.max\_pool1d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.functional.max_pool1d.html#torch.ao.nn.quantized.functional.max_pool1d)

**产品支持情况**：

<!-- npu="910b" id581 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id581 -->
<!-- npu="A3" id582 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id582 -->
<!-- npu="950" id583 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id583 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.max\_pool2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.functional.max\_pool2d](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.functional.max_pool2d.html#torch.ao.nn.quantized.functional.max_pool2d)

**产品支持情况**：

<!-- npu="910b" id584 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id584 -->
<!-- npu="A3" id585 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id585 -->
<!-- npu="950" id586 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id586 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.hardtanh

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.functional.hardtanh](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.functional.hardtanh.html#torch.ao.nn.quantized.functional.hardtanh)

**产品支持情况**：

<!-- npu="910b" id587 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id587 -->
<!-- npu="A3" id588 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id588 -->
<!-- npu="950" id589 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id589 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.hardswish

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.functional.hardswish](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.functional.hardswish.html#torch.ao.nn.quantized.functional.hardswish)

**产品支持情况**：

<!-- npu="910b" id590 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id590 -->
<!-- npu="A3" id591 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id591 -->
<!-- npu="950" id592 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id592 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.threshold

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.functional.threshold](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.functional.threshold.html#torch.ao.nn.quantized.functional.threshold)

**产品支持情况**：

<!-- npu="910b" id593 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id593 -->
<!-- npu="A3" id594 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id594 -->
<!-- npu="950" id595 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id595 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.elu

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.functional.elu](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.functional.elu.html#torch.ao.nn.quantized.functional.elu)

**产品支持情况**：

<!-- npu="910b" id596 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id596 -->
<!-- npu="A3" id597 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id597 -->
<!-- npu="950" id598 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id598 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.hardsigmoid

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.functional.hardsigmoid](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.functional.hardsigmoid.html#torch.ao.nn.quantized.functional.hardsigmoid)

**产品支持情况**：

<!-- npu="910b" id599 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id599 -->
<!-- npu="A3" id600 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id600 -->
<!-- npu="950" id601 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id601 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.clamp

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.functional.clamp](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.functional.clamp.html#torch.ao.nn.quantized.functional.clamp)

**产品支持情况**：

<!-- npu="910b" id602 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id602 -->
<!-- npu="A3" id603 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id603 -->
<!-- npu="950" id604 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id604 -->

</div>

## torch.ao.nn.quantizable

### <code><i>class</i></code> torch.ao.nn.quantized.FloatFunctional

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.FloatFunctional](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.FloatFunctional.html#torch.ao.nn.quantized.FloatFunctional)

**产品支持情况**：

<!-- npu="910b" id392 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id392 -->
<!-- npu="A3" id393 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id393 -->
<!-- npu="950" id394 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id394 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.FXFloatFunctional

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.FXFloatFunctional](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.FXFloatFunctional.html#torch.ao.nn.quantized.FXFloatFunctional)

**产品支持情况**：

<!-- npu="910b" id398 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id398 -->
<!-- npu="A3" id399 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id399 -->
<!-- npu="950" id400 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id400 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.leaky_relu

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.functional.leaky_relu](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.functional.leaky_relu.html#torch.ao.nn.quantized.functional.leaky_relu)

**产品支持情况**：

<!-- npu="910b" id395 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id395 -->
<!-- npu="A3" id396 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id396 -->
<!-- npu="950" id397 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id397 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantizable.LSTM

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantizable.LSTM](https://pytorch.org/docs/2.9/generated/torch.ao.nn.quantizable.LSTM.html)

**产品支持情况**：

<!-- npu="910b" id349 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id349 -->
<!-- npu="A3" id350 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id350 -->
<!-- npu="950" id351 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id351 -->

</div>

## torch.ao.nn.quantized.dynamic

### <code><i>class</i></code> torch.ao.nn.quantizable.MultiheadAttention

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantizable.MultiheadAttention](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantizable.MultiheadAttention.html#torch.ao.nn.quantizable.MultiheadAttention)

**产品支持情况**：

<!-- npu="910b" id389 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id389 -->
<!-- npu="A3" id390 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id390 -->
<!-- npu="950" id391 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id391 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantizable.MultiheadAttention.forward](https://docs.pytorch.org/docs/2.9/generated/torch.ao.nn.quantizable.MultiheadAttention.html#torch.ao.nn.quantizable.MultiheadAttention.forward)

**产品支持情况**：

<!-- npu="910b" id636 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id636 -->
<!-- npu="A3" id637 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id637 -->
<!-- npu="950" id638 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id638 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.LSTM

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.LSTM](https://pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.dynamic.LSTM.html)

**产品支持情况**：

<!-- npu="910b" id355 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id355 -->
<!-- npu="A3" id356 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id356 -->
<!-- npu="950" id357 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id357 -->

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.GRU

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.GRU](https://pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.dynamic.GRU.html)

**产品支持情况**：

<!-- npu="910b" id358 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id358 -->
<!-- npu="A3" id359 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id359 -->
<!-- npu="950" id360 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id360 -->

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.RNNCell

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.RNNCell](https://pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.dynamic.RNNCell.html)

**产品支持情况**：

<!-- npu="910b" id361 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id361 -->
<!-- npu="A3" id362 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id362 -->
<!-- npu="950" id363 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id363 -->

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.LSTMCell

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.LSTMCell](https://pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.dynamic.LSTMCell.html)

**产品支持情况**：

<!-- npu="910b" id364 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id364 -->
<!-- npu="A3" id365 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id365 -->
<!-- npu="950" id366 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id366 -->

**限制与说明**： `input`仅支持fp32

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.GRUCell

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.GRUCell](https://pytorch.org/docs/2.9/generated/torch.ao.nn.quantized.dynamic.GRUCell.html)

**产品支持情况**：

<!-- npu="910b" id367 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id367 -->
<!-- npu="A3" id368 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id368 -->
<!-- npu="950" id369 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id369 -->

**限制与说明**： `input`仅支持fp32

</div>

## Numerical Debugging Tooling

### torch.ao.ns.fx.utils.compute_sqnr

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns.fx.utils.compute_sqnr](https://docs.pytorch.org/docs/2.9/quantization.html#torch.ao.ns.fx.utils.compute_sqnr)

**产品支持情况**：

<!-- npu="910b" id370 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id370 -->
<!-- npu="A3" id371 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id371 -->
<!-- npu="950" id372 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id372 -->

</div>
