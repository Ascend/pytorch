# Quantization

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.7/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.7/quantization-support.html).
> - Note: In [Quantization Accuracy Debugging](https://docs.pytorch.org/docs/2.7/quantization-accuracy-debugging.html), you can find [Numerical Debugging Tooling](#numerical-debugging-tooling) for the usage of the APIs below.

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

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

### <code><i>class</i></code> torch.ao.quantization.quantize\_qat

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.quantize\_qat](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.quantize_qat.html#torch.ao.quantization.quantize_qat)

**Support Status**:

<!-- npu="910b" id540 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id540 -->
<!-- npu="A3" id541 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id541 -->
<!-- npu="950" id542 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id542 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.prepare

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.prepare](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.prepare.html#torch.ao.quantization.prepare)

**Support Status**:

<!-- npu="910b" id543 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id543 -->
<!-- npu="A3" id544 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id544 -->
<!-- npu="950" id545 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id545 -->

</div>

### torch.ao.quantization.prepare_qat

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.prepare_qat](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.prepare_qat.html)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id3 -->

</div>

<!-- npu="910b,A3" id490 -->
### torch.ao.quantization.convert

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.convert](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.convert.html)

**Support Status**:
<!-- end id490 -->

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->

<!-- npu="910b,A3" id491 -->
</div>
<!-- end id491 -->

### <code><i>class</i></code> torch.ao.quantization.fuse\_modules.fuse\_modules

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fuse\_modules.fuse\_modules](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.fuse_modules.fuse_modules.html#torch.ao.quantization.fuse_modules.fuse_modules)

**Support Status**:

<!-- npu="910b" id546 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id546 -->
<!-- npu="A3" id547 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id547 -->
<!-- npu="950" id548 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id548 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.QuantStub

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.QuantStub](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.QuantStub.html)

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id12 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.DeQuantStub

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.DeQuantStub](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.DeQuantStub.html)

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id15 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.QuantWrapper

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.QuantWrapper](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.QuantWrapper.html)

**Support Status**:

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id9 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.add\_quant\_dequant

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.add\_quant\_dequant](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.add_quant_dequant.html#torch.ao.quantization.add_quant_dequant)

**Support Status**:

<!-- npu="910b" id549 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id549 -->
<!-- npu="A3" id550 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id550 -->
<!-- npu="950" id551 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id551 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.swap\_module

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.swap\_module](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.swap_module.html#torch.ao.quantization.swap_module)

**Support Status**:

<!-- npu="910b" id552 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id552 -->
<!-- npu="A3" id553 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id553 -->
<!-- npu="950" id554 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id554 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.propagate\_qconfig\_

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.propagate\_qconfig\_](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.propagate_qconfig_.html#torch.ao.quantization.propagate_qconfig_)

**Support Status**:

<!-- npu="910b" id555 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id555 -->
<!-- npu="A3" id556 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id556 -->
<!-- npu="950" id557 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id557 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.default\_eval\_fn

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.default\_eval\_fn](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.default_eval_fn.html#torch.ao.quantization.default_eval_fn)

**Support Status**:

<!-- npu="910b" id558 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id558 -->
<!-- npu="A3" id559 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id559 -->
<!-- npu="950" id560 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id560 -->

</div>

## torch.ao.quantization.quantize_fx

### <code><i>class</i></code> torch.ao.quantization.quantize\_fx.prepare\_fx

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.quantize\_fx.prepare\_fx](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.quantize_fx.prepare_fx.html#torch.ao.quantization.quantize_fx.prepare_fx)

**Support Status**:

<!-- npu="910b" id561 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id561 -->
<!-- npu="A3" id562 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id562 -->
<!-- npu="950" id563 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id563 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.quantize\_fx.prepare\_qat\_fx

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.quantize\_fx.prepare\_qat\_fx](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.quantize_fx.prepare_qat_fx.html#torch.ao.quantization.quantize_fx.prepare_qat_fx)

**Support Status**:

<!-- npu="910b" id564 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id564 -->
<!-- npu="A3" id565 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id565 -->
<!-- npu="950" id566 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id566 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.quantize\_fx.convert\_fx

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.quantize\_fx.convert\_fx](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.quantize_fx.convert_fx.html#torch.ao.quantization.quantize_fx.convert_fx)

**Support Status**:

<!-- npu="910b" id567 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id567 -->
<!-- npu="A3" id568 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id568 -->
<!-- npu="950" id569 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id569 -->

</div>

<!-- npu="910b,A3" id573 -->
### <code><i>class</i></code> torch.ao.quantization.quantize\_fx.fuse\_fx

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.quantize\_fx.fuse\_fx](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.quantize_fx.fuse_fx.html#torch.ao.quantization.quantize_fx.fuse_fx)

**Support Status**:

<!-- npu="910b" id570 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id570 -->
<!-- npu="A3" id571 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id571 -->

</div>

<!-- end id573 -->

## torch.ao.quantization.qconfig_mapping

### <code><i>class</i></code> torch.ao.quantization.qconfig_mapping.QConfigMapping

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig_mapping.QConfigMapping](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html)

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id18 -->

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig_mapping.QConfigMapping.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.from_dict)

**Support Status**:

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id21 -->

</div>

> <font size="3">set_global()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig_mapping.QConfigMapping.set_global](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_global)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id24 -->

</div>

<!-- npu="910b,A3" id27 -->
> <font size="3">set_module_name()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name)

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->

</div>
<!-- end id27 -->

> <font size="3">set_module_name_object_type_order()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_object_type_order](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_object_type_order)

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id30 -->

</div>

> <font size="3">set_module_name_regex()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_regex](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_regex)

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id33 -->

</div>

> <font size="3">set_object_type()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig_mapping.QConfigMapping.set_object_type](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_object_type)

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id36 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig_mapping.QConfigMapping.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.to_dict)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id39 -->

</div>

</div>

### torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping.html)

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id42 -->

</div>

### torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping.html)

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id45 -->

</div>

## torch.ao.quantization.backend_config

### <code><i>class</i></code> torch.ao.quantization.backend_config.BackendConfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html)

**Support Status**:

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id48 -->

> <font size="3">configs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendConfig.configs](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.configs)

**Support Status**:

<!-- npu="910b" id49 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id51 -->

</div>

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.from_dict)

**Support Status**:

<!-- npu="910b" id52 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id54 -->

</div>

> <font size="3">set_backend_pattern_config()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_config](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_config)

**Support Status**:

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id57 -->

</div>

> <font size="3">set_backend_pattern_configs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_configs](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_configs)

**Support Status**:

<!-- npu="910b" id58 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id60 -->

</div>

> <font size="3">set_name()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendConfig.set_name](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_name)

**Support Status**:

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id63 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.to_dict)

**Support Status**:

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id66 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.backend_config.BackendPatternConfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendPatternConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html)

**Support Status**:

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id69 -->

> <font size="3">add_dtype_config()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendPatternConfig.add_dtype_config](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.add_dtype_config)

**Support Status**:

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id72 -->

</div>

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendPatternConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.from_dict)

**Support Status**:

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id75 -->

</div>

> <font size="3">set_dtype_configs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendPatternConfig.set_dtype_configs](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_dtype_configs)

**Support Status**:

<!-- npu="910b" id76 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id78 -->

</div>

> <font size="3">set_fused_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendPatternConfig.set_fused_module](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_fused_module)

**Support Status**:

<!-- npu="910b" id79 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id81 -->

</div>

> <font size="3">set_fuser_method()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendPatternConfig.set_fuser_method](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_fuser_method)

**Support Status**:

<!-- npu="910b" id82 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id84 -->

</div>

> <font size="3">set_observation_type()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendPatternConfig.set_observation_type](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_observation_type)

**Support Status**:

<!-- npu="910b" id85 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id87 -->

</div>

> <font size="3">set_pattern()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendPatternConfig.set_pattern](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_pattern)

**Support Status**:

<!-- npu="910b" id88 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id90 -->

</div>

> <font size="3">set_qat_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendPatternConfig.set_qat_module](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_qat_module)

**Support Status**:

<!-- npu="910b" id91 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id93 -->

</div>

> <font size="3">set_reference_quantized_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendPatternConfig.set_reference_quantized_module](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_reference_quantized_module)

**Support Status**:

<!-- npu="910b" id94 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id96 -->

</div>

> <font size="3">set_root_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module)

**Support Status**:

<!-- npu="910b" id97 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id99 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.BackendPatternConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.to_dict)

**Support Status**:

<!-- npu="910b" id100 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id102 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.backend_config.DTypeConfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.DTypeConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.DTypeConfig.html)

**Support Status**:

<!-- npu="910b" id103 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id105 -->

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.DTypeConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.DTypeConfig.html#torch.ao.quantization.backend_config.DTypeConfig.from_dict)

**Support Status**:

<!-- npu="910b" id106 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id108 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.DTypeConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.DTypeConfig.html#torch.ao.quantization.backend_config.DTypeConfig.to_dict)

**Support Status**:

<!-- npu="910b" id109 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id111 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.backend_config.DTypeWithConstraints

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.DTypeWithConstraints](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.DTypeWithConstraints.html)

**Support Status**:

<!-- npu="910b" id112 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id114 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.backend_config.ObservationType

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.ObservationType](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.ObservationType.html)

**Support Status**:

<!-- npu="910b" id115 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id117 -->

> <font size="3">INPUT_OUTPUT_NOT_OBSERVED()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.ObservationType.INPUT_OUTPUT_NOT_OBSERVED](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.INPUT_OUTPUT_NOT_OBSERVED)

**Support Status**:

<!-- npu="910b" id118 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id120 -->

</div>

> <font size="3">OUTPUT_SHARE_OBSERVER_WITH_INPUT()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)

**Support Status**:

<!-- npu="910b" id121 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id123 -->

</div>

> <font size="3">OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.backend_config.ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)

**Support Status**:

<!-- npu="910b" id124 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id126 -->

</div>

</div>

## torch.ao.quantization.fx.custom_config

### <code><i>class</i></code> torch.ao.quantization.fx.custom_config.FuseCustomConfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.FuseCustomConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html)

**Support Status**:

<!-- npu="910b" id127 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id129 -->

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.FuseCustomConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.from_dict)

**Support Status**:

<!-- npu="910b" id130 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id132 -->

</div>

> <font size="3">set_preserved_attributes()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.FuseCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.set_preserved_attributes)

**Support Status**:

<!-- npu="910b" id133 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id135 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.FuseCustomConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.to_dict)

**Support Status**:

<!-- npu="910b" id136 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id138 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.fx.custom_config.PrepareCustomConfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.PrepareCustomConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html)

**Support Status**:

<!-- npu="910b" id139 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id141 -->

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict)

**Support Status**:

<!-- npu="910b" id142 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id144 -->

</div>

> <font size="3">set_float_to_observed_mapping()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_float_to_observed_mapping](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_float_to_observed_mapping)

**Support Status**:

<!-- npu="910b" id145 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id147 -->

</div>

> <font size="3">set_input_quantized_indexes()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_input_quantized_indexes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_input_quantized_indexes)

**Support Status**:

<!-- npu="910b" id148 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id150 -->

</div>

> <font size="3">set_non_traceable_module_classes()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_classes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_classes)

**Support Status**:

<!-- npu="910b" id151 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id153 -->

</div>

> <font size="3">set_non_traceable_module_names()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_names](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_names)

**Support Status**:

<!-- npu="910b" id154 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id156 -->

</div>

> <font size="3">set_output_quantized_indexes()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_output_quantized_indexes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_output_quantized_indexes)

**Support Status**:

<!-- npu="910b" id157 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id159 -->

</div>

> <font size="3">set_preserved_attributes()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_preserved_attributes)

**Support Status**:

<!-- npu="910b" id160 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id162 -->

</div>

> <font size="3">set_standalone_module_class()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_class](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_class)

**Support Status**:

<!-- npu="910b" id163 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id165 -->

</div>

> <font size="3">set_standalone_module_name()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_name](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_name)

**Support Status**:

<!-- npu="910b" id166 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id168 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.PrepareCustomConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.to_dict)

**Support Status**:

<!-- npu="910b" id169 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id171 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.fx.custom_config.ConvertCustomConfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.ConvertCustomConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html)

**Support Status**:

<!-- npu="910b" id172 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id174 -->

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict)

**Support Status**:

<!-- npu="910b" id175 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id177 -->

</div>

> <font size="3">set_observed_to_quantized_mapping()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_observed_to_quantized_mapping](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_observed_to_quantized_mapping)

**Support Status**:

<!-- npu="910b" id178 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id180 -->

</div>

> <font size="3">set_preserved_attributes()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_preserved_attributes)

**Support Status**:

<!-- npu="910b" id181 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id183 -->

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.ConvertCustomConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.to_dict)

**Support Status**:

<!-- npu="910b" id184 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id186 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry.html)

**Support Status**:

<!-- npu="910b" id187 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id189 -->

</div>

## PT2 Export (pt2e) Numeric Debugger

### <code><i>class</i></code> torch.ao.quantization.prepare\_for\_propagation\_comparison

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.prepare\_for\_propagation\_comparison](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.prepare_for_propagation_comparison.html#torch.ao.quantization.prepare_for_propagation_comparison)

**Support Status**:

<!-- npu="910b" id574 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id574 -->
<!-- npu="A3" id575 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id575 -->
<!-- npu="950" id576 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id576 -->

</div>

<!-- npu="910b,A3" id580 -->
### <code><i>class</i></code> torch.ao.quantization.extract\_results\_from\_loggers

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.extract\_results\_from\_loggers](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.extract_results_from_loggers.html#torch.ao.quantization.extract_results_from_loggers)

**Support Status**:

<!-- npu="910b" id577 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id577 -->
<!-- npu="A3" id578 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id578 -->

</div>

<!-- end id580 -->

### <code><i>class</i></code> torch.ao.quantization.compare\_results

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.compare\_results](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.compare_results.html#torch.ao.quantization.compare_results)

**Support Status**:

<!-- npu="910b" id581 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id581 -->
<!-- npu="A3" id582 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id582 -->
<!-- npu="950" id583 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id583 -->

</div>

## torch.ao.quantization.observer

### <code><i>class</i></code> torch.ao.quantization.pt2e.export_utils.model_is_exported

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.pt2e.export_utils.model_is_exported](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.pt2e.export_utils.model_is_exported.html#torch.ao.quantization.pt2e.export_utils.model_is_exported)

**Support Status**:

<!-- npu="910b" id534 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id534 -->
<!-- npu="A3" id535 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id535 -->
<!-- npu="950" id536 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id536 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.generate_numeric_debug_handle

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.generate_numeric_debug_handle](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.generate_numeric_debug_handle.html#torch.ao.quantization.generate_numeric_debug_handle)

**Support Status**:

<!-- npu="910b" id522 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id522 -->
<!-- npu="A3" id523 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id523 -->
<!-- npu="950" id524 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id524 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.ObserverBase

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.ObserverBase](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.ObserverBase.html)

**Support Status**:

<!-- npu="910b" id190 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id192 -->

> <font size="3">with_args()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.ObserverBase.with_args](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.ObserverBase.html#torch.ao.quantization.observer.ObserverBase.with_args)

**Support Status**:

<!-- npu="910b" id193 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id195 -->

</div>

> <font size="3">with_callable_args()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.ObserverBase.with_callable_args](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.ObserverBase.html#torch.ao.quantization.observer.ObserverBase.with_callable_args)

**Support Status**:

<!-- npu="910b" id196 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id198 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.MinMaxObserver

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.MinMaxObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MinMaxObserver.html)

**Support Status**:

<!-- npu="910b" id199 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id201 -->

> <font size="3">calculate_qparams()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.MinMaxObserver.calculate_qparams](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.calculate_qparams)

**Support Status**:

<!-- npu="910b" id202 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id204 -->

</div>

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.MinMaxObserver.forward](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.forward)

**Support Status**:

<!-- npu="910b" id205 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id207 -->

</div>

> <font size="3">reset_min_max_vals()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals)

**Support Status**:

<!-- npu="910b" id208 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id210 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.MovingAverageMinMaxObserver

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.MovingAverageMinMaxObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MovingAverageMinMaxObserver.html)

**Support Status**:

<!-- npu="910b" id211 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id213 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.PerChannelMinMaxObserver

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.PerChannelMinMaxObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.PerChannelMinMaxObserver.html)

**Support Status**:

<!-- npu="910b" id214 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id215 -->
<!-- npu="950" id216 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id216 -->

> <font size="3">reset_min_max_vals()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.PerChannelMinMaxObserver.html#torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals)

**Support Status**:

<!-- npu="910b" id217 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id219 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver.html)

**Support Status**:

<!-- npu="910b" id220 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id222 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.HistogramObserver

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.HistogramObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.HistogramObserver.html)

**Support Status**:

<!-- npu="910b" id223 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id223 -->
<!-- npu="A3" id224 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id224 -->
<!-- npu="950" id225 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id225 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.PlaceholderObserver

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.PlaceholderObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.PlaceholderObserver.html)

**Support Status**:

<!-- npu="910b" id226 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id226 -->
<!-- npu="A3" id227 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id227 -->
<!-- npu="950" id228 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id228 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.RecordingObserver

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.RecordingObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.RecordingObserver.html)

**Support Status**:

<!-- npu="910b" id229 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id229 -->
<!-- npu="A3" id230 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id230 -->
<!-- npu="950" id231 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id231 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.NoopObserver

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.NoopObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.NoopObserver.html)

**Support Status**:

<!-- npu="910b" id232 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id232 -->
<!-- npu="A3" id233 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id233 -->
<!-- npu="950" id234 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id234 -->

</div>

### torch.ao.quantization.observer.get_observer_state_dict

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.get_observer_state_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.get_observer_state_dict.html)

**Support Status**:

<!-- npu="910b" id235 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id235 -->
<!-- npu="A3" id236 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id236 -->
<!-- npu="950" id237 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id237 -->

</div>

### torch.ao.quantization.observer.load_observer_state_dict

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.load_observer_state_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.load_observer_state_dict.html)

**Support Status**:

<!-- npu="910b" id238 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id238 -->
<!-- npu="A3" id239 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id239 -->
<!-- npu="950" id240 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id240 -->

</div>

### torch.ao.quantization.observer.default_observer

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.default_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_observer.html)

**Support Status**:

<!-- npu="910b" id241 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id241 -->
<!-- npu="A3" id242 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id242 -->
<!-- npu="950" id243 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id243 -->

</div>

### torch.ao.quantization.observer.default_placeholder_observer

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.default_placeholder_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_placeholder_observer.html)

**Support Status**:

<!-- npu="910b" id244 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id244 -->
<!-- npu="A3" id245 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id245 -->
<!-- npu="950" id246 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id246 -->

</div>

### torch.ao.quantization.observer.default_debug_observer

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.default_debug_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_debug_observer.html)

**Support Status**:

<!-- npu="910b" id247 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id247 -->
<!-- npu="A3" id248 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id248 -->
<!-- npu="950" id249 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id249 -->

</div>

### torch.ao.quantization.observer.default_weight_observer

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.default_weight_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_weight_observer.html)

**Support Status**:

<!-- npu="910b" id250 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id250 -->
<!-- npu="A3" id251 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id251 -->
<!-- npu="950" id252 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id252 -->

</div>

### torch.ao.quantization.observer.default_histogram_observer

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.default_histogram_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_histogram_observer.html)

**Support Status**:

<!-- npu="910b" id253 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id253 -->
<!-- npu="A3" id254 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id254 -->
<!-- npu="950" id255 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id255 -->

</div>

### torch.ao.quantization.observer.default_per_channel_weight_observer

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.default_per_channel_weight_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_per_channel_weight_observer.html)

**Support Status**:

<!-- npu="910b" id256 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id256 -->
<!-- npu="A3" id257 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id257 -->
<!-- npu="950" id258 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id258 -->

</div>

### torch.ao.quantization.observer.default_dynamic_quant_observer

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.default_dynamic_quant_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_dynamic_quant_observer.html)

**Support Status**:

<!-- npu="910b" id259 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id259 -->
<!-- npu="A3" id260 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id260 -->
<!-- npu="950" id261 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id261 -->

</div>

### torch.ao.quantization.observer.default_float_qparams_observer

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.default_float_qparams_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_float_qparams_observer.html)

**Support Status**:

<!-- npu="910b" id262 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id262 -->
<!-- npu="A3" id263 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id263 -->
<!-- npu="950" id264 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id264 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.Granularity

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.Granularity](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.Granularity.html#torch.ao.quantization.observer.Granularity)

**Support Status**:

<!-- npu="910b" id729 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id729 -->
<!-- npu="A3" id730 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id730 -->
<!-- npu="950" id731 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id731 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.PerAxis

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.PerAxis](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.PerAxis.html#torch.ao.quantization.observer.PerAxis)

**Support Status**:

<!-- npu="910b" id732 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id732 -->
<!-- npu="A3" id733 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id733 -->
<!-- npu="950" id734 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id734 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.PerBlock

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.PerBlock](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.PerBlock.html#torch.ao.quantization.observer.PerBlock)

**Support Status**:

<!-- npu="910b" id735 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id735 -->
<!-- npu="A3" id736 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id736 -->
<!-- npu="950" id737 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id737 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.PerRow

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.PerRow](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.PerRow.html#torch.ao.quantization.observer.PerRow)

**Support Status**:

<!-- npu="910b" id738 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id738 -->
<!-- npu="A3" id739 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id739 -->
<!-- npu="950" id740 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id740 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.PerTensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.PerTensor](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.PerTensor.html#torch.ao.quantization.observer.PerTensor)

**Support Status**:

<!-- npu="910b" id741 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id741 -->
<!-- npu="A3" id742 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id742 -->
<!-- npu="950" id743 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id743 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.PerToken

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.PerToken](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.PerToken.html#torch.ao.quantization.observer.PerToken)

**Support Status**:

<!-- npu="910b" id744 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id744 -->
<!-- npu="A3" id745 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id745 -->
<!-- npu="950" id746 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id746 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.TorchAODType

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.TorchAODType](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.TorchAODType.html#torch.ao.quantization.observer.TorchAODType)

**Support Status**:

<!-- npu="910b" id747 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id747 -->
<!-- npu="A3" id748 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id748 -->
<!-- npu="950" id749 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id749 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.get\_block\_size

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.get\_block\_size](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.get_block_size.html#torch.ao.quantization.observer.get_block_size)

**Support Status**:

<!-- npu="910b" id750 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id750 -->
<!-- npu="A3" id751 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id751 -->
<!-- npu="950" id752 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id752 -->

</div>

## torch.ao.quantization.fake_quantize

### <code><i>class</i></code> torch.ao.quantization.observer.AffineQuantizedObserverBase

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.AffineQuantizedObserverBase](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.AffineQuantizedObserverBase.html#torch.ao.quantization.observer.AffineQuantizedObserverBase)

**Support Status**:

<!-- npu="910b" id537 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id537 -->
<!-- npu="A3" id538 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id538 -->
<!-- npu="950" id539 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id539 -->

> <font size="3">calculate_qparams()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.AffineQuantizedObserverBase.calculate_qparams](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.AffineQuantizedObserverBase.html#torch.ao.quantization.observer.AffineQuantizedObserverBase.calculate_qparams)

**Support Status**:

<!-- npu="910b" id525 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id525 -->
<!-- npu="A3" id526 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id526 -->
<!-- npu="950" id527 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id527 -->

</div>

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.AffineQuantizedObserverBase.forward](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.AffineQuantizedObserverBase.html#torch.ao.quantization.observer.AffineQuantizedObserverBase.forward)

**Support Status**:

<!-- npu="910b" id528 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id528 -->
<!-- npu="A3" id529 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id529 -->
<!-- npu="950" id530 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id530 -->

</div>

> <font size="3">with_args()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.AffineQuantizedObserverBase.with\_args](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.AffineQuantizedObserverBase.html#torch.ao.quantization.observer.AffineQuantizedObserverBase.with_args)

**Support Status**:

<!-- npu="910b" id787 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id787 -->
<!-- npu="A3" id788 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id788 -->
<!-- npu="950" id789 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id789 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.quantization.observer.MappingType

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.observer.MappingType](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MappingType.html#torch.ao.quantization.observer.MappingType)

**Support Status**:

<!-- npu="910b" id531 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id531 -->
<!-- npu="A3" id532 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id532 -->
<!-- npu="950" id533 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id533 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.fake\_quantize.FakeQuantizeBase

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fake\_quantize.FakeQuantizeBase](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.FakeQuantizeBase.html#torch.ao.quantization.fake_quantize.FakeQuantizeBase)

**Support Status**:

<!-- npu="910b" id726 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id726 -->
<!-- npu="A3" id727 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id727 -->
<!-- npu="950" id728 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id728 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.fake_quantize.FakeQuantize

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fake_quantize.FakeQuantize](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)

**Support Status**:

<!-- npu="910b" id265 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id265 -->
<!-- npu="A3" id266 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id266 -->
<!-- npu="950" id267 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id267 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

### <code><i>class</i></code> torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize.html)

**Support Status**:

<!-- npu="910b" id268 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id268 -->
<!-- npu="A3" id269 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id269 -->
<!-- npu="950" id270 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id270 -->

</div>

### <code><i>class</i></code> torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize.html)

**Support Status**:

<!-- npu="910b" id271 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id271 -->
<!-- npu="A3" id272 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id272 -->
<!-- npu="950" id273 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id273 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

### torch.ao.quantization.fake_quantize.disable_fake_quant

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fake_quantize.disable_fake_quant](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.disable_fake_quant.html)

**Support Status**:

<!-- npu="910b" id274 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id274 -->
<!-- npu="A3" id275 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id275 -->
<!-- npu="950" id276 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id276 -->

</div>

### torch.ao.quantization.fake_quantize.enable_fake_quant

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fake_quantize.enable_fake_quant](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.enable_fake_quant.html)

**Support Status**:

<!-- npu="910b" id277 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id277 -->
<!-- npu="A3" id278 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id278 -->
<!-- npu="950" id279 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id279 -->

</div>

### torch.ao.quantization.fake_quantize.disable_observer

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fake_quantize.disable_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.disable_observer.html)

**Support Status**:

<!-- npu="910b" id280 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id280 -->
<!-- npu="A3" id281 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id281 -->
<!-- npu="950" id282 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id282 -->

</div>

### torch.ao.quantization.fake_quantize.enable_observer

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fake_quantize.enable_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.enable_observer.html)

**Support Status**:

<!-- npu="910b" id283 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id283 -->
<!-- npu="A3" id284 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id284 -->
<!-- npu="950" id285 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id285 -->

</div>

### torch.ao.quantization.fake\_quantize.default\_fake\_quant

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fake\_quantize.default\_fake\_quant](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.default_fake_quant.html#torch.ao.quantization.fake_quantize.default_fake_quant)

**Support Status**:

<!-- npu="910b" id766 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id766 -->
<!-- npu="A3" id767 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id767 -->
<!-- npu="950" id768 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id768 -->

</div>

### torch.ao.quantization.fake\_quantize.default\_weight\_fake\_quant

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fake\_quantize.default\_weight\_fake\_quant](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.default_weight_fake_quant.html#torch.ao.quantization.fake_quantize.default_weight_fake_quant)

**Support Status**:

<!-- npu="910b" id784 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id784 -->
<!-- npu="A3" id785 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id785 -->
<!-- npu="950" id786 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id786 -->

</div>

### torch.ao.quantization.fake\_quantize.default\_per\_channel\_weight\_fake\_quant

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fake\_quantize.default\_per\_channel\_weight\_fake\_quant](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.default_per_channel_weight_fake_quant.html#torch.ao.quantization.fake_quantize.default_per_channel_weight_fake_quant)

**Support Status**:

<!-- npu="910b" id781 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id781 -->
<!-- npu="A3" id782 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id782 -->
<!-- npu="950" id783 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id783 -->

</div>

### torch.ao.quantization.fake\_quantize.default\_histogram\_fake\_quant

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fake\_quantize.default\_histogram\_fake\_quant](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.default_histogram_fake_quant.html#torch.ao.quantization.fake_quantize.default_histogram_fake_quant)

**Support Status**:

<!-- npu="910b" id778 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id778 -->
<!-- npu="A3" id779 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id779 -->
<!-- npu="950" id780 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id780 -->

</div>

### torch.ao.quantization.fake\_quantize.default\_fused\_act\_fake\_quant

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fake\_quantize.default\_fused\_act\_fake\_quant](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.default_fused_act_fake_quant.html#torch.ao.quantization.fake_quantize.default_fused_act_fake_quant)

**Support Status**:

<!-- npu="910b" id769 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id769 -->
<!-- npu="A3" id770 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id770 -->
<!-- npu="950" id771 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id771 -->

</div>

### torch.ao.quantization.fake\_quantize.default\_fused\_wt\_fake\_quant

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fake\_quantize.default\_fused\_wt\_fake\_quant](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.default_fused_wt_fake_quant.html#torch.ao.quantization.fake_quantize.default_fused_wt_fake_quant)

**Support Status**:

<!-- npu="910b" id775 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id775 -->
<!-- npu="A3" id776 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id776 -->
<!-- npu="950" id777 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id777 -->

</div>

### torch.ao.quantization.fake\_quantize.default\_fused\_per\_channel\_wt\_fake\_quant

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.fake\_quantize.default\_fused\_per\_channel\_wt\_fake\_quant](https://docs.pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.default_fused_per_channel_wt_fake_quant.html#torch.ao.quantization.fake_quantize.default_fused_per_channel_wt_fake_quant)

**Support Status**:

<!-- npu="910b" id772 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id772 -->
<!-- npu="A3" id773 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id773 -->
<!-- npu="950" id774 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id774 -->

</div>

## torch.ao.quantization.qconfig

### <code><i>class</i></code> torch.ao.quantization.qconfig.QConfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig.QConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.QConfig.html)

**Support Status**:

<!-- npu="910b" id286 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id286 -->
<!-- npu="A3" id287 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id287 -->
<!-- npu="950" id288 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id288 -->

</div>

### torch.ao.quantization.qconfig.default_qconfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig.default_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_qconfig.html)

**Support Status**:

<!-- npu="910b" id289 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id289 -->
<!-- npu="A3" id290 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id290 -->
<!-- npu="950" id291 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id291 -->

</div>

### torch.ao.quantization.qconfig.default_debug_qconfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig.default_debug_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_debug_qconfig.html)

**Support Status**:

<!-- npu="910b" id292 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id292 -->
<!-- npu="A3" id293 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id293 -->
<!-- npu="950" id294 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id294 -->

</div>

### torch.ao.quantization.qconfig.default_per_channel_qconfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig.default_per_channel_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_per_channel_qconfig.html)

**Support Status**:

<!-- npu="910b" id295 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id295 -->
<!-- npu="A3" id296 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id296 -->
<!-- npu="950" id297 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id297 -->

</div>

### torch.ao.quantization.qconfig.default_dynamic_qconfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig.default_dynamic_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_dynamic_qconfig.html)

**Support Status**:

<!-- npu="910b" id298 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id298 -->
<!-- npu="A3" id299 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id299 -->
<!-- npu="950" id300 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id300 -->

</div>

### torch.ao.quantization.qconfig.float16_dynamic_qconfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig.float16_dynamic_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.float16_dynamic_qconfig.html)

**Support Status**:

<!-- npu="910b" id301 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id301 -->
<!-- npu="A3" id302 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id302 -->
<!-- npu="950" id303 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id303 -->

</div>

### torch.ao.quantization.qconfig.float16_static_qconfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig.float16_static_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.float16_static_qconfig.html)

**Support Status**:

<!-- npu="910b" id304 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id304 -->
<!-- npu="A3" id305 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id305 -->
<!-- npu="950" id306 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id306 -->

</div>

### torch.ao.quantization.qconfig.per_channel_dynamic_qconfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig.per_channel_dynamic_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.per_channel_dynamic_qconfig.html)

**Support Status**:

<!-- npu="910b" id307 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id307 -->
<!-- npu="A3" id308 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id308 -->
<!-- npu="950" id309 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id309 -->

</div>

### torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig.html)

**Support Status**:

<!-- npu="910b" id310 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id310 -->
<!-- npu="A3" id311 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id311 -->
<!-- npu="950" id312 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id312 -->

</div>

### torch.ao.quantization.qconfig.default_qat_qconfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig.default_qat_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_qat_qconfig.html)

**Support Status**:

<!-- npu="910b" id313 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id313 -->
<!-- npu="A3" id314 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id314 -->
<!-- npu="950" id315 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id315 -->

</div>

### torch.ao.quantization.qconfig.default_weight_only_qconfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig.default_weight_only_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_weight_only_qconfig.html)

**Support Status**:

<!-- npu="910b" id316 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id316 -->
<!-- npu="A3" id317 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id317 -->
<!-- npu="950" id318 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id318 -->

</div>

### torch.ao.quantization.qconfig.default_activation_only_qconfig

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig.default_activation_only_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_activation_only_qconfig.html)

**Support Status**:

<!-- npu="910b" id319 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id319 -->
<!-- npu="A3" id320 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id320 -->
<!-- npu="950" id321 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id321 -->

</div>

### torch.ao.quantization.qconfig.default_qat_qconfig_v2

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.quantization.qconfig.default_qat_qconfig_v2](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_qat_qconfig_v2.html)

**Support Status**:

<!-- npu="910b" id322 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id322 -->
<!-- npu="A3" id323 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id323 -->
<!-- npu="950" id324 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id324 -->

</div>

## torch.ao.nn.intrinsic

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvReLU1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.ConvReLU1d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.ConvReLU1d.html#torch.ao.nn.intrinsic.ConvReLU1d)

**Support Status**:

<!-- npu="910b" id507 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id507 -->
<!-- npu="A3" id508 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id508 -->
<!-- npu="950" id509 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id509 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvReLU2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.ConvReLU2d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.ConvReLU2d.html#torch.ao.nn.intrinsic.ConvReLU2d)

**Support Status**:

<!-- npu="910b" id584 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id584 -->
<!-- npu="A3" id585 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id585 -->
<!-- npu="950" id586 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id586 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvReLU3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.ConvReLU3d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.ConvReLU3d.html#torch.ao.nn.intrinsic.ConvReLU3d)

**Support Status**:

<!-- npu="910b" id587 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id587 -->
<!-- npu="A3" id588 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id588 -->
<!-- npu="950" id589 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id589 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.LinearReLU

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.LinearReLU](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.LinearReLU.html)

**Support Status**:

<!-- npu="910b" id325 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id325 -->
<!-- npu="A3" id326 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id326 -->
<!-- npu="950" id327 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id327 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvBn1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.ConvBn1d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.ConvBn1d.html#torch.ao.nn.intrinsic.ConvBn1d)

**Support Status**:

<!-- npu="910b" id590 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id590 -->
<!-- npu="A3" id591 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id591 -->
<!-- npu="950" id592 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id592 -->

<!-- npu="950" id794 -->
**Constraints & Notes**: On <term>Ascend 950DT products</term>, the W dimension of the underlying convolution calculation cannot exceed 1000000.
<!-- end id794 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvBn2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.ConvBn2d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.ConvBn2d.html#torch.ao.nn.intrinsic.ConvBn2d)

**Support Status**:

<!-- npu="910b" id593 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id593 -->
<!-- npu="A3" id594 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id594 -->
<!-- npu="950" id595 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id595 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvBn3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.ConvBn3d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.ConvBn3d.html#torch.ao.nn.intrinsic.ConvBn3d)

**Support Status**:

<!-- npu="910b" id596 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id596 -->
<!-- npu="A3" id597 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id597 -->
<!-- npu="950" id598 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id598 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvBnReLU1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.ConvBnReLU1d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.ConvBnReLU1d.html#torch.ao.nn.intrinsic.ConvBnReLU1d)

**Support Status**:

<!-- npu="910b" id599 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id599 -->
<!-- npu="A3" id600 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id600 -->
<!-- npu="950" id601 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id601 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvBnReLU2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.ConvBnReLU2d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.ConvBnReLU2d.html#torch.ao.nn.intrinsic.ConvBnReLU2d)

**Support Status**:

<!-- npu="910b" id602 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id602 -->
<!-- npu="A3" id603 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id603 -->
<!-- npu="950" id604 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id604 -->

</div>

## torch.ao.nn.intrinsic.qat

### <code><i>class</i></code> torch.ao.nn.intrinsic.ConvBnReLU3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.ConvBnReLU3d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.ConvBnReLU3d.html#torch.ao.nn.intrinsic.ConvBnReLU3d)

**Support Status**:

<!-- npu="910b" id504 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id504 -->
<!-- npu="A3" id505 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id505 -->
<!-- npu="950" id506 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id506 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.BNReLU2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.BNReLU2d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.BNReLU2d.html#torch.ao.nn.intrinsic.BNReLU2d)

**Support Status**:

<!-- npu="910b" id498 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id498 -->
<!-- npu="A3" id499 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id499 -->
<!-- npu="950" id500 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id500 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.BNReLU3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.BNReLU3d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.BNReLU3d.html#torch.ao.nn.intrinsic.BNReLU3d)

**Support Status**:

<!-- npu="910b" id501 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id501 -->
<!-- npu="A3" id502 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id502 -->
<!-- npu="950" id503 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id503 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.LinearReLU

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.qat.LinearReLU](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.LinearReLU.html)

**Support Status**:

<!-- npu="910b" id328 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id328 -->
<!-- npu="A3" id329 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id329 -->
<!-- npu="950" id330 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id330 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBn1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.qat.ConvBn1d](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.ConvBn1d.html)

**Support Status**:

<!-- npu="910b" id331 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id331 -->
<!-- npu="A3" id332 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id332 -->
<!-- npu="950" id333 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id333 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBnReLU1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.qat.ConvBnReLU1d](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.ConvBnReLU1d.html)

**Support Status**:

<!-- npu="910b" id334 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id334 -->
<!-- npu="A3" id335 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id335 -->
<!-- npu="950" id336 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id336 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBn2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.qat.ConvBn2d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.ConvBn2d.html#torch.ao.nn.intrinsic.qat.ConvBn2d)

**Support Status**:

<!-- npu="910b" id605 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id605 -->
<!-- npu="A3" id606 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id606 -->
<!-- npu="950" id607 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id607 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBnReLU2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.qat.ConvBnReLU2d](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.ConvBnReLU2d.html)

**Support Status**:

<!-- npu="910b" id337 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id337 -->
<!-- npu="A3" id338 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id338 -->
<!-- npu="950" id339 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id339 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvReLU2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.qat.ConvReLU2d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.ConvReLU2d.html#torch.ao.nn.intrinsic.qat.ConvReLU2d)

**Support Status**:

<!-- npu="910b" id608 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id608 -->
<!-- npu="A3" id609 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id609 -->
<!-- npu="950" id610 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id610 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBn3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.qat.ConvBn3d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.ConvBn3d.html#torch.ao.nn.intrinsic.qat.ConvBn3d)

**Support Status**:

<!-- npu="910b" id611 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id611 -->
<!-- npu="A3" id612 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id612 -->
<!-- npu="950" id613 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id613 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.qat.ConvBnReLU3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.qat.ConvBnReLU3d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.ConvBnReLU3d.html#torch.ao.nn.intrinsic.qat.ConvBnReLU3d)

**Support Status**:

<!-- npu="910b" id614 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id614 -->
<!-- npu="A3" id615 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id615 -->
<!-- npu="950" id616 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id616 -->

</div>

### torch.ao.nn.intrinsic.qat.update_bn_stats

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.qat.update_bn_stats](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.update_bn_stats.html)

**Support Status**:

<!-- npu="910b" id340 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id340 -->
<!-- npu="A3" id341 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id341 -->
<!-- npu="950" id342 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id342 -->

</div>

### torch.ao.nn.intrinsic.qat.freeze_bn_stats

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.qat.freeze_bn_stats](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.freeze_bn_stats.html)

**Support Status**:

<!-- npu="910b" id343 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id343 -->
<!-- npu="A3" id344 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id344 -->
<!-- npu="950" id345 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id345 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

## torch.ao.nn.intrinsic.quantized

### <code><i>class</i></code> torch.ao.nn.intrinsic.quantized.BNReLU2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.quantized.BNReLU2d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.quantized.BNReLU2d.html#torch.ao.nn.intrinsic.quantized.BNReLU2d)

**Support Status**:

<!-- npu="910b" id617 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id617 -->
<!-- npu="A3" id618 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id618 -->
<!-- npu="950" id619 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id619 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.quantized.BNReLU3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.quantized.BNReLU3d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.quantized.BNReLU3d.html#torch.ao.nn.intrinsic.quantized.BNReLU3d)

**Support Status**:

<!-- npu="910b" id621 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id621 -->
<!-- npu="A3" id622 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id622 -->
<!-- npu="950" id623 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id623 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.quantized.ConvReLU1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.quantized.ConvReLU1d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.quantized.ConvReLU1d.html#torch.ao.nn.intrinsic.quantized.ConvReLU1d)

**Support Status**:

<!-- npu="910b" id625 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id625 -->
<!-- npu="A3" id626 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id626 -->
<!-- npu="950" id627 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id627 -->

</div>

### <code><i>class</i></code> torch.ao.nn.intrinsic.quantized.ConvReLU2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.intrinsic.quantized.ConvReLU2d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.quantized.ConvReLU2d.html#torch.ao.nn.intrinsic.quantized.ConvReLU2d)

**Support Status**:

<!-- npu="910b" id629 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id629 -->
<!-- npu="A3" id630 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id630 -->
<!-- npu="950" id631 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id631 -->

</div>

## torch.ao.nn.qat

### <code><i>class</i></code> torch.ao.nn.qat.Conv2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.qat.Conv2d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.qat.Conv2d.html#torch.ao.nn.qat.Conv2d)

**Support Status**:

<!-- npu="910b" id633 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id633 -->
<!-- npu="A3" id634 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id634 -->
<!-- npu="950" id635 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id635 -->

</div>

### <code><i>class</i></code> torch.ao.nn.qat.Conv3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.qat.Conv3d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.qat.Conv3d.html#torch.ao.nn.qat.Conv3d)

**Support Status**:

<!-- npu="910b" id637 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id637 -->
<!-- npu="A3" id638 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id638 -->
<!-- npu="950" id639 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id639 -->

</div>

### <code><i>class</i></code> torch.ao.nn.qat.Linear

<!-- npu="910b,A3" id793 -->
<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.qat.Linear](https://pytorch.org/docs/2.7/generated/torch.ao.nn.qat.Linear.html)

**Support Status**:

<!-- npu="910b" id346 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id346 -->
<!-- npu="A3" id347 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id347 -->
<!-- npu="950" id348 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id348 -->

<!-- end id793 -->

> <font size="3">from_float()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.qat.Linear.from\_float](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.qat.Linear.html#torch.ao.nn.qat.Linear.from_float)

**Support Status**:

<!-- npu="910b" id753 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id753 -->
<!-- npu="A3" id754 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id754 -->
<!-- npu="950" id755 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id755 -->

</div>

<!-- npu="910b,A3" id756 -->
</div>

<!-- end id756 -->

## torch.ao.nn.quantized

### <code><i>class</i></code> torch.ao.nn.quantized.ReLU6

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.ReLU6](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.ReLU6.html#torch.ao.nn.quantized.ReLU6)

**Support Status**:

<!-- npu="910b" id641 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id641 -->
<!-- npu="A3" id642 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id642 -->
<!-- npu="950" id643 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id643 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.ELU

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.ELU](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.ELU.html#torch.ao.nn.quantized.ELU)

**Support Status**:

<!-- npu="910b" id644 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id644 -->
<!-- npu="A3" id645 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id645 -->
<!-- npu="950" id646 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id646 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.LeakyReLU

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.LeakyReLU](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.LeakyReLU.html#torch.ao.nn.quantized.LeakyReLU)

**Support Status**:

<!-- npu="910b" id647 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id647 -->
<!-- npu="A3" id648 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id648 -->
<!-- npu="950" id649 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id649 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.Sigmoid

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.Sigmoid](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.Sigmoid.html#torch.ao.nn.quantized.Sigmoid)

**Support Status**:

<!-- npu="910b" id650 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id650 -->
<!-- npu="A3" id651 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id651 -->
<!-- npu="950" id652 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id652 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.BatchNorm2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.BatchNorm2d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.BatchNorm2d.html#torch.ao.nn.quantized.BatchNorm2d)

**Support Status**:

<!-- npu="910b" id653 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id653 -->
<!-- npu="A3" id654 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id654 -->
<!-- npu="950" id655 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id655 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.BatchNorm3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.BatchNorm3d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.BatchNorm3d.html#torch.ao.nn.quantized.BatchNorm3d)

**Support Status**:

<!-- npu="910b" id656 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id656 -->
<!-- npu="A3" id657 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id657 -->
<!-- npu="950" id658 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id658 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.Conv1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.Conv1d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.Conv1d.html#torch.ao.nn.quantized.Conv1d)

**Support Status**:

<!-- npu="910b" id659 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id659 -->
<!-- npu="A3" id660 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id660 -->
<!-- npu="950" id661 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id661 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.Conv2d

<div style="margin-left: 2em">

> <font size="3">from_float()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.Conv2d.from\_float](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.Conv2d.html#torch.ao.nn.quantized.Conv2d.from_float)

**Support Status**:

<!-- npu="910b" id760 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id760 -->
<!-- npu="A3" id761 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id761 -->
<!-- npu="950" id762 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id762 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.Conv3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.Conv3d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.Conv3d.html#torch.ao.nn.quantized.Conv3d)

**Support Status**:

<!-- npu="910b" id662 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id662 -->
<!-- npu="A3" id663 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id663 -->
<!-- npu="950" id664 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id664 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.ConvTranspose1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.ConvTranspose1d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.ConvTranspose1d.html#torch.ao.nn.quantized.ConvTranspose1d)

**Support Status**:

<!-- npu="910b" id665 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id665 -->
<!-- npu="A3" id666 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id666 -->
<!-- npu="950" id667 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id667 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.ConvTranspose2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.ConvTranspose2d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.ConvTranspose2d.html#torch.ao.nn.quantized.ConvTranspose2d)

**Support Status**:

<!-- npu="910b" id668 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id668 -->
<!-- npu="A3" id669 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id669 -->
<!-- npu="950" id670 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id670 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.Embedding

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.Embedding](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.Embedding.html#torch.ao.nn.quantized.Embedding)

**Support Status**:

<!-- npu="910b" id671 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id671 -->
<!-- npu="A3" id672 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id672 -->
<!-- npu="950" id673 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id673 -->

> <font size="3">from_float()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.Embedding.from\_float](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.Embedding.html#torch.ao.nn.quantized.Embedding.from_float)

**Support Status**:

<!-- npu="910b" id763 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id763 -->
<!-- npu="A3" id764 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id764 -->
<!-- npu="950" id765 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id765 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.QFunctional

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.QFunctional](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.QFunctional.html#torch.ao.nn.quantized.QFunctional)

**Support Status**:

<!-- npu="910b" id674 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id674 -->
<!-- npu="A3" id675 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id675 -->
<!-- npu="950" id676 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id676 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.LayerNorm

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.LayerNorm](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.LayerNorm.html#torch.ao.nn.quantized.LayerNorm)

**Support Status**:

<!-- npu="910b" id677 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id677 -->
<!-- npu="A3" id678 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id678 -->
<!-- npu="950" id679 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id679 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.InstanceNorm1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.InstanceNorm1d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.InstanceNorm1d.html#torch.ao.nn.quantized.InstanceNorm1d)

**Support Status**:

<!-- npu="910b" id680 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id680 -->
<!-- npu="A3" id681 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id681 -->
<!-- npu="950" id682 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id682 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.InstanceNorm2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.InstanceNorm2d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.InstanceNorm2d.html#torch.ao.nn.quantized.InstanceNorm2d)

**Support Status**:

<!-- npu="910b" id683 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id683 -->
<!-- npu="A3" id684 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id684 -->
<!-- npu="950" id685 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id685 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.InstanceNorm3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.InstanceNorm3d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.InstanceNorm3d.html#torch.ao.nn.quantized.InstanceNorm3d)

**Support Status**:

<!-- npu="910b" id686 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id686 -->
<!-- npu="A3" id687 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id687 -->
<!-- npu="950" id688 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id688 -->

</div>

## torch.ao.nn.quantized.functional

### <code><i>class</i></code> torch.ao.nn.quantized.functional.avg\_pool2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.functional.avg\_pool2d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.functional.avg_pool2d.html#torch.ao.nn.quantized.functional.avg_pool2d)

**Support Status**:

<!-- npu="910b" id689 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id689 -->
<!-- npu="A3" id690 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id690 -->
<!-- npu="950" id691 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id691 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.avg\_pool3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.functional.avg\_pool3d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.functional.avg_pool3d.html#torch.ao.nn.quantized.functional.avg_pool3d)

**Support Status**:

<!-- npu="950" id692 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id692 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.conv3d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.functional.conv3d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.functional.conv3d.html#torch.ao.nn.quantized.functional.conv3d)

**Support Status**:

<!-- npu="910b" id693 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id693 -->
<!-- npu="A3" id694 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id694 -->
<!-- npu="950" id695 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id695 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.interpolate

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.functional.interpolate](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.functional.interpolate.html#torch.ao.nn.quantized.functional.interpolate)

**Support Status**:

<!-- npu="910b" id696 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id696 -->
<!-- npu="A3" id697 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id697 -->
<!-- npu="950" id698 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id698 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.linear

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.functional.linear](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.functional.linear.html#torch.ao.nn.quantized.functional.linear)

**Support Status**:

<!-- npu="910b" id699 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id699 -->
<!-- npu="A3" id700 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id700 -->
<!-- npu="950" id701 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id701 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.max\_pool1d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.functional.max\_pool1d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.functional.max_pool1d.html#torch.ao.nn.quantized.functional.max_pool1d)

**Support Status**:

<!-- npu="910b" id702 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id702 -->
<!-- npu="A3" id703 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id703 -->
<!-- npu="950" id704 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id704 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.max\_pool2d

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.functional.max\_pool2d](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.functional.max_pool2d.html#torch.ao.nn.quantized.functional.max_pool2d)

**Support Status**:

<!-- npu="910b" id705 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id705 -->
<!-- npu="A3" id706 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id706 -->
<!-- npu="950" id707 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id707 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.hardtanh

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.functional.hardtanh](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.functional.hardtanh.html#torch.ao.nn.quantized.functional.hardtanh)

**Support Status**:

<!-- npu="910b" id708 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id708 -->
<!-- npu="A3" id709 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id709 -->
<!-- npu="950" id710 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id710 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.hardswish

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.functional.hardswish](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.functional.hardswish.html#torch.ao.nn.quantized.functional.hardswish)

**Support Status**:

<!-- npu="910b" id711 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id711 -->
<!-- npu="A3" id712 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id712 -->
<!-- npu="950" id713 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id713 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.threshold

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.functional.threshold](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.functional.threshold.html#torch.ao.nn.quantized.functional.threshold)

**Support Status**:

<!-- npu="910b" id714 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id714 -->
<!-- npu="A3" id715 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id715 -->
<!-- npu="950" id716 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id716 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.elu

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.functional.elu](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.functional.elu.html#torch.ao.nn.quantized.functional.elu)

**Support Status**:

<!-- npu="910b" id717 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id717 -->
<!-- npu="A3" id718 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id718 -->
<!-- npu="950" id719 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id719 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.hardsigmoid

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.functional.hardsigmoid](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.functional.hardsigmoid.html#torch.ao.nn.quantized.functional.hardsigmoid)

**Support Status**:

<!-- npu="910b" id720 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id720 -->
<!-- npu="A3" id721 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id721 -->
<!-- npu="950" id722 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id722 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.clamp

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.functional.clamp](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.functional.clamp.html#torch.ao.nn.quantized.functional.clamp)

**Support Status**:

<!-- npu="910b" id723 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id723 -->
<!-- npu="A3" id724 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id724 -->
<!-- npu="950" id725 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id725 -->

</div>

## torch.ao.nn.quantizable

### <code><i>class</i></code> torch.ao.nn.quantized.FloatFunctional

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.FloatFunctional](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.FloatFunctional.html#torch.ao.nn.quantized.FloatFunctional)

**Support Status**:

<!-- npu="910b" id513 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id513 -->
<!-- npu="A3" id514 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id514 -->
<!-- npu="950" id515 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id515 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.FXFloatFunctional

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.FXFloatFunctional](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.FXFloatFunctional.html#torch.ao.nn.quantized.FXFloatFunctional)

**Support Status**:

<!-- npu="910b" id519 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id519 -->
<!-- npu="A3" id520 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id520 -->
<!-- npu="950" id521 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id521 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.functional.leaky_relu

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.functional.leaky_relu](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.functional.leaky_relu.html#torch.ao.nn.quantized.functional.leaky_relu)

**Support Status**:

<!-- npu="910b" id516 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id516 -->
<!-- npu="A3" id517 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id517 -->
<!-- npu="950" id518 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id518 -->

</div>

### <code><i>class</i></code> torch.ao.nn.quantizable.LSTM

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantizable.LSTM](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantizable.LSTM.html)

**Support Status**:

<!-- npu="910b" id349 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id349 -->
<!-- npu="A3" id350 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id350 -->
<!-- npu="950" id351 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id351 -->

</div>

## torch.ao.nn.quantized.dynamic

### <code><i>class</i></code> torch.ao.nn.quantizable.MultiheadAttention

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantizable.MultiheadAttention](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantizable.MultiheadAttention.html#torch.ao.nn.quantizable.MultiheadAttention)

**Support Status**:

<!-- npu="910b" id510 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id510 -->
<!-- npu="A3" id511 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id511 -->
<!-- npu="950" id512 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id512 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantizable.MultiheadAttention.forward](https://docs.pytorch.org/docs/2.7/generated/torch.ao.nn.quantizable.MultiheadAttention.html#torch.ao.nn.quantizable.MultiheadAttention.forward)

**Support Status**:

<!-- npu="910b" id757 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id757 -->
<!-- npu="A3" id758 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id758 -->
<!-- npu="950" id759 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id759 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.LSTM

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.dynamic.LSTM](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.LSTM.html)

**Support Status**:

<!-- npu="910b" id355 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id355 -->
<!-- npu="A3" id356 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id356 -->
<!-- npu="950" id357 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id357 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.GRU

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.dynamic.GRU](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.GRU.html)

**Support Status**:

<!-- npu="910b" id358 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id358 -->
<!-- npu="A3" id359 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id359 -->
<!-- npu="950" id360 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id360 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.RNNCell

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.dynamic.RNNCell](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.RNNCell.html)

**Support Status**:

<!-- npu="910b" id361 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id361 -->
<!-- npu="A3" id362 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id362 -->
<!-- npu="950" id363 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id363 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.LSTMCell

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.dynamic.LSTMCell](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.LSTMCell.html)

**Support Status**:

<!-- npu="910b" id364 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id364 -->
<!-- npu="A3" id365 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id365 -->
<!-- npu="950" id366 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id366 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.ao.nn.quantized.dynamic.GRUCell

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.nn.quantized.dynamic.GRUCell](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.GRUCell.html)

**Support Status**:

<!-- npu="910b" id367 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id367 -->
<!-- npu="A3" id368 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id368 -->
<!-- npu="950" id369 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id369 -->

**Constraints & Notes**: `input` only supports fp32

</div>

## Numerical Debugging Tooling

### torch.ao.ns._numeric_suite.compare_weights

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.compare_weights](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.compare_weights)

**Support Status**:

<!-- npu="910b" id370 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id370 -->
<!-- npu="A3" id371 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id371 -->
<!-- npu="950" id372 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id372 -->

</div>

### torch.ao.ns._numeric_suite.get_logger_dict

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.get_logger_dict](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.get_logger_dict)

**Support Status**:

<!-- npu="910b" id373 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id373 -->
<!-- npu="A3" id374 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id374 -->
<!-- npu="950" id375 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id375 -->

</div>

### <code><i>class</i></code> torch.ao.ns._numeric_suite.Logger

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.Logger](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Logger)

**Support Status**:

<!-- npu="910b" id376 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id376 -->
<!-- npu="A3" id377 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id377 -->
<!-- npu="950" id378 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id378 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.Logger.forward](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Logger.forward)

**Support Status**:

<!-- npu="910b" id379 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id379 -->
<!-- npu="A3" id380 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id380 -->
<!-- npu="950" id381 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id381 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.ns._numeric_suite.ShadowLogger

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.ShadowLogger](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.ShadowLogger)

**Support Status**:

<!-- npu="910b" id382 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id382 -->
<!-- npu="A3" id383 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id383 -->
<!-- npu="950" id384 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id384 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.ShadowLogger.forward](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.ShadowLogger.forward)

**Support Status**:

<!-- npu="910b" id385 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id385 -->
<!-- npu="A3" id386 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id386 -->
<!-- npu="950" id387 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id387 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.ns._numeric_suite.OutputLogger

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.OutputLogger](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.OutputLogger)

**Support Status**:

<!-- npu="910b" id388 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id388 -->
<!-- npu="A3" id389 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id389 -->
<!-- npu="950" id390 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id390 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.OutputLogger.forward](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.OutputLogger.forward)

**Support Status**:

<!-- npu="910b" id391 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id391 -->
<!-- npu="A3" id392 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id392 -->
<!-- npu="950" id393 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id393 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.ns._numeric_suite.Shadow

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.Shadow](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Shadow)

**Support Status**:

<!-- npu="910b" id394 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id394 -->
<!-- npu="A3" id395 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id395 -->
<!-- npu="950" id396 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id396 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.Shadow.forward](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Shadow.forward)

**Support Status**:

<!-- npu="910b" id397 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id397 -->
<!-- npu="A3" id398 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id398 -->
<!-- npu="950" id399 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id399 -->

</div>

> <font size="3">add()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.Shadow.add](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Shadow.add)

**Support Status**:

<!-- npu="910b" id400 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id400 -->
<!-- npu="A3" id401 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id401 -->
<!-- npu="950" id402 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id402 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">add_scalar()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.Shadow.add_scalar](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Shadow.add_scalar)

**Support Status**:

<!-- npu="910b" id403 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id403 -->
<!-- npu="A3" id404 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id404 -->
<!-- npu="950" id405 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id405 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

<!-- npu="910b,A3" id492 -->
> <font size="3">mul()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.Shadow.mul](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Shadow.mul)

**Support Status**:
<!-- end id492 -->

<!-- npu="910b" id406 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id406 -->
<!-- npu="A3" id407 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id407 -->

<!-- npu="910b,A3" id493 -->
**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>
<!-- end id493 -->

<!-- npu="910b,A3" id494 -->
> <font size="3">mul_scalar()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.Shadow.mul_scalar](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Shadow.mul_scalar)

**Support Status**:
<!-- end id494 -->

<!-- npu="910b" id409 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id409 -->
<!-- npu="A3" id410 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id410 -->

<!-- npu="910b,A3" id495 -->
**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>
<!-- end id495 -->

> <font size="3">cat()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.Shadow.cat](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Shadow.cat)

**Support Status**:

<!-- npu="910b" id412 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id412 -->
<!-- npu="A3" id413 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id413 -->
<!-- npu="950" id414 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id414 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">add_relu()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.Shadow.add_relu](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.Shadow.add_relu)

**Support Status**:

<!-- npu="910b" id415 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id415 -->
<!-- npu="A3" id416 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id416 -->
<!-- npu="950" id417 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id417 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int32, int64

</div>

</div>

### torch.ao.ns._numeric_suite.prepare_model_with_stubs

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.prepare_model_with_stubs](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.prepare_model_with_stubs)

**Support Status**:

<!-- npu="910b" id418 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id418 -->
<!-- npu="A3" id419 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id419 -->
<!-- npu="950" id420 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id420 -->

</div>

### torch.ao.ns._numeric_suite.compare_model_stub

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.compare_model_stub](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.compare_model_stub)

**Support Status**:

<!-- npu="910b" id421 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id421 -->
<!-- npu="A3" id422 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id422 -->
<!-- npu="950" id423 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id423 -->

</div>

### torch.ao.ns._numeric_suite.get_matching_activations

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.get_matching_activations](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.get_matching_activations)

**Support Status**:

<!-- npu="910b" id424 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id424 -->
<!-- npu="A3" id425 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id425 -->
<!-- npu="950" id426 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id426 -->

</div>

### torch.ao.ns._numeric_suite.prepare_model_outputs

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.prepare_model_outputs](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.prepare_model_outputs)

**Support Status**:

<!-- npu="910b" id427 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id427 -->
<!-- npu="A3" id428 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id428 -->
<!-- npu="950" id429 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id429 -->

</div>

### torch.ao.ns._numeric_suite.compare_model_outputs

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite.compare_model_outputs](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite.html#torch.ao.ns._numeric_suite.compare_model_outputs)

**Support Status**:

<!-- npu="910b" id430 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id430 -->
<!-- npu="A3" id431 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id431 -->
<!-- npu="950" id432 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id432 -->

</div>

### <code><i>class</i></code> torch.ao.ns._numeric_suite_fx.OutputLogger

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.OutputLogger](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.OutputLogger)

**Support Status**:

<!-- npu="910b" id433 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id433 -->
<!-- npu="A3" id434 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id434 -->
<!-- npu="950" id435 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id435 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.OutputLogger.forward](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.OutputLogger.forward)

**Support Status**:

<!-- npu="910b" id436 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id436 -->
<!-- npu="A3" id437 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id437 -->
<!-- npu="950" id438 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id438 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.ns._numeric_suite_fx.OutputComparisonLogger

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.OutputComparisonLogger](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.OutputComparisonLogger)

**Support Status**:

<!-- npu="910b" id439 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id439 -->
<!-- npu="A3" id440 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id440 -->
<!-- npu="950" id441 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id441 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.OutputComparisonLogger.forward](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.OutputComparisonLogger.forward)

**Support Status**:

<!-- npu="910b" id442 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id442 -->
<!-- npu="A3" id443 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id443 -->
<!-- npu="950" id444 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id444 -->

</div>

</div>

### <code><i>class</i></code> torch.ao.ns._numeric_suite_fx.NSTracer

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.NSTracer](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.NSTracer)

**Support Status**:

<!-- npu="910b" id445 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id445 -->
<!-- npu="A3" id446 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id446 -->
<!-- npu="950" id447 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id447 -->

> <font size="3">is_leaf_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.NSTracer.is_leaf_module](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.NSTracer.is_leaf_module)

**Support Status**:

<!-- npu="910b" id448 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id448 -->
<!-- npu="A3" id449 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id449 -->
<!-- npu="950" id450 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id450 -->

</div>

</div>

### torch.ao.ns._numeric_suite_fx.extract_weights

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.extract_weights](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.extract_weights)

**Support Status**:

<!-- npu="910b" id451 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id451 -->
<!-- npu="A3" id452 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id452 -->
<!-- npu="950" id453 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id453 -->

</div>

### torch.ao.ns._numeric_suite_fx.add_loggers

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.add_loggers](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.add_loggers)

**Support Status**:

<!-- npu="910b" id454 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id454 -->
<!-- npu="A3" id455 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id455 -->
<!-- npu="950" id456 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id456 -->

</div>

### torch.ao.ns._numeric_suite_fx.extract_logger_info

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.extract_logger_info](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.extract_logger_info)

**Support Status**:

<!-- npu="910b" id457 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id457 -->
<!-- npu="A3" id458 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id458 -->
<!-- npu="950" id459 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id459 -->

</div>

### torch.ao.ns._numeric_suite_fx.add_shadow_loggers

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.add_shadow_loggers](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.add_shadow_loggers)

**Support Status**:

<!-- npu="910b" id460 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id460 -->
<!-- npu="A3" id461 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id461 -->
<!-- npu="950" id462 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id462 -->

</div>

### torch.ao.ns._numeric_suite_fx.extract_shadow_logger_info

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.extract_shadow_logger_info](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.extract_shadow_logger_info)

**Support Status**:

<!-- npu="910b" id463 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id463 -->
<!-- npu="A3" id464 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id464 -->
<!-- npu="950" id465 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id465 -->

</div>

### torch.ao.ns._numeric_suite_fx.extend_logger_results_with_comparison

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.extend_logger_results_with_comparison](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.extend_logger_results_with_comparison)

**Support Status**:

<!-- npu="910b" id466 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id466 -->
<!-- npu="A3" id467 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id467 -->
<!-- npu="950" id468 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id468 -->

</div>

### torch.ao.ns._numeric_suite_fx.prepare_n_shadows_model

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.prepare_n_shadows_model](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.prepare_n_shadows_model)

**Support Status**:

<!-- npu="910b" id469 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id469 -->
<!-- npu="A3" id470 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id470 -->
<!-- npu="950" id471 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id471 -->

</div>

### torch.ao.ns._numeric_suite_fx.loggers_set_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.loggers_set_enabled](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.loggers_set_enabled)

**Support Status**:

<!-- npu="910b" id472 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id472 -->
<!-- npu="A3" id473 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id473 -->
<!-- npu="950" id474 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id474 -->

</div>

### torch.ao.ns._numeric_suite_fx.loggers_set_save_activations

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.loggers_set_save_activations](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.loggers_set_save_activations)

**Support Status**:

<!-- npu="910b" id475 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id475 -->
<!-- npu="A3" id476 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id476 -->
<!-- npu="950" id477 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id477 -->

</div>

### torch.ao.ns._numeric_suite_fx.convert_n_shadows_model

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.convert_n_shadows_model](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.convert_n_shadows_model)

**Support Status**:

<!-- npu="910b" id478 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id478 -->
<!-- npu="A3" id479 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id479 -->
<!-- npu="950" id480 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id480 -->

</div>

### torch.ao.ns._numeric_suite_fx.extract_results_n_shadows_model

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.extract_results_n_shadows_model](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.extract_results_n_shadows_model)

**Support Status**:

<!-- npu="910b" id481 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id481 -->
<!-- npu="A3" id482 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id482 -->
<!-- npu="950" id483 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id483 -->

</div>

### torch.ao.ns._numeric_suite_fx.print_comparisons_n_shadows_model

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns._numeric_suite_fx.print_comparisons_n_shadows_model](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns._numeric_suite_fx.print_comparisons_n_shadows_model)

**Support Status**:

<!-- npu="910b" id484 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id484 -->
<!-- npu="A3" id485 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id485 -->
<!-- npu="950" id486 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id486 -->

</div>

### torch.ao.ns.fx.utils.compute_sqnr

<div style="margin-left: 2em">

**Native Documentation**: [torch.ao.ns.fx.utils.compute_sqnr](https://pytorch.org/docs/2.7/torch.ao.ns._numeric_suite_fx.html#torch.ao.ns.fx.utils.compute_sqnr)

**Support Status**:

<!-- npu="910b" id487 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id487 -->
<!-- npu="A3" id488 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id488 -->
<!-- npu="950" id489 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id489 -->

</div>