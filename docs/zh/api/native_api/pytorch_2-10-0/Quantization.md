# Quantization

> [!NOTE]   
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)

## base API

### torch.ao.quantization.prepare_qat

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.prepare_qat](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.prepare_qat.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.ao.quantization.convert

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.convert](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.convert.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.ao.quantization.QuantStub

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.QuantStub](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.QuantStub.html)

**是否支持**：是

</div>

### _`class`_ torch.ao.quantization.DeQuantStub

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.DeQuantStub](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.DeQuantStub.html)

**是否支持**：是

</div>

### _`class`_ torch.ao.quantization.QuantWrapper

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.QuantWrapper](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.QuantWrapper.html)

**是否支持**：是

</div>

### _`class`_ torch.ao.quantization.qconfig_mapping.QConfigMapping

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html)

**是否支持**：是

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.from_dict](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.from_dict)

**是否支持**：是

</div>

> <font size="3">set_global()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_global](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_global)

**是否支持**：是

</div>

> <font size="3">set_module_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name)

**是否支持**：是

</div>

> <font size="3">set_module_name_object_type_order()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_object_type_order](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_object_type_order)

**是否支持**：是

</div>

> <font size="3">set_module_name_regex()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_regex](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_regex)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_object_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_object_type](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_object_type)

**是否支持**：是

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.QConfigMapping.to_dict](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.to_dict)

**是否支持**：是

</div>

</div>

### torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping.html)

**是否支持**：是

</div>

### torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping.html)

**是否支持**：是

</div>

### _`class`_ torch.ao.quantization.backend_config.BackendConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendConfig.html)

**是否支持**：是

> <font size="3">configs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.configs](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.configs)

**是否支持**：是

</div>

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.from_dict](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.from_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_backend_pattern_config()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_config](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_config)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_backend_pattern_configs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_configs](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_configs)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.set_name](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_name)

**是否支持**：是

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendConfig.to_dict](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.to_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.ao.quantization.backend_config.BackendPatternConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html)

**是否支持**：是

> <font size="3">add_dtype_config()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.add_dtype_config](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.add_dtype_config)

**是否支持**：是

</div>

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.from_dict](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.from_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_dtype_configs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_dtype_configs](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_dtype_configs)

**是否支持**：是

</div>

> <font size="3">set_fused_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_fused_module](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_fused_module)

**是否支持**：是

</div>

> <font size="3">set_fuser_method()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_fuser_method](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_fuser_method)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_observation_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_observation_type](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_observation_type)

**是否支持**：是

</div>

> <font size="3">set_pattern()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_pattern](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_pattern)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_qat_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_qat_module](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_qat_module)

**是否支持**：是

</div>

> <font size="3">set_reference_quantized_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_reference_quantized_module](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_reference_quantized_module)

**是否支持**：是

</div>

> <font size="3">set_root_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module)

**是否支持**：是

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.BackendPatternConfig.to_dict](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.to_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.ao.quantization.backend_config.DTypeConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.DTypeConfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.DTypeConfig.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.DTypeConfig.from_dict](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.DTypeConfig.html#torch.ao.quantization.backend_config.DTypeConfig.from_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.DTypeConfig.to_dict](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.DTypeConfig.html#torch.ao.quantization.backend_config.DTypeConfig.to_dict)

**是否支持**：是

</div>

</div>

### _`class`_ torch.ao.quantization.backend_config.DTypeWithConstraints

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.DTypeWithConstraints](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.DTypeWithConstraints.html)

**是否支持**：是

</div>

### _`class`_ torch.ao.quantization.backend_config.ObservationType

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.ObservationType](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.ObservationType.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">INPUT_OUTPUT_NOT_OBSERVED()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.ObservationType.INPUT_OUTPUT_NOT_OBSERVED](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.INPUT_OUTPUT_NOT_OBSERVED)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">OUTPUT_SHARE_OBSERVER_WITH_INPUT()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.backend_config.ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)

**是否支持**：是

</div>

</div>

### _`class`_ torch.ao.quantization.fx.custom_config.FuseCustomConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.FuseCustomConfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.FuseCustomConfig.from_dict](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.from_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_preserved_attributes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.FuseCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.set_preserved_attributes)

**是否支持**：是

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.FuseCustomConfig.to_dict](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.to_dict)

**是否支持**：是

</div>

</div>

### _`class`_ torch.ao.quantization.fx.custom_config.PrepareCustomConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict)

**是否支持**：是

</div>

> <font size="3">set_float_to_observed_mapping()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_float_to_observed_mapping](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_float_to_observed_mapping)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_input_quantized_indexes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_input_quantized_indexes](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_input_quantized_indexes)

**是否支持**：是

</div>

> <font size="3">set_non_traceable_module_classes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_classes](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_classes)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_non_traceable_module_names()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_names](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_names)

**是否支持**：是

</div>

> <font size="3">set_output_quantized_indexes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_output_quantized_indexes](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_output_quantized_indexes)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_preserved_attributes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_preserved_attributes)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_standalone_module_class()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_class](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_class)

**是否支持**：是

</div>

> <font size="3">set_standalone_module_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_name](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_name)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.to_dict](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.to_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.ao.quantization.fx.custom_config.ConvertCustomConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">from_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict)

**是否支持**：是

</div>

> <font size="3">set_observed_to_quantized_mapping()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_observed_to_quantized_mapping](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_observed_to_quantized_mapping)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_preserved_attributes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_preserved_attributes)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">to_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.to_dict](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.to_dict)

**是否支持**：是

</div>

</div>

### _`class`_ torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry.html)

**是否支持**：是

</div>

### _`class`_ torch.ao.quantization.observer.ObserverBase

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.ObserverBase](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.ObserverBase.html)

**是否支持**：是

> <font size="3">with_args()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.ObserverBase.with_args](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.ObserverBase.html#torch.ao.quantization.observer.ObserverBase.with_args)

**是否支持**：是

</div>

> <font size="3">with_callable_args()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.ObserverBase.with_callable_args](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.ObserverBase.html#torch.ao.quantization.observer.ObserverBase.with_callable_args)

**是否支持**：是

</div>

</div>

### _`class`_ torch.ao.quantization.observer.MinMaxObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MinMaxObserver](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.MinMaxObserver.html)

**是否支持**：是

> <font size="3">calculate_qparams()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MinMaxObserver.calculate_qparams](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.calculate_qparams)

**是否支持**：是

</div>

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MinMaxObserver.forward](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.forward)

**是否支持**：是

</div>

> <font size="3">reset_min_max_vals()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals)

**是否支持**：是

</div>

</div>

### _`class`_ torch.ao.quantization.observer.MovingAverageMinMaxObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MovingAverageMinMaxObserver](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.MovingAverageMinMaxObserver.html)

**是否支持**：是

</div>

### _`class`_ torch.ao.quantization.observer.PerChannelMinMaxObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.PerChannelMinMaxObserver](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.PerChannelMinMaxObserver.html)

**是否支持**：是

> <font size="3">reset_min_max_vals()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.PerChannelMinMaxObserver.html#torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.ao.quantization.observer.HistogramObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.HistogramObserver](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.HistogramObserver.html)

**是否支持**：是

</div>

### _`class`_ torch.ao.quantization.observer.PlaceholderObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.PlaceholderObserver](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.PlaceholderObserver.html)

**是否支持**：是

</div>

### _`class`_ torch.ao.quantization.observer.RecordingObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.RecordingObserver](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.RecordingObserver.html)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

### _`class`_ torch.ao.quantization.observer.NoopObserver

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.NoopObserver](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.NoopObserver.html)

**是否支持**：是

</div>

### torch.ao.quantization.observer.get_observer_state_dict

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.get_observer_state_dict](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.get_observer_state_dict.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.ao.quantization.observer.load_observer_state_dict

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.load_observer_state_dict](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.load_observer_state_dict.html)

**是否支持**：是

</div>

### torch.ao.quantization.observer.default_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_observer](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.default_observer.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.ao.quantization.observer.default_placeholder_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_placeholder_observer](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.default_placeholder_observer.html)

**是否支持**：是

</div>

### torch.ao.quantization.observer.default_debug_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_debug_observer](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.default_debug_observer.html)

**是否支持**：是

</div>

### torch.ao.quantization.observer.default_weight_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_weight_observer](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.default_weight_observer.html)

**是否支持**：是

</div>

### torch.ao.quantization.observer.default_histogram_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_histogram_observer](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.default_histogram_observer.html)

**是否支持**：是

</div>

### torch.ao.quantization.observer.default_per_channel_weight_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_per_channel_weight_observer](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.default_per_channel_weight_observer.html)

**是否支持**：是

</div>

### torch.ao.quantization.observer.default_dynamic_quant_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_dynamic_quant_observer](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.default_dynamic_quant_observer.html)

**是否支持**：是

</div>

### torch.ao.quantization.observer.default_float_qparams_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.observer.default_float_qparams_observer](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.observer.default_float_qparams_observer.html)

**是否支持**：是

</div>

### _`class`_ torch.ao.quantization.fake_quantize.FakeQuantize

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.FakeQuantize](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

### _`class`_ torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize.html)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

### torch.ao.quantization.fake_quantize.disable_fake_quant

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.disable_fake_quant](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fake_quantize.disable_fake_quant.html)

**是否支持**：是

</div>

### torch.ao.quantization.fake_quantize.enable_fake_quant

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.enable_fake_quant](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fake_quantize.enable_fake_quant.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.ao.quantization.fake_quantize.disable_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.disable_observer](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fake_quantize.disable_observer.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.ao.quantization.fake_quantize.enable_observer

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.fake_quantize.enable_observer](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.fake_quantize.enable_observer.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.ao.quantization.qconfig.QConfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.QConfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig.QConfig.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.ao.quantization.qconfig.default_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_qconfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig.default_qconfig.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.ao.quantization.qconfig.default_debug_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_debug_qconfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig.default_debug_qconfig.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.ao.quantization.qconfig.default_per_channel_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_per_channel_qconfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig.default_per_channel_qconfig.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.ao.quantization.qconfig.default_dynamic_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_dynamic_qconfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig.default_dynamic_qconfig.html)

**是否支持**：是

</div>

### torch.ao.quantization.qconfig.float16_dynamic_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.float16_dynamic_qconfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig.float16_dynamic_qconfig.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.ao.quantization.qconfig.float16_static_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.float16_static_qconfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig.float16_static_qconfig.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.ao.quantization.qconfig.per_channel_dynamic_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.per_channel_dynamic_qconfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig.per_channel_dynamic_qconfig.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig.html)

**是否支持**：是

</div>

### torch.ao.quantization.qconfig.default_qat_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_qat_qconfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig.default_qat_qconfig.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.ao.quantization.qconfig.default_weight_only_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_weight_only_qconfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig.default_weight_only_qconfig.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.ao.quantization.qconfig.default_activation_only_qconfig

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_activation_only_qconfig](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig.default_activation_only_qconfig.html)

**是否支持**：是

</div>

### torch.ao.quantization.qconfig.default_qat_qconfig_v2

<div style="margin-left: 2em">

**原生文档**：[torch.ao.quantization.qconfig.default_qat_qconfig_v2](https://pytorch.org/docs/2.10/generated/torch.ao.quantization.qconfig.default_qat_qconfig_v2.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.ao.nn.intrinsic.LinearReLU

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.LinearReLU](https://pytorch.org/docs/2.10/generated/torch.ao.nn.intrinsic.LinearReLU.html)

**是否支持**：否

</div>

### _`class`_ torch.ao.nn.intrinsic.qat.LinearReLU

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.LinearReLU](https://pytorch.org/docs/2.10/generated/torch.ao.nn.intrinsic.qat.LinearReLU.html)

**是否支持**：否

</div>

### _`class`_ torch.ao.nn.intrinsic.qat.ConvBn1d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.ConvBn1d](https://pytorch.org/docs/2.10/generated/torch.ao.nn.intrinsic.qat.ConvBn1d.html)

**是否支持**：否

</div>

### _`class`_ torch.ao.nn.intrinsic.qat.ConvBnReLU1d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.ConvBnReLU1d](https://pytorch.org/docs/2.10/generated/torch.ao.nn.intrinsic.qat.ConvBnReLU1d.html)

**是否支持**：否

</div>

### _`class`_ torch.ao.nn.intrinsic.qat.ConvBnReLU2d

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.ConvBnReLU2d](https://pytorch.org/docs/2.10/generated/torch.ao.nn.intrinsic.qat.ConvBnReLU2d.html)

**是否支持**：否

</div>

### torch.ao.nn.intrinsic.qat.update_bn_stats

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.update_bn_stats](https://pytorch.org/docs/2.10/generated/torch.ao.nn.intrinsic.qat.update_bn_stats.html)

**是否支持**：是

</div>

### torch.ao.nn.intrinsic.qat.freeze_bn_stats

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.intrinsic.qat.freeze_bn_stats](https://pytorch.org/docs/2.10/generated/torch.ao.nn.intrinsic.qat.freeze_bn_stats.html)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

### _`class`_ torch.ao.nn.qat.Linear

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.qat.Linear](https://pytorch.org/docs/2.10/generated/torch.ao.nn.qat.Linear.html)

**是否支持**：否

</div>

### _`class`_ torch.ao.nn.quantizable.LSTM

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantizable.LSTM](https://pytorch.org/docs/2.10/generated/torch.ao.nn.quantizable.LSTM.html)

**是否支持**：否

</div>

### _`class`_ torch.ao.nn.quantized.dynamic.Linear

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.Linear](https://pytorch.org/docs/2.10/generated/torch.ao.nn.quantized.dynamic.Linear.html)

**是否支持**：否

</div>

### _`class`_ torch.ao.nn.quantized.dynamic.LSTM

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.LSTM](https://pytorch.org/docs/2.10/generated/torch.ao.nn.quantized.dynamic.LSTM.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.ao.nn.quantized.dynamic.GRU

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.GRU](https://pytorch.org/docs/2.10/generated/torch.ao.nn.quantized.dynamic.GRU.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.ao.nn.quantized.dynamic.RNNCell

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.RNNCell](https://pytorch.org/docs/2.10/generated/torch.ao.nn.quantized.dynamic.RNNCell.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.ao.nn.quantized.dynamic.LSTMCell

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.LSTMCell](https://pytorch.org/docs/2.10/generated/torch.ao.nn.quantized.dynamic.LSTMCell.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.ao.nn.quantized.dynamic.GRUCell

<div style="margin-left: 2em">

**原生文档**：[torch.ao.nn.quantized.dynamic.GRUCell](https://pytorch.org/docs/2.10/generated/torch.ao.nn.quantized.dynamic.GRUCell.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.ao.ns.fx.utils.compute_sqnr

<div style="margin-left: 2em">

**原生文档**：[torch.ao.ns.fx.utils.compute_sqnr](https://pytorch.org/docs/2.10/quantization.html#torch.ao.ns.fx.utils.compute_sqnr)

**是否支持**：否

</div>
