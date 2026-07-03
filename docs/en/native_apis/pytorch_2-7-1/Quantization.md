# Quantization

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T01:04:12.151Z pushedAt=2026-06-15T02:04:36.486Z -->

> [!NOTE]
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.ao.quantization.prepare_qat](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.prepare_qat.html)|Yes|-|
|[torch.ao.quantization.convert](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.convert.html)|Yes|-|
|[torch.ao.quantization.QuantStub](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.QuantStub.html)|Yes|-|
|[torch.ao.quantization.DeQuantStub](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.DeQuantStub.html)|Yes|-|
|[torch.ao.quantization.QuantWrapper](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.QuantWrapper.html)|Yes|-|
|[torch.ao.quantization.qconfig_mapping.QConfigMapping](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html)|Yes|-|
|[torch.ao.quantization.qconfig_mapping.QConfigMapping.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.from_dict)|Yes|-|
|[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_global](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_global)|Yes|-|
|[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name)|Yes|-|
|[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_object_type_order](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_object_type_order)|Yes|-|
|[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_regex](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_regex)|Yes|-|
|[torch.ao.quantization.qconfig_mapping.QConfigMapping.set_object_type](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.set_object_type)|Yes|-|
|[torch.ao.quantization.qconfig_mapping.QConfigMapping.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping.to_dict)|Yes|-|
|[torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping.html)|Yes|-|
|[torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping.html)|Yes|-|
|[torch.ao.quantization.backend_config.BackendConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html)|Yes|-|
|[torch.ao.quantization.backend_config.BackendConfig.configs](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.configs)|Yes|-|
|[torch.ao.quantization.backend_config.BackendConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.from_dict)|Yes|-|
|[torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_config](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_config)|Yes|-|
|[torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_configs](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_configs)|Yes|-|
|[torch.ao.quantization.backend_config.BackendConfig.set_name](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.set_name)|Yes|-|
|[torch.ao.quantization.backend_config.BackendConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig.to_dict)|Yes|-|
|[torch.ao.quantization.backend_config.BackendPatternConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html)|Yes|-|
|[torch.ao.quantization.backend_config.BackendPatternConfig.add_dtype_config](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.add_dtype_config)|Yes|-|
|[torch.ao.quantization.backend_config.BackendPatternConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.from_dict)|Yes|-|
|[torch.ao.quantization.backend_config.BackendPatternConfig.set_dtype_configs](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_dtype_configs)|Yes|-|
|[torch.ao.quantization.backend_config.BackendPatternConfig.set_fused_module](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_fused_module)|Yes|-|
|[torch.ao.quantization.backend_config.BackendPatternConfig.set_fuser_method](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_fuser_method)|Yes|-|
|[torch.ao.quantization.backend_config.BackendPatternConfig.set_observation_type](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_observation_type)|Yes|-|
|[torch.ao.quantization.backend_config.BackendPatternConfig.set_pattern](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_pattern)|Yes|-|
|[torch.ao.quantization.backend_config.BackendPatternConfig.set_qat_module](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_qat_module)|Yes|-|
|[torch.ao.quantization.backend_config.BackendPatternConfig.set_reference_quantized_module](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_reference_quantized_module)|Yes|-|
|[torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module)|Yes|-|
|[torch.ao.quantization.backend_config.BackendPatternConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.BackendPatternConfig.html#torch.ao.quantization.backend_config.BackendPatternConfig.to_dict)|Yes|-|
|[torch.ao.quantization.backend_config.DTypeConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.DTypeConfig.html)|Yes|-|
|[torch.ao.quantization.backend_config.DTypeConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.DTypeConfig.html#torch.ao.quantization.backend_config.DTypeConfig.from_dict)|Yes|-|
|[torch.ao.quantization.backend_config.DTypeConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.DTypeConfig.html#torch.ao.quantization.backend_config.DTypeConfig.to_dict)|Yes|-|
|[torch.ao.quantization.backend_config.DTypeWithConstraints](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.DTypeWithConstraints.html)|Yes|-|
|[torch.ao.quantization.backend_config.ObservationType](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.ObservationType.html)|Yes|-|
|[torch.ao.quantization.backend_config.ObservationType.INPUT_OUTPUT_NOT_OBSERVED](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.INPUT_OUTPUT_NOT_OBSERVED)|Yes|-|
|[torch.ao.quantization.backend_config.ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)|Yes|-|
|[torch.ao.quantization.backend_config.ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.backend_config.ObservationType.html#torch.ao.quantization.backend_config.ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)|Yes|-|
|[torch.ao.quantization.fx.custom_config.FuseCustomConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html)|Yes|-|
|[torch.ao.quantization.fx.custom_config.FuseCustomConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.from_dict)|Yes|-|
|[torch.ao.quantization.fx.custom_config.FuseCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.set_preserved_attributes)|Yes|-|
|[torch.ao.quantization.fx.custom_config.FuseCustomConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.FuseCustomConfig.html#torch.ao.quantization.fx.custom_config.FuseCustomConfig.to_dict)|Yes|-|
|[torch.ao.quantization.fx.custom_config.PrepareCustomConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html)|Yes|-|
|[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict)|Yes|-|
|[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_float_to_observed_mapping](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_float_to_observed_mapping)|Yes|-|
|[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_input_quantized_indexes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_input_quantized_indexes)|Yes|-|
|[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_classes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_classes)|Yes|-|
|[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_names](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_names)|Yes|-|
|[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_output_quantized_indexes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_output_quantized_indexes)|Yes|-|
|[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_preserved_attributes)|Yes|-|
|[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_class](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_class)|Yes|-|
|[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_name](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_name)|Yes|-|
|[torch.ao.quantization.fx.custom_config.PrepareCustomConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.PrepareCustomConfig.html#torch.ao.quantization.fx.custom_config.PrepareCustomConfig.to_dict)|Yes|-|
|[torch.ao.quantization.fx.custom_config.ConvertCustomConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html)|Yes|-|
|[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict)|Yes|-|
|[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_observed_to_quantized_mapping](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_observed_to_quantized_mapping)|Yes|-|
|[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_preserved_attributes](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_preserved_attributes)|Yes|-|
|[torch.ao.quantization.fx.custom_config.ConvertCustomConfig.to_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.ConvertCustomConfig.html#torch.ao.quantization.fx.custom_config.ConvertCustomConfig.to_dict)|Yes|-|
|[torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry.html)|Yes|-|
|[torch.ao.quantization.observer.ObserverBase](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.ObserverBase.html)|Yes|-|
|[torch.ao.quantization.observer.ObserverBase.with_args](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.ObserverBase.html#torch.ao.quantization.observer.ObserverBase.with_args)|Yes|-|
|[torch.ao.quantization.observer.ObserverBase.with_callable_args](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.ObserverBase.html#torch.ao.quantization.observer.ObserverBase.with_callable_args)|Yes|-|
|[torch.ao.quantization.observer.MinMaxObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MinMaxObserver.html)|Yes|-|
|[torch.ao.quantization.observer.MinMaxObserver.calculate_qparams](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.calculate_qparams)|Yes|-|
|[torch.ao.quantization.observer.MinMaxObserver.forward](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.forward)|Yes|-|
|[torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals)|Yes|-|
|[torch.ao.quantization.observer.MovingAverageMinMaxObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MovingAverageMinMaxObserver.html)|Yes|-|
|[torch.ao.quantization.observer.PerChannelMinMaxObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.PerChannelMinMaxObserver.html)|Yes|-|
|[torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.PerChannelMinMaxObserver.html#torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals)|Yes|-|
|[torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver.html)|Yes|-|
|[torch.ao.quantization.observer.HistogramObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.HistogramObserver.html)|Yes|-|
|[torch.ao.quantization.observer.PlaceholderObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.PlaceholderObserver.html)|Yes|-|
|[torch.ao.quantization.observer.RecordingObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.RecordingObserver.html)|Yes|May fall back to CPU execution|
|[torch.ao.quantization.observer.NoopObserver](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.NoopObserver.html)|Yes|-|
|[torch.ao.quantization.observer.get_observer_state_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.get_observer_state_dict.html)|Yes|-|
|[torch.ao.quantization.observer.load_observer_state_dict](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.load_observer_state_dict.html)|Yes|-|
|[torch.ao.quantization.observer.default_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_observer.html)|Yes|-|
|[torch.ao.quantization.observer.default_placeholder_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_placeholder_observer.html)|Yes|-|
|[torch.ao.quantization.observer.default_debug_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_debug_observer.html)|Yes|-|
|[torch.ao.quantization.observer.default_weight_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_weight_observer.html)|Yes|-|
|[torch.ao.quantization.observer.default_histogram_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_histogram_observer.html)|Yes|-|
|[torch.ao.quantization.observer.default_per_channel_weight_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_per_channel_weight_observer.html)|Yes|-|
|[torch.ao.quantization.observer.default_dynamic_quant_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_dynamic_quant_observer.html)|Yes|-|
|[torch.ao.quantization.observer.default_float_qparams_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.observer.default_float_qparams_observer.html)|Yes|-|
|[torch.ao.quantization.fake_quantize.FakeQuantize](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)|Yes|May fall back to CPU execution|
|[torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize.html)|Yes|-|
|[torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize.html)|Yes|May fall back to CPU execution|
|[torch.ao.quantization.fake_quantize.disable_fake_quant](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.disable_fake_quant.html)|Yes|-|
|[torch.ao.quantization.fake_quantize.enable_fake_quant](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.enable_fake_quant.html)|Yes|-|
|[torch.ao.quantization.fake_quantize.disable_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.disable_observer.html)|Yes|-|
|[torch.ao.quantization.fake_quantize.enable_observer](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.fake_quantize.enable_observer.html)|Yes|-|
|[torch.ao.quantization.qconfig.QConfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.QConfig.html)|Yes|-|
|[torch.ao.quantization.qconfig.default_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_qconfig.html)|Yes|-|
|[torch.ao.quantization.qconfig.default_debug_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_debug_qconfig.html)|Yes|-|
|[torch.ao.quantization.qconfig.default_per_channel_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_per_channel_qconfig.html)|Yes|-|
|[torch.ao.quantization.qconfig.default_dynamic_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_dynamic_qconfig.html)|Yes|-|
|[torch.ao.quantization.qconfig.float16_dynamic_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.float16_dynamic_qconfig.html)|Yes|-|
|[torch.ao.quantization.qconfig.float16_static_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.float16_static_qconfig.html)|Yes|-|
|[torch.ao.quantization.qconfig.per_channel_dynamic_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.per_channel_dynamic_qconfig.html)|Yes|-|
|[torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig.html)|Yes|-|
|[torch.ao.quantization.qconfig.default_qat_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_qat_qconfig.html)|Yes|-|
|[torch.ao.quantization.qconfig.default_weight_only_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_weight_only_qconfig.html)|Yes|-|
|[torch.ao.quantization.qconfig.default_activation_only_qconfig](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_activation_only_qconfig.html)|Yes|-|
|[torch.ao.quantization.qconfig.default_qat_qconfig_v2](https://pytorch.org/docs/2.7/generated/torch.ao.quantization.qconfig.default_qat_qconfig_v2.html)|Yes|-|
|[torch.ao.nn.intrinsic.LinearReLU](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.LinearReLU.html)|No|-|
|[torch.ao.nn.intrinsic.qat.LinearReLU](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.LinearReLU.html)|No|-|
|[torch.ao.nn.intrinsic.qat.ConvBn1d](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.ConvBn1d.html)|No|-|
|[torch.ao.nn.intrinsic.qat.ConvBnReLU1d](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.ConvBnReLU1d.html)|No|-|
|[torch.ao.nn.intrinsic.qat.ConvBnReLU2d](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.ConvBnReLU2d.html)|No|-|
|[torch.ao.nn.intrinsic.qat.update_bn_stats](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.update_bn_stats.html)|Yes|-|
|[torch.ao.nn.intrinsic.qat.freeze_bn_stats](https://pytorch.org/docs/2.7/generated/torch.ao.nn.intrinsic.qat.freeze_bn_stats.html)|Yes|May fall back to CPU execution|
|[torch.ao.nn.qat.Linear](https://pytorch.org/docs/2.7/generated/torch.ao.nn.qat.Linear.html)|No|-|
|[torch.ao.nn.quantizable.LSTM](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantizable.LSTM.html)|No|-|
|[torch.ao.nn.quantized.dynamic.Linear](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.Linear.html)|No|-|
|[torch.ao.nn.quantized.dynamic.LSTM](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.LSTM.html)|Yes|Supports fp32|
|[torch.ao.nn.quantized.dynamic.GRU](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.GRU.html)|Yes|Supports fp32|
|[torch.ao.nn.quantized.dynamic.RNNCell](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.RNNCell.html)|Yes|Supports fp32|
|[torch.ao.nn.quantized.dynamic.LSTMCell](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.LSTMCell.html)|Yes|Supports fp32|
|[torch.ao.nn.quantized.dynamic.GRUCell](https://pytorch.org/docs/2.7/generated/torch.ao.nn.quantized.dynamic.GRUCell.html)|Yes|Supports fp32|
|[torch.ao.ns._numeric_suite.compare_weights](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.compare_weights)|Yes|-|
|[torch.ao.ns._numeric_suite.get_logger_dict](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.get_logger_dict)|Yes|-|
|[torch.ao.ns._numeric_suite.Logger](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.Logger)|Yes|-|
|[torch.ao.ns._numeric_suite.Logger.forward](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.Logger.forward)|Yes|-|
|[torch.ao.ns._numeric_suite.ShadowLogger](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.ShadowLogger)|Yes|-|
|[torch.ao.ns._numeric_suite.ShadowLogger.forward](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.ShadowLogger.forward)|Yes|-|
|[torch.ao.ns._numeric_suite.OutputLogger](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.OutputLogger)|Yes|-|
|[torch.ao.ns._numeric_suite.OutputLogger.forward](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.OutputLogger.forward)|Yes|-|
|[torch.ao.ns._numeric_suite.Shadow](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.Shadow)|Yes|-|
|[torch.ao.ns._numeric_suite.Shadow.forward](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.Shadow.forward)|Yes|-|
|[torch.ao.ns._numeric_suite.Shadow.add](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.Shadow.add)|Yes|Supports bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool|
|[torch.ao.ns._numeric_suite.Shadow.add_scalar](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.Shadow.add_scalar)|Yes|Supports bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool|
|[torch.ao.ns._numeric_suite.Shadow.mul](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.Shadow.mul)|Yes|Supports bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool|
|[torch.ao.ns._numeric_suite.Shadow.mul_scalar](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.Shadow.mul_scalar)|Yes|Supports bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool|
|[torch.ao.ns._numeric_suite.Shadow.cat](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.Shadow.cat)|Yes|Supports bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool|
|[torch.ao.ns._numeric_suite.Shadow.add_relu](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.Shadow.add_relu)|Yes|Supports bf16，fp16，fp32，uint8，int8，int32，int64|
|[torch.ao.ns._numeric_suite.prepare_model_with_stubs](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.prepare_model_with_stubs)|Yes|-|
|[torch.ao.ns._numeric_suite.compare_model_stub](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.compare_model_stub)|Yes|-|
|[torch.ao.ns._numeric_suite.get_matching_activations](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.get_matching_activations)|Yes|-|
|[torch.ao.ns._numeric_suite.prepare_model_outputs](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.prepare_model_outputs)|Yes|-|
|[torch.ao.ns._numeric_suite.compare_model_outputs](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite.compare_model_outputs)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.OutputLogger](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.OutputLogger)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.OutputLogger.forward](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.OutputLogger.forward)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.OutputComparisonLogger](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.OutputComparisonLogger)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.OutputComparisonLogger.forward](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.OutputComparisonLogger.forward)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.NSTracer](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.NSTracer)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.NSTracer.is_leaf_module](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.NSTracer.is_leaf_module)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.extract_weights](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.extract_weights)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.add_loggers](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.add_loggers)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.extract_logger_info](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.extract_logger_info)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.add_shadow_loggers](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.add_shadow_loggers)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.extract_shadow_logger_info](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.extract_shadow_logger_info)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.extend_logger_results_with_comparison](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.extend_logger_results_with_comparison)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.prepare_n_shadows_model](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.prepare_n_shadows_model)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.loggers_set_enabled](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.loggers_set_enabled)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.loggers_set_save_activations](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.loggers_set_save_activations)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.convert_n_shadows_model](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.convert_n_shadows_model)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.extract_results_n_shadows_model](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.extract_results_n_shadows_model)|Yes|-|
|[torch.ao.ns._numeric_suite_fx.print_comparisons_n_shadows_model](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns._numeric_suite_fx.print_comparisons_n_shadows_model)|Yes|-|
|[torch.ao.ns.fx.utils.compute_sqnr](https://pytorch.org/docs/2.7/ao.html#torch.ao.ns.fx.utils.compute_sqnr)|No|-|
