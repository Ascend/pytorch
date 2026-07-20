# Quantization

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T01:04:12.151Z pushedAt=2026-06-15T02:04:36.486Z -->

> [!NOTE]
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.ao.quantization.prepare_qat|Yes|-|
|torch.ao.quantization.convert|Yes|-|
|torch.ao.quantization.QuantStub|Yes|-|
|torch.ao.quantization.DeQuantStub|Yes|-|
|torch.ao.quantization.QuantWrapper|Yes|-|
|torch.ao.quantization.qconfig_mapping.QConfigMapping|Yes|-|
|torch.ao.quantization.qconfig_mapping.QConfigMapping.from_dict|Yes|-|
|torch.ao.quantization.qconfig_mapping.QConfigMapping.set_global|Yes|-|
|torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name|Yes|-|
|torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_object_type_order|Yes|-|
|torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_regex|Yes|-|
|torch.ao.quantization.qconfig_mapping.QConfigMapping.set_object_type|Yes|-|
|torch.ao.quantization.qconfig_mapping.QConfigMapping.to_dict|Yes|-|
|torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping|Yes|-|
|torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping|Yes|-|
|torch.ao.quantization.backend_config.BackendConfig|Yes|-|
|torch.ao.quantization.backend_config.BackendConfig.configs|Yes|-|
|torch.ao.quantization.backend_config.BackendConfig.from_dict|Yes|-|
|torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_config|Yes|-|
|torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_configs|Yes|-|
|torch.ao.quantization.backend_config.BackendConfig.set_name|Yes|-|
|torch.ao.quantization.backend_config.BackendConfig.to_dict|Yes|-|
|torch.ao.quantization.backend_config.BackendPatternConfig|Yes|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.add_dtype_config|Yes|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.from_dict|Yes|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.set_dtype_configs|Yes|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.set_fused_module|Yes|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.set_fuser_method|Yes|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.set_observation_type|Yes|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.set_pattern|Yes|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.set_qat_module|Yes|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.set_reference_quantized_module|Yes|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module|Yes|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.to_dict|Yes|-|
|torch.ao.quantization.backend_config.DTypeConfig|Yes|-|
|torch.ao.quantization.backend_config.DTypeConfig.from_dict|Yes|-|
|torch.ao.quantization.backend_config.DTypeConfig.to_dict|Yes|-|
|torch.ao.quantization.backend_config.DTypeWithConstraints|Yes|-|
|torch.ao.quantization.backend_config.ObservationType|Yes|-|
|torch.ao.quantization.backend_config.ObservationType.INPUT_OUTPUT_NOT_OBSERVED|Yes|-|
|torch.ao.quantization.backend_config.ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT|Yes|-|
|torch.ao.quantization.backend_config.ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT|Yes|-|
|torch.ao.quantization.fx.custom_config.FuseCustomConfig|Yes|-|
|torch.ao.quantization.fx.custom_config.FuseCustomConfig.from_dict|Yes|-|
|torch.ao.quantization.fx.custom_config.FuseCustomConfig.set_preserved_attributes|Yes|-|
|torch.ao.quantization.fx.custom_config.FuseCustomConfig.to_dict|Yes|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig|Yes|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict|Yes|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_float_to_observed_mapping|Yes|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_input_quantized_indexes|Yes|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_classes|Yes|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_names|Yes|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_output_quantized_indexes|Yes|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_preserved_attributes|Yes|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_class|Yes|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_name|Yes|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.to_dict|Yes|-|
|torch.ao.quantization.fx.custom_config.ConvertCustomConfig|Yes|-|
|torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict|Yes|-|
|torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_observed_to_quantized_mapping|Yes|-|
|torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_preserved_attributes|Yes|-|
|torch.ao.quantization.fx.custom_config.ConvertCustomConfig.to_dict|Yes|-|
|torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry|Yes|-|
|torch.ao.quantization.observer.ObserverBase|Yes|-|
|torch.ao.quantization.observer.ObserverBase.with_args|Yes|-|
|torch.ao.quantization.observer.ObserverBase.with_callable_args|Yes|-|
|torch.ao.quantization.observer.MinMaxObserver|Yes|-|
|torch.ao.quantization.observer.MinMaxObserver.calculate_qparams|Yes|-|
|torch.ao.quantization.observer.MinMaxObserver.forward|Yes|-|
|torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals|Yes|-|
|torch.ao.quantization.observer.MovingAverageMinMaxObserver|Yes|-|
|torch.ao.quantization.observer.PerChannelMinMaxObserver|Yes|-|
|torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals|Yes|-|
|torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver|Yes|-|
|torch.ao.quantization.observer.HistogramObserver|Yes|-|
|torch.ao.quantization.observer.PlaceholderObserver|Yes|-|
|torch.ao.quantization.observer.RecordingObserver|Yes|May fall back to CPU execution|
|torch.ao.quantization.observer.NoopObserver|Yes|-|
|torch.ao.quantization.observer.get_observer_state_dict|Yes|-|
|torch.ao.quantization.observer.load_observer_state_dict|Yes|-|
|torch.ao.quantization.observer.default_observer|Yes|-|
|torch.ao.quantization.observer.default_placeholder_observer|Yes|-|
|torch.ao.quantization.observer.default_debug_observer|Yes|-|
|torch.ao.quantization.observer.default_weight_observer|Yes|-|
|torch.ao.quantization.observer.default_histogram_observer|Yes|-|
|torch.ao.quantization.observer.default_per_channel_weight_observer|Yes|-|
|torch.ao.quantization.observer.default_dynamic_quant_observer|Yes|-|
|torch.ao.quantization.observer.default_float_qparams_observer|Yes|-|
|torch.ao.quantization.fake_quantize.FakeQuantize|Yes|May fall back to CPU execution|
|torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize|Yes|-|
|torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize|Yes|May fall back to CPU execution|
|torch.ao.quantization.fake_quantize.disable_fake_quant|Yes|-|
|torch.ao.quantization.fake_quantize.enable_fake_quant|Yes|-|
|torch.ao.quantization.fake_quantize.disable_observer|Yes|-|
|torch.ao.quantization.fake_quantize.enable_observer|Yes|-|
|torch.ao.quantization.qconfig.QConfig|Yes|-|
|torch.ao.quantization.qconfig.default_qconfig|Yes|-|
|torch.ao.quantization.qconfig.default_debug_qconfig|Yes|-|
|torch.ao.quantization.qconfig.default_per_channel_qconfig|Yes|-|
|torch.ao.quantization.qconfig.default_dynamic_qconfig|Yes|-|
|torch.ao.quantization.qconfig.float16_dynamic_qconfig|Yes|-|
|torch.ao.quantization.qconfig.float16_static_qconfig|Yes|-|
|torch.ao.quantization.qconfig.per_channel_dynamic_qconfig|Yes|-|
|torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig|Yes|-|
|torch.ao.quantization.qconfig.default_qat_qconfig|Yes|-|
|torch.ao.quantization.qconfig.default_weight_only_qconfig|Yes|-|
|torch.ao.quantization.qconfig.default_activation_only_qconfig|Yes|-|
|torch.ao.quantization.qconfig.default_qat_qconfig_v2|Yes|-|
|torch.ao.nn.intrinsic.LinearReLU|No|-|
|torch.ao.nn.intrinsic.qat.LinearReLU|No|-|
|torch.ao.nn.intrinsic.qat.ConvBn1d|No|-|
|torch.ao.nn.intrinsic.qat.ConvBnReLU1d|No|-|
|torch.ao.nn.intrinsic.qat.ConvBnReLU2d|No|-|
|torch.ao.nn.intrinsic.qat.update_bn_stats|Yes|-|
|torch.ao.nn.intrinsic.qat.freeze_bn_stats|Yes|May fall back to CPU execution|
|torch.ao.nn.qat.Linear|No|-|
|torch.ao.nn.quantizable.LSTM|No|-|
|torch.ao.nn.quantized.dynamic.Linear|No|-|
|torch.ao.nn.quantized.dynamic.LSTM|Yes|Supports fp32|
|torch.ao.nn.quantized.dynamic.GRU|Yes|Supports fp32|
|torch.ao.nn.quantized.dynamic.RNNCell|Yes|Supports fp32|
|torch.ao.nn.quantized.dynamic.LSTMCell|Yes|Supports fp32|
|torch.ao.nn.quantized.dynamic.GRUCell|Yes|Supports fp32|
|torch.ao.ns._numeric_suite.compare_weights|Yes|-|
|torch.ao.ns._numeric_suite.get_logger_dict|Yes|-|
|torch.ao.ns._numeric_suite.Logger|Yes|-|
|torch.ao.ns._numeric_suite.Logger.forward|Yes|-|
|torch.ao.ns._numeric_suite.ShadowLogger|Yes|-|
|torch.ao.ns._numeric_suite.ShadowLogger.forward|Yes|-|
|torch.ao.ns._numeric_suite.OutputLogger|Yes|-|
|torch.ao.ns._numeric_suite.OutputLogger.forward|Yes|-|
|torch.ao.ns._numeric_suite.Shadow|Yes|-|
|torch.ao.ns._numeric_suite.Shadow.forward|Yes|-|
|torch.ao.ns._numeric_suite.Shadow.add|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.ao.ns._numeric_suite.Shadow.add_scalar|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.ao.ns._numeric_suite.Shadow.mul|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.ao.ns._numeric_suite.Shadow.mul_scalar|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.ao.ns._numeric_suite.Shadow.cat|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.ao.ns._numeric_suite.Shadow.add_relu|Yes|Supports bf16, fp16, fp32, uint8, int8, int32, int64|
|torch.ao.ns._numeric_suite.prepare_model_with_stubs|Yes|-|
|torch.ao.ns._numeric_suite.compare_model_stub|Yes|-|
|torch.ao.ns._numeric_suite.get_matching_activations|Yes|-|
|torch.ao.ns._numeric_suite.prepare_model_outputs|Yes|-|
|torch.ao.ns._numeric_suite.compare_model_outputs|Yes|-|
|torch.ao.ns._numeric_suite_fx.OutputLogger|Yes|-|
|torch.ao.ns._numeric_suite_fx.OutputLogger.forward|Yes|-|
|torch.ao.ns._numeric_suite_fx.OutputComparisonLogger|Yes|-|
|torch.ao.ns._numeric_suite_fx.OutputComparisonLogger.forward|Yes|-|
|torch.ao.ns._numeric_suite_fx.NSTracer|Yes|-|
|torch.ao.ns._numeric_suite_fx.NSTracer.is_leaf_module|Yes|-|
|torch.ao.ns._numeric_suite_fx.extract_weights|Yes|-|
|torch.ao.ns._numeric_suite_fx.add_loggers|Yes|-|
|torch.ao.ns._numeric_suite_fx.extract_logger_info|Yes|-|
|torch.ao.ns._numeric_suite_fx.add_shadow_loggers|Yes|-|
|torch.ao.ns._numeric_suite_fx.extract_shadow_logger_info|Yes|-|
|torch.ao.ns._numeric_suite_fx.extend_logger_results_with_comparison|Yes|-|
|torch.ao.ns._numeric_suite_fx.prepare_n_shadows_model|Yes|-|
|torch.ao.ns._numeric_suite_fx.loggers_set_enabled|Yes|-|
|torch.ao.ns._numeric_suite_fx.loggers_set_save_activations|Yes|-|
|torch.ao.ns._numeric_suite_fx.convert_n_shadows_model|Yes|-|
|torch.ao.ns._numeric_suite_fx.extract_results_n_shadows_model|Yes|-|
|torch.ao.ns._numeric_suite_fx.print_comparisons_n_shadows_model|Yes|-|
|torch.ao.ns.fx.utils.compute_sqnr|No|-|
