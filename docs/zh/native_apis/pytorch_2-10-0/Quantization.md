# Quantization

> [!NOTE]   
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.ao.quantization.prepare_qat|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.convert|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.QuantStub|是|-|
|torch.ao.quantization.DeQuantStub|是|-|
|torch.ao.quantization.QuantWrapper|是|-|
|torch.ao.quantization.qconfig_mapping.QConfigMapping|是|-|
|torch.ao.quantization.qconfig_mapping.QConfigMapping.from_dict|是|-|
|torch.ao.quantization.qconfig_mapping.QConfigMapping.set_global|是|-|
|torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name|是|-|
|torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_object_type_order|是|-|
|torch.ao.quantization.qconfig_mapping.QConfigMapping.set_module_name_regex|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.qconfig_mapping.QConfigMapping.set_object_type|是|-|
|torch.ao.quantization.qconfig_mapping.QConfigMapping.to_dict|是|-|
|torch.ao.quantization.qconfig_mapping.get_default_qconfig_mapping|是|-|
|torch.ao.quantization.qconfig_mapping.get_default_qat_qconfig_mapping|是|-|
|torch.ao.quantization.backend_config.BackendConfig|是|-|
|torch.ao.quantization.backend_config.BackendConfig.configs|是|-|
|torch.ao.quantization.backend_config.BackendConfig.from_dict|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_config|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.backend_config.BackendConfig.set_backend_pattern_configs|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.backend_config.BackendConfig.set_name|是|-|
|torch.ao.quantization.backend_config.BackendConfig.to_dict|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.backend_config.BackendPatternConfig|是|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.add_dtype_config|是|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.from_dict|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.set_dtype_configs|是|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.set_fused_module|是|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.set_fuser_method|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.set_observation_type|是|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.set_pattern|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.set_qat_module|是|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.set_reference_quantized_module|是|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module|是|-|
|torch.ao.quantization.backend_config.BackendPatternConfig.to_dict|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.backend_config.DTypeConfig|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.backend_config.DTypeConfig.from_dict|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.backend_config.DTypeConfig.to_dict|是|-|
|torch.ao.quantization.backend_config.DTypeWithConstraints|是|-|
|torch.ao.quantization.backend_config.ObservationType|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.backend_config.ObservationType.INPUT_OUTPUT_NOT_OBSERVED|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.backend_config.ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.backend_config.ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT|是|-|
|torch.ao.quantization.fx.custom_config.FuseCustomConfig|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.fx.custom_config.FuseCustomConfig.from_dict|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.fx.custom_config.FuseCustomConfig.set_preserved_attributes|是|-|
|torch.ao.quantization.fx.custom_config.FuseCustomConfig.to_dict|是|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict|是|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_float_to_observed_mapping|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_input_quantized_indexes|是|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_classes|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_non_traceable_module_names|是|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_output_quantized_indexes|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_preserved_attributes|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_class|是|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.set_standalone_module_name|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.fx.custom_config.PrepareCustomConfig.to_dict|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.fx.custom_config.ConvertCustomConfig|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict|是|-|
|torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_observed_to_quantized_mapping|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.fx.custom_config.ConvertCustomConfig.set_preserved_attributes|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.fx.custom_config.ConvertCustomConfig.to_dict|是|-|
|torch.ao.quantization.fx.custom_config.StandaloneModuleConfigEntry|是|-|
|torch.ao.quantization.observer.ObserverBase|是|-|
|torch.ao.quantization.observer.ObserverBase.with_args|是|-|
|torch.ao.quantization.observer.ObserverBase.with_callable_args|是|-|
|torch.ao.quantization.observer.MinMaxObserver|是|-|
|torch.ao.quantization.observer.MinMaxObserver.calculate_qparams|是|-|
|torch.ao.quantization.observer.MinMaxObserver.forward|是|-|
|torch.ao.quantization.observer.MinMaxObserver.reset_min_max_vals|是|-|
|torch.ao.quantization.observer.MovingAverageMinMaxObserver|是|-|
|torch.ao.quantization.observer.PerChannelMinMaxObserver|是|-|
|torch.ao.quantization.observer.PerChannelMinMaxObserver.reset_min_max_vals|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.observer.HistogramObserver|是|-|
|torch.ao.quantization.observer.PlaceholderObserver|是|-|
|torch.ao.quantization.observer.RecordingObserver|是|可能回退至CPU执行|
|torch.ao.quantization.observer.NoopObserver|是|-|
|torch.ao.quantization.observer.get_observer_state_dict|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.observer.load_observer_state_dict|是|-|
|torch.ao.quantization.observer.default_observer|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.observer.default_placeholder_observer|是|-|
|torch.ao.quantization.observer.default_debug_observer|是|-|
|torch.ao.quantization.observer.default_weight_observer|是|-|
|torch.ao.quantization.observer.default_histogram_observer|是|-|
|torch.ao.quantization.observer.default_per_channel_weight_observer|是|-|
|torch.ao.quantization.observer.default_dynamic_quant_observer|是|-|
|torch.ao.quantization.observer.default_float_qparams_observer|是|-|
|torch.ao.quantization.fake_quantize.FakeQuantize|是|可能回退至CPU执行|
|torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize|是|可能回退至CPU执行|
|torch.ao.quantization.fake_quantize.disable_fake_quant|是|-|
|torch.ao.quantization.fake_quantize.enable_fake_quant|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.fake_quantize.disable_observer|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.fake_quantize.enable_observer|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.qconfig.QConfig|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.qconfig.default_qconfig|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.qconfig.default_debug_qconfig|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.qconfig.default_per_channel_qconfig|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.qconfig.default_dynamic_qconfig|是|-|
|torch.ao.quantization.qconfig.float16_dynamic_qconfig|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.qconfig.float16_static_qconfig|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.qconfig.per_channel_dynamic_qconfig|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig|是|-|
|torch.ao.quantization.qconfig.default_qat_qconfig|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.qconfig.default_weight_only_qconfig|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.quantization.qconfig.default_activation_only_qconfig|是|-|
|torch.ao.quantization.qconfig.default_qat_qconfig_v2|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.ao.nn.intrinsic.LinearReLU|否|-|
|torch.ao.nn.intrinsic.qat.LinearReLU|否|-|
|torch.ao.nn.intrinsic.qat.ConvBn1d|否|-|
|torch.ao.nn.intrinsic.qat.ConvBnReLU1d|否|-|
|torch.ao.nn.intrinsic.qat.ConvBnReLU2d|否|-|
|torch.ao.nn.intrinsic.qat.update_bn_stats|是|-|
|torch.ao.nn.intrinsic.qat.freeze_bn_stats|是|可能回退至CPU执行|
|torch.ao.nn.qat.Linear|否|-|
|torch.ao.nn.quantizable.LSTM|否|-|
|torch.ao.nn.quantized.dynamic.Linear|否|-|
|torch.ao.nn.quantized.dynamic.LSTM|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.ao.nn.quantized.dynamic.GRU|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.ao.nn.quantized.dynamic.RNNCell|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.ao.nn.quantized.dynamic.LSTMCell|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.ao.nn.quantized.dynamic.GRUCell|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.ao.ns.fx.utils.compute_sqnr|否|-|
