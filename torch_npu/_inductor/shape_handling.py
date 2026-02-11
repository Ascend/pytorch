__all__ = ["NPUShapeHandling"]

from typing import Any, Callable, Dict, List, Optional, Tuple
import copy

from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
import torch
import torch_npu._C


class NPUShapeHandling(torch_npu._C._NPUShapeHandling):
    r"""Wrapper around a NPU shape handling configuration.
    Args:
        configs: List of configuration dictionaries that define shape handling rules.
        transform_pre_fn: Pre-processing function to convert inputs to tensor lists for transformation (optional).
        transform_post_fn: Post-processing function to convert tensor lists to structured outputs for transformation (optional).
        recover_pre_fn: Pre-processing function to convert inputs to tensor lists for recovery (optional).
        recover_post_fn: Post-processing function tp convert tensor lists to structured outputs for recovery (optional).
    
    Each config dictionary in configs supports the following keys:
        - type (str):
            Logical dimension type. Supported values:
            "BATCHSIZE" | "SEQLEN"
        - dimensions (int or List[int]):
            For BATCHSIZE: all affected tensors must share the same batch dimensions, if a list is provided, only the
                           first element is used.
            For SEQLEN: if an int or a single-element list is provided, the value is automatically applied to all affected tensors;
                        if a list is provided, it specifies the sequence dimension position for each affected tensor respectively,
                        allowing different tensors to have the sequence dimension at different positions.
        - indices (List[int]):
            Indices of tensors that this rule applies to. Empty list means "apply to all tensors".
        - value (float):
            Padding value used when increasing size to reach the next gear.
        - gears (List[int]):
            Explicit list of allowed sizes(gears). If non-empty, overrides min_size/max_size/policy.
        - min_size (int):
            Minimum allowed size for this dimension (inclusive). Default: 1.
        - max_size (int):
            Maximun allowed size for this dimension (inclusive). Default: 1024.
        - policy (str):
            Gear generation strategy. Supported values:
            "TIMES" | "CUSTOM"
    
    If no configs are provided at construction, a default configuration handling batch size on dimension 0 is created.
    """
    def __init__(
        self,
        configs: List[Dict[str, Any]] = None,
        transform_pre_fn: Optional[Callable[..., List[torch.Tensor]]] = None,
        transform_post_fn: Optional[Callable[[List[List[torch.Tensor]]], Tuple[List[Tuple], List[Dict]]]] = None,
        recover_pre_fn: Optional[Callable[[List[Any]], List[List[torch.Tensor]]]] = None,
        recover_post_fn: Optional[Callable[[List[torch.Tensor]], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.delay_init = False
        self.shape_type_map = {
            "BATCHSIZE": torch_npu._C.ShapeType.BATCHSIZE,
            "SEQLEN": torch_npu._C.ShapeType.SEQLEN
        }

        self.policy_map = {
            "TIMES": torch_npu._C.ShapePolicy.TIMES,
            "CUSTOM": torch_npu._C.ShapePolicy.CUSTOM
        }
        
        # Register processing functions
        self.transform_pre_fn = transform_pre_fn
        self.transform_post_fn = transform_post_fn
        self.recover_pre_fn = recover_pre_fn
        self.recover_post_fn = recover_post_fn
        if configs and len(configs) > 0:
            self._validate_configs(configs)
            self.configs = configs
            self._initialize_from_configs(configs)
        else:
            self.delay_init = True
            self.configs = [{
                "type": "BATCHSIZE",
                "dimensions": [0],
                "indices": [],
                "value": 0.0,
                "gears": [],
                "min_size": 1,
                "max_size": 1024,
                "policy": "TIMES"
            }]

    def _validate_configs(self, configs: List[Dict[str, Any]]) -> None:
        if not configs or len(configs) == 0:
            return
        if len(configs) > 2:
            raise ValueError("NPUShapeHandling currently supports only two dimensions.")
        
        required_fields = ["type"]
        int_list_fields = ["dimensions", "indices", "gears"]
        int_fields = ["min_size", "max_size"]
        for i, config in enumerate(configs):
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Config {i} missing required field: {field}.")

            if not isinstance(config["type"], str):
                raise ValueError(f"Config {i} {field} must be a str, got {type(config['type'])}.")
            if config["type"] not in self.shape_type_map:
                raise ValueError(
                    f"Invalid 'type' in config[{i}]: {config['type']}. "
                    f"Must be one of: {', '.join(repr(k) for k in self.shape_type_map.keys())}."
                )
            
            for field in int_list_fields:
                if field not in config:
                    continue
                
                if field == "dimensions":
                    if isinstance(config[field], int):
                        config[field] = [config[field]]
                    if config["type"] == "BATCHSIZE" and len(config[field]) > 1:
                        warning.warn("For BATCHSIZE, only the first element of 'dimensions' is used")
                        config[field] = config[field][0]

                if not isinstance(config[field], (list, tuple)):
                    raise ValueError(f"Config {i} {field} must be a list, got {type(config[field])}.")
                
                for item in config[field]:
                    if not isinstance(item, int):
                        raise ValueError(f"Config {i} {field} must contain integers, got {type(item)}.")
            
            for field in int_fields:
                if field not in config:
                    continue
                if not isinstance(config[field], int):
                    raise ValueError(f"Config {i} {field} must be an integer, got {type(config[field])}.")
            
            if "value" in config and not isinstance(config["value"], (int, float)):
                raise ValueError(f"Config {i} 'value' must be a number, got {type(config['value'])}.")
            
            if "policy" in config:
                if not isinstance(config["policy"], str):
                    raise ValueError(f"Config {i} 'policy' must be a str, got {type(config['policy'])}.")
                if config["policy"] not in self.policy_map:
                    raise ValueError(
                        f"Invalid 'policy' in config[{i}]: {config['policy']}. "
                        f"Must be one of: {', '.join(repr(k) for k in self.policy_map.keys())}."
                    )
                
        if len(configs) == 2 and configs[0]["type"] == configs[1]["type"]:
            raise ValueError("Cannot initialize the same type repeatedly.")

    def _initialize_from_configs(self, configs: List[Dict[str, Any]]) -> None:
        for config in configs:
            shape_type = self.shape_type_map.get(config.get("type"))
            indices = config.get("indices", [])
            value = config.get("value", 0.0)
            gears = config.get("gears", [])

            dimensions = config.get("dimensions", [])
            if shape_type == torch_npu._C.ShapeType.BATCHSIZE:
                if not dimensions:
                    # Empty list
                    dimensions = [0]
            elif shape_type == torch_npu._C.ShapeType.SEQLEN and len(indices) != 0:
                if len(dimensions) == 1:
                    dimensions = [dimensions[0] for _ in range(len(indices))]
                if not dimensions:
                    dimensions = [1 for _ in range(len(indices))]
                
            
            if len(dimensions) == 0 or len(indices) == 0:
                self.delay_init = True
                continue

            if len(gears) > 0:
                self.initialize(shape_type, gears, dimensions, indices, value)
            else:
                min_size = config.get("min_size", 1)
                max_size = config.get("max_size", 1024)
                policy = self.policy_map.get(config.get("policy", "TIMES"))
                self.initialize(shape_type, min_size, max_size, policy, dimensions, indices, value)

    def _construct_indices(self, tensors: List[torch.Tensor], dimensions, dimension_type):
        if dimension_type == "BATCHSIZE":
            if not dimensions:
                dimensions = [0]
            dimensions = [dimensions[0] for _ in range(len(tensors))]
        
        if dimension_type == "SEQLEN":
            if not dimensions:
                dimensions = [1]
            if len(dimensions) == 1:
                dimensions = [dimensions[0] for _ in range(len(tensors))]
        
        index = 0
        indices = []
        for dimension, tensor in zip(dimensions, tensors):
            if tensor.ndim > dimension:
                indices.append(index)
            index += 1
        
        return indices
        
    def delay_initialize(self, tensors: List[torch.Tensor]):
        delay_init_configs = []
        for config in self.configs:
            init_flag = False
            if "indices" not in config or len(config["indices"]) == 0:
                init_flag = True
                config["indices"] = self._construct_indices(tensors, config.get("dimensions", []), config["type"])
            
            if init_flag:
                delay_init_configs.append(config)
        if len(delay_init_configs) > 0:
            self._initialize_from_configs(delay_init_configs)
        self.delay_init = False

    def transform(self, tensors: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        if self.delay_init:
            self.delay_initialize(tensors)
        return super().transform(tensors)

    def recover(self, tensor_groups: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        return super().recover(tensor_groups)

    def transform_hook(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Tuple[List[Tuple], List[Dict]]:
        # 预处理阶段优化：统一使用预定义函数或默认逻辑
        if self.transform_pre_fn:
            inputs = self.transform_pre_fn(*args, **kwargs)
        else:
            inputs, indices, leaves, spec = self._process_inputs(args, kwargs)
        
        # 执行核心转换操作
        trans_outputs = self.transform(tensors=inputs)
        
        # 后处理阶段优化：避免嵌套循环
        if self.transform_post_fn:
            outputs = self.transform_post_fn(trans_outputs)
        else:
            outputs = self._recover_inputs(trans_outputs, indices, leaves, spec)
        
        return outputs

    def flatten_to_tensors(self, structure: Any) -> Tuple[List[torch.Tensor], List[int], List[Any], TreeSpec]:
        leaves, spec = tree_flatten(structure)
        indexed_tensors = [(i, leaf) for i, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)]
        indices = []
        tensors = []
        if indexed_tensors is not None and len(indexed_tensors) > 0:
            indices, tensors = zip(*indexed_tensors)
        return tensors, indices, leaves, spec
    
    def unflatten_from_tensors(
        self,
        tensors: List[torch.Tensor],
        indices: List[int],
        leaves: List[Any],
        spec: TreeSpec
    ) -> Any:
        for idx, tensor in zip(indices, tensors):
            leaves[idx] = tensor
        return tree_unflatten(leaves, spec)

    def _process_inputs(self, args: Tuple, kwargs: dict) -> List[torch.Tensor]:
        return self.flatten_to_tensors((args, kwargs))
    
    def _recover_inputs(
        self,
        transform_res: List[List[torch.Tensor]],
        indices: List[int],
        leaves: List[Any],
        spec: TreeSpec
    ) -> Tuple[List[Tuple], List[Dict]]:
        res = []
        for processd_tensors in transform_res:
            res.append(self.unflatten_from_tensors(processd_tensors, indices, list(leaves), spec))
        return zip(*res)
    
    def _process_outputs(
        self,
        outputs_list: List[Any]
    ) -> Tuple[List[List[torch.Tensor]], List[int], List[Any], TreeSpec]:
        tensors_list = []
        leaves = []
        indices = []
        spec = None
        for output in outputs_list:
            tensors, indices, leaves, spec = self.flatten_to_tensors(output)
            tensors_list.append(tensors)
        return tensors_list, indices, leaves, spec

    def _recover_outputs(
        self,
        recover_res: List[torch.Tensor],
        indices: List[int],
        leaves: List[Any],
        spec: TreeSpec
    ) -> Any:
        return self.unflatten_from_tensors(recover_res, indices, leaves, spec)

    def recover_hook(
        self,
        groups: List[Any]
    ) -> Any:
        """
        Process input groups through recovery pipeline.
        
        Args:
            groups: List of input data to be processed.
        
        Returns:
            Processed outputs after recovery and postprocessing.
        """
        # 预处理：使用自定义函数或默认方法
        if self.recover_pre_fn:
            inputs = self.recover_pre_fn(groups)
        else:
            inputs, indices, leaves, spec = self._process_outputs(groups)
        
        # 执行恢复操作
        re_outputs = self.recover(tensor_groups=inputs)
        
        # 后处理：使用自定义函数或默认方法
        if self.recover_post_fn:
            outputs = self.recover_post_fn(re_outputs)
        else:
            outputs = self._recover_outputs(re_outputs, indices, leaves, spec)
        
        return outputs


def unified_copy(data: Any) -> Any:
    """
    对输入数据进行安全且统一的深拷贝。
    支持PyTorch Tensor、字典、列表等常见数据类型。
    
    Args:
        data: 输入数据，可以是Tensor、dict、list等
        
    Returns:
        数据的独立副本
    """
    if data is None:
        return None
    
    # 处理PyTorch Tensor
    if isinstance(data, torch.Tensor):
        return data.clone().detach()
    
    # 处理字典类型
    elif isinstance(data, dict):
        return {key: unified_copy(value) for key, value in data.items()}
    
    # 处理列表类型
    elif isinstance(data, list):
        return [unified_copy(item) for item in data]
    
    # 处理元组类型
    elif isinstance(data, tuple):
        return tuple(unified_copy(item) for item in data)

    else:
        try:
            return copy.deepcopy(data)
        except (TypeError, ValueError):
            return data


def patch_dynamo_context():
    import contextlib
    import inspect
    from torch._dynamo.eval_frame import _TorchDynamoContext
    from torch._dynamo.types import DynamoCallback
    from torch._dynamo.convert_frame import CatchErrorsWrapper, ConvertFrame
    from torch._dynamo.repro.after_dynamo import WrapBackendDebug
    src_call = _TorchDynamoContext.__call__
    src_init = _TorchDynamoContext.__init__
    null_context = contextlib.nullcontext

    def is_enable_shape_handling(callback: DynamoCallback, compiler_config=None):
        """
        The shape handling feature is only available when enable_shape_handling is True and the backend is inductor
        """
        if compiler_config is None or not compiler_config.get("enable_shape_handling", False):
            return False
        
        if not isinstance(callback, CatchErrorsWrapper):
            return False
        
        orig_callable = callback._torchdynamo_orig_callable
        if not isinstance(orig_callable, ConvertFrame):
            return False
        
        deep_callable = orig_callable._torchdynamo_orig_callable
        if not isinstance(deep_callable, WrapBackendDebug):
            return False
        
        if getattr(deep_callable, "_compiler_name", None) != "inductor":
            return False
        
        return True

    def nothing():
        pass

    def new_init(self,
                 callback: DynamoCallback,
                 on_enter=nothing,
                 backend_ctx_ctor=null_context,
                 patch_fn=nothing,
                 first_ctx=False,
                 *,
                 export=False,
                 dynamic=None,
                 compiler_config=None,
        ) -> None:
        src_init(
            self,
            callback,
            on_enter=on_enter,
            backend_ctx_ctor=backend_ctx_ctor,
            patch_fn=patch_fn,
            first_ctx=first_ctx,
            export=export,
            dynamic=dynamic,
            compiler_config=compiler_config
        )
        if (is_enable_shape_handling(callback, compiler_config=compiler_config)):
            trans_pre_fn = None
            trans_post_fn = None
            re_pre_fn = None
            re_post_fn = None
            function_dict = compiler_config.get("shape_handling_dict")
            if function_dict is not None:
                trans_pre_fn = function_dict.get("trans_pre_fn", None)
                trans_post_fn = function_dict.get("trans_post_fn", None)
                re_pre_fn = function_dict.get("re_pre_fn", None)
                re_post_fn = function_dict.get("re_post_fn", None)
            
            self.shape_handling = NPUShapeHandling(
                configs=compiler_config.get("shape_handling_configs"),
                transform_pre_fn=trans_pre_fn,
                transform_post_fn=trans_post_fn,
                recover_pre_fn=re_pre_fn,
                recover_post_fn=re_post_fn,
            )

    def new_call(self, fn):
        src_fn = src_call(self, fn)
        if isinstance(fn, torch.nn.Module) or inspect.isclass(fn):
            return src_fn
        
        def new_fn(*args, **kwargs):
            if (is_enable_shape_handling(self.callback, compiler_config=self.compiler_config)):
                new_args, new_kwargs = self.shape_handling.transform_hook(*args, **kwargs)
                args_is_split = len(args) != 0 and len(new_args) > 1
                kwargs_is_split = len(kwargs) != 0 and len(new_kwargs) > 1
                zipped_params = zip(new_args, new_kwargs)
                res = [
                    unified_copy(src_fn(*arg, **kwargs)) if args_is_split or kwargs_is_split
                    else src_fn(*arg, **kwargs)
                    for arg, kwargs in zipped_params
                ]
                return self.shape_handling.recover_hook(res)
            return src_fn(*args, **kwargs)
        return new_fn
    _TorchDynamoContext.__call__ = new_call
    _TorchDynamoContext.__init__ = new_init


def patch_shape_handling():
    if getattr(patch_shape_handling, "_is_patched", False):
        return
    patch_dynamo_context()
    patch_shape_handling._is_patched = True