__all__ = ["NPUShapeHandling"]

from typing import Any, Callable, Dict, List, Optional, Tuple

import copy

import torch
import torch_npu._C


class NPUShapeHandling(torch_npu._C._NPUShapeHandling):
    r"""Wrapper around a NPU shape handling configuration.
    Args:
        min_size: Minimum size of shape handling (default: 1).
        max_size: Maximum size of shape handling (default: 256).
        policy: Policy type for shape handling generation (e.g., "TIMES") (default: "TIMES").
        sizes: Predefined list of sizes (optional). If provided, overrides min_size/max_size.
        dim: Dimension to change shape.
        value: Real value for padding.
        transform_pre_fn: Pre-processing function to convert inputs to tensor lists for transformation (optional).
        transform_post_fn: Post-processing function to convert tensor lists to structured outputs for transformation (optional).
        recover_pre_fn: Pre-processing function to convert inputs to tensor lists for recovery (optional).
        recover_post_fn: Post-processing function tp convert tensor lists to structured outputs for recovery (optional).
    """
    def __init__(
        self,
        min_size: int = 1,
        max_size: int = 256,
        policy: str = "TIMES",
        dim: int = 0,
        value: float = 0.0,
        sizes: Optional[List[int]] = None,
        indexs: Optional[List[int]] = None,
        transform_pre_fn: Optional[Callable[..., List[torch.Tensor]]] = None,
        transform_post_fn: Optional[Callable[[List[List[torch.Tensor]]], Tuple[List[Tuple], List[Dict]]]] = None,
        recover_pre_fn: Optional[Callable[[List[Any]], List[List[torch.Tensor]]]] = None,
        recover_post_fn: Optional[Callable[[List[torch.Tensor]], torch.Tensor]] = None,
    ) -> None:
        self.dim = dim
        self.value = value
        self.indexs = indexs
        # Register processing functions
        self.transform_pre_fn = transform_pre_fn
        self.transform_post_fn = transform_post_fn
        self.recover_pre_fn = recover_pre_fn
        self.recover_post_fn = recover_post_fn

        # Initialize base class based on sizes parameter
        if sizes is not None:
            return super().__init__(sizes)
        return super().__init__(min_size, max_size, policy)

    def transform(self,
                  tensors: List[torch.Tensor],
                  indexs: Optional[List[int]] = None
    ) -> List[List[torch.Tensor]]:
        if indexs is not None and len(indexs) > 0:
            return super().transform(tensors, indexs, self.dim, self.value)
        
        if self.indexs is not None and len(self.indexs) > 0:
            return super().transform(tensors, self.indexs, self.dim, self.value)
        
        indexs = [i for i in range(len(tensors))]
        return super().transform(tensors, indexs, self.dim, self.value)
    
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
            inputs, args_output, args_trans_indices, kwargs_output, kwargs_trans_keys = self._transform_preprocess_inputs(args, kwargs)
        
        # 执行核心转换操作
        trans_outputs = self.transform(tensors=inputs)
        
        # 后处理阶段优化：避免嵌套循环
        if self.transform_post_fn:
            outputs = self.transform_post_fn(trans_outputs)
        else:
            outputs = self._transform_postprocess_outputs(
                trans_outputs, 
                args_output, 
                args_trans_indices, 
                kwargs_output, 
                kwargs_trans_keys
            )
        
        return outputs

    def _transform_preprocess_inputs(
        self, 
        args: Tuple[Any], 
        kwargs: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], List[Any], List[int], Dict[str, Any], List[str]]:
        """预处理输入参数，分离张量与非张量数据"""
        # 初始化中间数据结构
        inputs = []
        args_output = list(args)
        args_trans_indices = []
        kwargs_output = dict(kwargs)
        kwargs_trans_keys = []
        
        # 分离位置参数中的张量
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                inputs.append(arg)
                args_trans_indices.append(i)
                args_output[i] = None  # 标记待替换位置
        
        # 分离关键字参数中的张量
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                inputs.append(v)
                kwargs_trans_keys.append(k)
                kwargs_output[k] = None  # 标记待替换位置
        
        return inputs, args_output, args_trans_indices, kwargs_output, kwargs_trans_keys

    def _transform_postprocess_outputs(
        self,
        trans_outputs: List[List[torch.Tensor]],
        args_output: List[Any],
        args_trans_indices: List[int],
        kwargs_output: Dict[str, Any],
        kwargs_trans_keys: List[str]
    ) -> Tuple[List[Tuple], List[Dict]]:
        """重组转换后的张量到原始参数结构"""
        new_args_list = []
        new_kwargs_list = []
        num_args_tensors = len(args_trans_indices)
        
        for tensor_group in trans_outputs:
            # 复制参数模板（浅拷贝避免修改原始模板）
            new_args = args_output.copy()
            new_kwargs = kwargs_output.copy()
            
            # 填充位置参数
            for idx, tensor_idx in enumerate(args_trans_indices):
                if idx < len(tensor_group):
                    new_args[tensor_idx] = tensor_group[idx]
            
            # 填充关键字参数
            for idx, key in enumerate(kwargs_trans_keys, start=num_args_tensors):
                if idx < len(tensor_group):
                    new_kwargs[key] = tensor_group[idx]
            
            new_args_list.append(tuple(new_args))
            new_kwargs_list.append(new_kwargs)
        
        return new_args_list, new_kwargs_list

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
        inputs = self.recover_pre_fn(groups) if self.recover_pre_fn else self._recover_preprocess_inputs(groups)
        
        # 执行恢复操作
        re_outputs = self.recover(tensor_groups=inputs)
        
        # 后处理：使用自定义函数或默认方法
        outputs = self.recover_post_fn(re_outputs) if self.recover_post_fn else self._recover_postprocess_outputs(re_outputs)
        
        return outputs

    def _recover_preprocess_inputs(self, groups: List[Any]) -> List[List[torch.Tensor]]:
        """
        Convert input groups into a list of tensor lists.
        
        Args:
            groups: Input data, each element should be a tensor or tensor list.
        
        Returns:
            Nested list of tensors.
        
        Raises:
            TypeError: If any element is neither a tensor nor a tensor list.
        """
        processed = []
        for group in groups:
            if isinstance(group, torch.Tensor):
                processed.append([group])
            else:
                raise TypeError(f"Input must be torch.Tensor or list of tensors, got {type(group)}")
        return processed

    def _recover_postprocess_outputs(self, re_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Convert list of output tensors into a tuple."""
        return re_outputs[0]


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
                trans_pre_fn = function_dict['trans_pre_fn']
                trans_post_fn = function_dict['trans_post_fn']
                re_pre_fn = function_dict['re_pre_fn']
                re_post_fn = function_dict['re_post_fn']

            self.shape_handling = NPUShapeHandling(
                min_size=compiler_config.get("shape_handling_min_size", 1),
                max_size=compiler_config.get("shape_handling_max_size", 256),
                policy=compiler_config.get("shape_handling_policy", "TIMES"),
                sizes=compiler_config.get("shape_handling_sizes"),
                indexs=compiler_config.get("shape_handling_indexs"),
                dim=compiler_config.get("shape_handling_dim"),
                value=compiler_config.get("shape_handling_value"),
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


def patch_inductor_get_config_copy():
    from torch.utils._config_module import Config, ConfigModule, _ConfigEntry

    src_get_config_copy = ConfigModule.get_config_copy

    def new_get_config_copy(self) -> Dict[str, Any]:
        ori_dict = src_get_config_copy(self)
        if "triton.cudagraphs" not in ori_dict:
            return ori_dict
        if "enable_shape_handling" in ori_dict:
            return ori_dict
        
        ori_dict["enable_shape_handling"] = False
        ori_dict["shape_handling_min_size"] = 1
        ori_dict["shape_handling_max_size"] = 256
        ori_dict["shape_handling_policy"] = "TIMES"
        ori_dict["shape_handling_sizes"] = None
        ori_dict["shape_handling_indexs"] = None
        ori_dict["shape_handling_dim"] = 0
        ori_dict["shape_handling_value"] = 0
        ori_dict["shape_handling_dict"] = None
        self._config["enable_shape_handling"] = _ConfigEntry(
            Config(default=False, value_type=bool)
        )
        self._config["shape_handling_min_size"] = _ConfigEntry(
            Config(default=1, value_type=int)
        )
        self._config["shape_handling_max_size"] = _ConfigEntry(
            Config(default=256, value_type=int)
        )
        self._config["shape_handling_policy"] = _ConfigEntry(
            Config(default="TIMES", value_type=str)
        )
        self._config["shape_handling_sizes"] = _ConfigEntry(
            Config(default=None, value_type=list)
        )
        self._config["shape_handling_indexs"] = _ConfigEntry(
            Config(default=None, value_type=list)
        )
        self._config["shape_handling_dim"] = _ConfigEntry(
            Config(default=0, value_type=int)
        )
        self._config["shape_handling_value"] = _ConfigEntry(
            Config(default=0.0, value_type=float)
        )
        self._config["shape_handling_dict"] = _ConfigEntry(
            Config(default=None, value_type=dict)
        )
        return ori_dict
    ConfigModule.get_config_copy = new_get_config_copy


def patch_shape_handling():
    patch_dynamo_context()
    patch_inductor_get_config_copy()