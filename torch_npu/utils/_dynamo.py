import importlib
import importlib.abc
import copy
import functools
import inspect
import logging
import os
import sys
import threading
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch_npu
import torch_npu._C
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten
from torch import _TorchCompileWrapper


use_jit_script = False
log = logging.getLogger(__name__)


def _create_npu_autocast_mode_variable(func, args, kwargs):
    from torch._dynamo.variables.base import VariableTracker
    from torch._dynamo.variables.ctx_manager import AutocastModeVariable

    bound_args = inspect.signature(func).bind(*args, **kwargs)
    bound_args.apply_defaults()
    target_values = []
    kwargs.clear()

    for key in ["device_type", "dtype", "enabled", "cache_enabled"]:
        if key == "device_type" and func in [
            torch_npu.npu.amp.autocast,
        ]:
            arg = "npu" if func is torch_npu.npu.amp.autocast else "cpu"
        else:
            arg = bound_args.arguments[key]
        if isinstance(arg, VariableTracker):
            target_values.append(arg.as_python_constant())
        else:
            target_values.append(arg)

    return AutocastModeVariable(target_values, initial_values=None, **kwargs)


def patch_SkipFunctionVariable():
    from torch._dynamo.variables.functions import SkipFunctionVariable
    from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

    def SkipFunctionVariable__new__(cls, value, reason=None, **kwargs):
        if value in [
            torch.npu.stream,
            torch_npu.npu.stream,
            torch_npu.npu.utils.stream,
        ]:
            return TorchInGraphFunctionVariable(value, **kwargs)
        return cls.__new__raw(cls)

    SkipFunctionVariable.__new__raw = SkipFunctionVariable.__new__
    SkipFunctionVariable.__new__ = SkipFunctionVariable__new__


def patch_TensorVariable_call_method():
    from torch._dynamo.utils import tensortype_to_dtype
    from torch._dynamo.variables.constant import ConstantVariable
    from torch._dynamo.variables.lists import TupleVariable
    from torch._dynamo.variables.tensor import TensorVariable

    def TensorVariable_call_method(self, tx, name, args, kwargs):
        if (
            name == "type"
            and self.dtype is not None
            and len(args) == 0
            and isinstance(self.device, torch.device)
            and self.device.type == "npu"
        ):
            tensortype = next(
                k for k, v in tensortype_to_dtype.items() if self.dtype in v
            )
            constant_result = ConstantVariable.create(
                f"torch.npu.{tensortype.__name__}"
            )

            if len(args) == 1:
                return constant_result.getitem_const(args[0])
            if args:
                return TupleVariable(
                    [constant_result.getitem_const(a) for a in args]
                )
            return constant_result
        return TensorVariable.call_method_raw(self, tx, name, args, kwargs)

    TensorVariable.call_method_raw = TensorVariable.call_method
    TensorVariable.call_method = TensorVariable_call_method


class _InductorNpuRegistry:
    _disabled_register = False
    _loaded_backend = None

    @classmethod
    def register_inductor_npu(cls):
        if cls._disabled_register:
            return

        current = os.getenv("TORCHINDUCTOR_NPU_BACKEND", "default")
        if cls._loaded_backend != current:
            if "torch_npu._inductor" not in sys.modules:
                importlib.import_module("torch_npu._inductor")
            else:
                sys.modules["torch_npu._inductor"]._load_backend()
            cls._loaded_backend = current


    @classmethod
    def disable_register(cls):
        cls._disabled_register = True

    @classmethod
    def enable_register(cls):
        cls._disabled_register = False

    @classmethod
    def has_initialized(cls):
        return cls._loaded_backend is not None


def is_inductor_npu_initialized():
    return _InductorNpuRegistry.has_initialized()


def disable_register_inductor_npu():
    _InductorNpuRegistry.disable_register()


def enable_register_inductor_npu():
    _InductorNpuRegistry.enable_register()


def register_inductor_npu():
    _InductorNpuRegistry.register_inductor_npu()


def _resolve_npu_backend(selected_backend=None) -> str:
    """Resolve NPU backend with priority: compile options > config > env."""
    if selected_backend not in (None, "", "default"):
        return selected_backend

    inductor_config = sys.modules.get("torch._inductor.config")
    global_backend = getattr(inductor_config, "npu_backend", None)
    if global_backend not in (None, "", "default"):
        return global_backend

    return os.getenv("TORCHINDUCTOR_NPU_BACKEND", "default")


def _resolve_npu_backend_from_wrapper(wrapper) -> str:
    return _resolve_npu_backend(wrapper.config.get("npu_backend"))


class _NpuBackendScope:
    """Apply resolved npu backend for one compile invocation and restore env."""

    def __init__(self, backend: str):
        self.backend = backend
        self._old_env = None

    def __enter__(self):
        self._old_env = os.environ.get("TORCHINDUCTOR_NPU_BACKEND")
        try:
            os.environ["TORCHINDUCTOR_NPU_BACKEND"] = self.backend
            register_inductor_npu()
            _lazy_inductor_setup()
            if self.backend == "ascendc":
                from torch_npu._inductor.deterministic_cache import (
                    patch_npu_deterministic_level_cache_keys,
                )

                patch_npu_deterministic_level_cache_keys()
        except BaseException:
            self._restore_backend_env()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        self._restore_backend_env()
        return False

    def _restore_backend_env(self):
        if self._old_env is None:
            os.environ.pop("TORCHINDUCTOR_NPU_BACKEND", None)
        else:
            os.environ["TORCHINDUCTOR_NPU_BACKEND"] = self._old_env


def patch_inductor_wrapper():
    from typing import Any, Literal, Optional

    from torch import _TorchCompileInductorWrapper
    from torch.utils._config_module import _ConfigEntry, Config, ConfigModule

    src_apply_options = _TorchCompileInductorWrapper.apply_options
    src_init = _TorchCompileInductorWrapper.__init__
    src_get_config_copy = ConfigModule.get_config_copy
    src_call = _TorchCompileInductorWrapper.__call__

    def new_apply_options(self, options: Optional[dict[str, Any]]):
        shape_handling_requested = (
            options is not None and options.get("enable_shape_handling", False)
        )
        src_apply_options(self, options)
        if shape_handling_requested:
            if getattr(self, "_npu_defer_shape_handling", False):
                self._npu_shape_handling_requested = True
            else:
                _patch_shape_handling()

    def new_get_config_copy(self) -> dict[str, Any]:
        ori_dict = src_get_config_copy(self)
        inductor_config = sys.modules.get("torch._inductor.config")
        if inductor_config is None or self is not inductor_config:
            return ori_dict
        NpuBackendType = Literal["default", "mlir", "dvm"]
        if "npu_backend" not in ori_dict:
            ori_dict["npu_backend"] = "default"
            self._config["npu_backend"] = _ConfigEntry(
                Config(default="default", value_type=NpuBackendType)
            )

        if "enable_shape_handling" not in ori_dict:
            ori_dict["enable_shape_handling"] = False
            self._config["enable_shape_handling"] = _ConfigEntry(
                Config(default=False, value_type=bool)
            )

        if "shape_handling_configs" not in ori_dict:
            ori_dict["shape_handling_configs"] = []
            self._config["shape_handling_configs"] = _ConfigEntry(
                Config(default=[], value_type=list)
            )

        if "shape_handling_dict" not in ori_dict:
            ori_dict["shape_handling_dict"] = None
            self._config["shape_handling_dict"] = _ConfigEntry(
                Config(default=None, value_type=dict)
            )
        return ori_dict

    def new_init(self, mode, options, dynamic):
        self._npu_defer_shape_handling = True
        self._npu_shape_handling_requested = False
        try:
            src_init(self, mode, options, dynamic)
            shape_handling_requested = self._npu_shape_handling_requested
        finally:
            del self._npu_defer_shape_handling
            del self._npu_shape_handling_requested
        _lazy_dynamo_setup()
        if shape_handling_requested:
            _patch_shape_handling()
        backend = _resolve_npu_backend_from_wrapper(self)
        if backend=="mlir":
            with _NpuBackendScope(backend):
                log.info("Running MLIR backend")
                device_id = torch_npu.npu.current_device()
                torch_npu._C._recovery_all_npu_stream(device_id)
        if backend=="dvm":
            with _NpuBackendScope(backend):
                log.info("Running dvm backend")

    def new_call(self, model_, inputs_):
        backend = _resolve_npu_backend_from_wrapper(self)
        with _NpuBackendScope(backend):
            if backend == "ascendc":
                from torch_npu.dynamo._deterministic_guard import (
                    install_npu_deterministic_level_guard,
                )

                install_npu_deterministic_level_guard()
            return src_call(self, model_, inputs_)

    _TorchCompileInductorWrapper.__call__ = new_call
    _TorchCompileInductorWrapper.apply_options = new_apply_options
    _TorchCompileInductorWrapper.__init__ = new_init
    ConfigModule.get_config_copy = new_get_config_copy


def patch_dynamo_optimize():
    from torch_npu.dynamo import _get_global_npu_backend

    src_optimize = torch._dynamo.optimize

    def npu_optimize(*args, **kwargs):
        backend = None
        if "backend" in kwargs:
            backend = kwargs["backend"]
        elif len(args) == 1:
            backend = args[0]

        backend_name = None
        if isinstance(backend, str):
            backend_name = backend
        elif isinstance(backend, _TorchCompileWrapper):
            backend_name = backend.compiler_name

        if backend_name == "npu":
            # Init torchair ahead of running model.
            _get_global_npu_backend(backend_name)
        return src_optimize(*args, **kwargs)

    torch._dynamo.optimize = npu_optimize


def patch_builtin_variable():
    origin_call_id = torch._dynamo.variables.builtin.BuiltinVariable.call_id

    def _wrap_call_id(self, tx, *args):
        if torch._dynamo.variables.builtin.istype(
            args[0], torch._dynamo.variables.streams.EventVariable
        ):
            return torch._dynamo.variables.ConstantVariable.create(id(args[0].value))
        return origin_call_id(self, tx, *args)

    torch._dynamo.variables.builtin.BuiltinVariable.call_id = _wrap_call_id


def patch_event_variable_python_type():
    """
    Add the 'python_type' method to the EventVariable class.
    """

    def python_type(self):
        return type(self.value)

    if "python_type" not in torch._dynamo.variables.streams.EventVariable.__dict__:
        torch._dynamo.variables.streams.EventVariable.python_type = python_type



def patch_npu_stream_context():
    from torch._dynamo.device_interface import get_interface_for_device
    from torch._dynamo.variables.base import VariableTracker
    from torch._dynamo.variables.streams import StreamContextVariable, StreamVariable
    from torch._dynamo.variables.torch import TorchInGraphFunctionVariable
    if TYPE_CHECKING:
        from torch._dynamo.symbolic_convert import InstructionTranslator

    class NpuStreamContextVariable(StreamContextVariable):
        """This represents NPU stream context with FX graph set_stream node creation."""

        @staticmethod
        def create(
            tx: "InstructionTranslator",
            stream_to_enter: "StreamVariable",
            **kwargs: dict[str, Any],
        ) -> "NpuStreamContextVariable":
            from torch._dynamo.device_interface import get_interface_for_device
            from torch._dynamo.variables.builder import wrap_fx_proxy_cls

            device_interface = get_interface_for_device(stream_to_enter.device)
            current_stream_var = wrap_fx_proxy_cls(
                StreamVariable,
                tx,
                tx.output.create_proxy(
                    "call_function",
                    device_interface.current_stream,
                    (None,),
                    {},
                ),
            )

            return NpuStreamContextVariable(
                stream_to_enter,
                current_stream=current_stream_var,
                device_interface=device_interface,
                **kwargs,
            )

        def __init__(
            self,
            stream: Optional["StreamVariable"],
            current_stream: Optional["StreamVariable"] = None,
            device_interface: Any | None = None,
            **kwargs: Any,
        ) -> None:
            self.current_stream = current_stream
            self.device_interface = device_interface
            super().__init__(stream, **kwargs)

        def enter(
            self, tx: "InstructionTranslator", *args: VariableTracker
        ) -> VariableTracker:
            if self.get_stream():
                tx.output.create_proxy(
                    "call_function",
                    self.device_interface.set_stream,
                    (self.get_stream().as_proxy(),),
                    {},
                )
            return super().enter(tx)

        def exit(
            self, tx: "InstructionTranslator", *args: VariableTracker
        ) -> VariableTracker:
            if self.get_stream():
                tx.output.create_proxy(
                    "call_function",
                    self.device_interface.set_stream,
                    (self.current_stream.as_proxy(),),
                    {},
                )
            return super().exit(tx, *args)

    def _handle_npu_device_interface_stream(self, tx, stream):
        return NpuStreamContextVariable.create(tx, stream)

    TorchInGraphFunctionVariable._get_handlers()[
        get_interface_for_device("npu").stream
    ] = _handle_npu_device_interface_stream


def fake_record_stream(self, s):
    """
    let dynamo trace Tensor.record_stream as this empty function,
    and you can replace it later in your compile backend to an actual function
    """
    if isinstance(self, torch._subclasses.fake_tensor.FakeTensor):
        return
    raise RuntimeError(
        "tensor.record_stream is not supported on torch.compile! "
        "You should write a pass to replace torch.npu.fake_record_stream to an actual function in FX graph "
        "before aot_autograd."
    )


def patch_record_stream():
    torch.npu.fake_record_stream = fake_record_stream

    def method_record_stream(self, s):
        tx = torch._dynamo.symbolic_convert.InstructionTranslator.current_tx()
        return torch._dynamo.variables.TorchInGraphFunctionVariable(
            torch.npu.fake_record_stream
        ).call_function(tx, [self, s], {})

    torch._dynamo.variables.tensor.TensorVariable.method_record_stream = (
        method_record_stream
    )


def patch_user_defined_class_variable():
    from torch._dynamo.variables.torch import (
        TorchCtxManagerClassVariable,
        TorchInGraphFunctionVariable,
    )
    from torch._dynamo.variables.user_defined import UserDefinedClassVariable

    original_method = UserDefinedClassVariable._in_graph_classes

    class NPUTorchCtxManagerClassVariable(TorchCtxManagerClassVariable):
        def call_function(self, tx, args, kwargs):
            return _create_npu_autocast_mode_variable(self.value, args, kwargs)

    @staticmethod
    @functools.lru_cache(None)
    def patched_in_graph_classes():
        result = original_method()
        result.add(torch.npu.Event)
        result.add(torch.npu.Stream)
        return result

    def UserDefinedClassVariable__new__(cls, value, **kwargs):
        if value in [
            torch.npu.amp.autocast,
            torch_npu.npu.amp.autocast,
            torch.npu.amp.autocast_mode.autocast,
            torch_npu.npu.amp.autocast_mode.autocast,
        ]:
            return NPUTorchCtxManagerClassVariable(value, **kwargs)
        if value in [
            torch_npu.npu.BoolTensor,
            torch_npu.npu.ByteTensor,
            torch_npu.npu.CharTensor,
            torch_npu.npu.DoubleTensor,
            torch_npu.npu.FloatTensor,
            torch_npu.npu.HalfTensor,
            torch_npu.npu.IntTensor,
            torch_npu.npu.LongTensor,
            torch_npu.npu.ShortTensor,
            torch_npu.npu.BFloat16Tensor,
        ]:
            return TorchInGraphFunctionVariable(value, **kwargs)
        return cls.__new__raw(cls)

    UserDefinedClassVariable._in_graph_classes = patched_in_graph_classes
    UserDefinedClassVariable.__new__raw = UserDefinedClassVariable.__new__
    UserDefinedClassVariable.__new__ = UserDefinedClassVariable__new__


def run_once(f):
    """Run a function successfully only once, waiting for concurrent callers."""
    condition = threading.Condition()

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        thread_id = threading.get_ident()
        with condition:
            while wrapper._is_running:
                if wrapper._running_thread == thread_id:
                    return None
                condition.wait()
            if wrapper.has_run:
                return None
            wrapper._is_running = True
            wrapper._running_thread = thread_id

        try:
            result = f(*args, **kwargs)
        except BaseException:
            with condition:
                wrapper._is_running = False
                wrapper._running_thread = None
                condition.notify_all()
            raise

        with condition:
            wrapper.has_run = True
            wrapper._is_running = False
            wrapper._running_thread = None
            condition.notify_all()
        return result

    wrapper.has_run = False
    wrapper._is_running = False
    wrapper._running_thread = None

    def reset_after_fork():
        # The parent thread running f may not exist in the child process.
        nonlocal condition
        condition = threading.Condition()
        wrapper._is_running = False
        wrapper._running_thread = None

    try:
        os.register_at_fork(after_in_child=reset_after_fork)
    except AttributeError:
        pass
    return wrapper


_COMPLETED_DYNAMO_SETUP_STEPS = set()


def _run_dynamo_setup_step(name, setup):
    """Keep successful setup steps idempotent when a later step fails."""
    if name in _COMPLETED_DYNAMO_SETUP_STEPS:
        return
    setup()
    _COMPLETED_DYNAMO_SETUP_STEPS.add(name)


@run_once
def _dynamo_register_interface_for_device():
    from torch._dynamo.device_interface import register_interface_for_device
    from torch_npu.utils._dynamo_device import NpuInterface

    register_interface_for_device("npu", NpuInterface)
    for i in range(32):
        register_interface_for_device(f"npu:{i}", NpuInterface)


def _find_spec_without_finder(finder, fullname):
    """Delegate to the remaining meta-path finders without bypassing them."""
    try:
        index = sys.meta_path.index(finder)
    except ValueError:
        return importlib.util.find_spec(fullname)

    sys.meta_path.pop(index)
    try:
        return importlib.util.find_spec(fullname)
    finally:
        sys.meta_path.insert(min(index, len(sys.meta_path)), finder)


class _DynamoPostImportLoader(importlib.abc.Loader):
    def __init__(self, loader, finder):
        self._loader = loader
        self._finder = finder

    def create_module(self, spec):
        create_module = getattr(self._loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module):
        self._loader.exec_module(module)
        _lazy_dynamo_setup()
        if self._finder in sys.meta_path:
            sys.meta_path.remove(self._finder)


class _DynamoPostImportFinder(importlib.abc.MetaPathFinder):
    _target = "torch._dynamo"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._target:
            return None
        spec = _find_spec_without_finder(self, fullname)
        if spec is not None and spec.loader is not None:
            spec.loader = _DynamoPostImportLoader(spec.loader, self)
        return spec


def _install_dynamo_post_import_trigger():
    """Set up NPU integration whenever Dynamo is first imported."""
    if "torch._dynamo" in sys.modules:
        _lazy_dynamo_setup()
        return
    if not any(isinstance(finder, _DynamoPostImportFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _DynamoPostImportFinder())


@run_once
def add_dynamo_methods_init():
    steps = (
        ("device_interface", _dynamo_register_interface_for_device),
        ("skip_function_variable", patch_SkipFunctionVariable),
        ("tensor_variable", patch_TensorVariable_call_method),
        ("user_defined_class_variable", patch_user_defined_class_variable),
        ("record_stream", patch_record_stream),
        ("event_variable", patch_event_variable_python_type),
        ("npu_stream_context", patch_npu_stream_context),
        ("builtin_variable", patch_builtin_variable),
    )
    for name, setup in steps:
        _run_dynamo_setup_step(name, setup)



@functools.lru_cache(None)
def has_triton() -> bool:
    from torch.utils._triton import has_triton_package

    if not has_triton_package():
        return False

    from torch._dynamo.device_interface import get_interface_for_device

    def cuda_extra_check(device_interface):
        return True

    def cpu_extra_check(device_interface):
        import triton.backends

        return "cpu" in triton.backends.backends

    def _return_true(device_interface):
        return True

    triton_supported_devices = {
        "cuda": cuda_extra_check,
        "xpu": _return_true,
        "cpu": cpu_extra_check,
        "npu": _return_true,
    }

    def is_device_compatible_with_triton():
        _dynamo_register_interface_for_device()
        for device, extra_check in triton_supported_devices.items():
            device_interface = get_interface_for_device(device)
            if device_interface.is_available() and extra_check(device_interface):
                return True
        return False

    return is_device_compatible_with_triton()


def patch_has_triton():
    from torch.utils import _triton

    _triton.has_triton = has_triton


@run_once
def _inject_inductor_npu_backend_config():
    """Inject NPU entries into torch._inductor.config on first use."""
    torch._inductor.config.get_config_copy()


@run_once
def _lazy_dynamo_setup():
    """Initialize the Dynamo integration on the first graph-capture operation."""
    add_dynamo_methods_init()

    from torch_npu.dynamo import _register_backends
    _run_dynamo_setup_step("backends", _register_backends)

    from torch_npu.dynamo.trace_rule import _patch_npu_trace_rules
    _run_dynamo_setup_step("trace_rules", _patch_npu_trace_rules)

    _run_dynamo_setup_step("dynamo_optimize", patch_dynamo_optimize)


@run_once
def _lazy_inductor_setup():
    """Initialize NPU Inductor support only for an Inductor-based backend."""
    register_inductor_npu()

    from torch_npu.utils._graph_tree import _apply_npugraph_tree_methods
    _apply_npugraph_tree_methods()

    _inject_inductor_npu_backend_config()


@run_once
def install_npugraph_mark_step_trigger():
    """Expose the public NPUGraph step API without importing compiler internals."""
    def npugraph_mark_step_begin():
        from torch_npu.npu._graph_tree_state import mark_step_begin
        return mark_step_begin()

    torch.compiler.npugraph_mark_step_begin = npugraph_mark_step_begin


def add_dynamo_methods():
    patch_has_triton()

    from torch_npu.dynamo import _install_lazy_torchair

    _install_lazy_torchair()
    _install_dynamo_post_import_trigger()
    if "npugraph_ex" not in sys.modules:
        from torch_npu.dynamo import _LazyNpuGraphEx
        sys.modules["npugraph_ex"] = _LazyNpuGraphEx("npugraph_ex")
    patch_inductor_wrapper()
    install_npugraph_mark_step_trigger()


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
            Maximum allowed size for this dimension (inclusive). Default: 1024.
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
                        warnings.warn("For BATCHSIZE, only the first element of 'dimensions' is used")
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

    def get_shape_safe(self, item):
        """递归获取 shape 的辅助函数"""
        if isinstance(item, torch.Tensor):
            return list(item.shape)
        elif isinstance(item, (list, tuple)):
            # 如果是列表，递归处理内部元素，并标注这是个容器
            return [self.get_shape_safe(i) for i in item]
        else:
            return type(item)


    def transform_hook(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Tuple[List[Tuple], List[Dict]]:
        # 获取 logger
        logger = logging.getLogger(__name__)
        # 预处理阶段优化：统一使用预定义函数或默认逻辑
        if self.transform_pre_fn:
            inputs = self.transform_pre_fn(*args, **kwargs)
        else:
            inputs, indices, leaves, spec = self._process_inputs(args, kwargs)

        # 提取转换前的形状 (inputs 通常是 Tensor 列表)
        if logger.isEnabledFor(logging.INFO):
            pre_shapes = [self.get_shape_safe(t) for t in inputs]
            logger.info("[Transform] Starting. Input tensors: %s, Shapes: %s", len(inputs), pre_shapes)

        # 执行核心转换操作
        trans_outputs = self.transform(tensors=inputs)

        # 提取转换后的形状
        if logger.isEnabledFor(logging.INFO):
            post_shapes = [self.get_shape_safe(t) for t in trans_outputs]
            logger.info("> Post-transform content: %s", post_shapes)

        # 后处理阶段优化：避免嵌套循环
        if self.transform_post_fn:
            outputs = self.transform_post_fn(trans_outputs)
        else:
            outputs = self._recover_inputs(trans_outputs, indices, leaves, spec)

        if not outputs:
            logger.error("CRITICAL: _recover_inputs returned NULL")

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


def _patch_dynamo_context():
    import inspect
    from torch._dynamo.eval_frame import _TorchDynamoContext
    from torch._dynamo.types import DynamoCallback
    from torch._dynamo.convert_frame import CatchErrorsWrapper, ConvertFrame
    from torch._dynamo.repro.after_dynamo import WrapBackendDebug
    src_call = _TorchDynamoContext.__call__
    src_init = _TorchDynamoContext.__init__

    def is_enable_shape_handling(callback: DynamoCallback, compiler_config=None):
        """
        The shape handling feature is only available when enable_shape_handling is True and the backend is inductor
        """
        if compiler_config is None or not compiler_config.get("enable_shape_handling", False):
            return False

        if callback is None or not isinstance(callback, CatchErrorsWrapper):
            return False

        convert_frame = getattr(callback, "_torchdynamo_orig_backend", None)
        if not isinstance(convert_frame, ConvertFrame):
            return False

        backend_debug = getattr(convert_frame, "_torchdynamo_orig_backend", None)
        if not isinstance(backend_debug, WrapBackendDebug):
            return False

        return getattr(backend_debug, "_compiler_name", None) == "inductor"

    def nothing():
        pass

    def new_init(self, callback: DynamoCallback, *args, **kwargs) -> None:
        src_init(self, callback, *args, **kwargs)
        compiler_config = kwargs.get("compiler_config")
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


def _patch_shape_handling():
    if getattr(_patch_shape_handling, "_is_patched", False):
        return
    _patch_dynamo_context()
    _patch_shape_handling._is_patched = True
