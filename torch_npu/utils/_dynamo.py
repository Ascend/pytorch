import inspect
import itertools
import operator
from typing import Dict, List

import torch
from torch._dynamo.utils import tensortype_to_dtype
from torch._dynamo.utils import get_fake_value
from torch._dynamo.exc import unimplemented
from torch._dynamo.variables.torch import TorchVariable
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.ctx_manager import AutocastModeVariable, CUDAStreamContextVariable, CUDAStreamVariable
from torch._dynamo.variables.user_defined import UserDefinedClassVariable
from torch._dynamo.variables.misc import SkipFilesVariable
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.tensor import TensorVariable
from torch._dynamo.variables.lists import TupleVariable
from torch._C import DispatchKey
from torch._prims.rng_prims import run_and_save_rng_state, run_with_rng_state
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._functorch import partitioners
from torch._dynamo import optimize
from torch import _TorchCompileWrapper, _TorchCompileInductorWrapper
import torch_npu
from torch_npu.dynamo import _get_global_npu_backend


def get_device_npu(args, kwargs):
    if kwargs.get("device"):
        device = kwargs.get("device")
        if isinstance(device, str):
            device = torch.device(device)
        return device.type

    devices = {arg.device.type for arg in args if isinstance(arg, torch.Tensor)}
    if any(dev == "cuda" for dev in devices):
        return "cuda"
    elif any(dev == "npu" for dev in devices):
        return "npu"
    elif any(dev == "cpu" for dev in devices):
        return "cpu"
    return None


def patch_higher_order_ops():
    impl_backend_raw = run_and_save_rng_state.python_key_mode_table[FakeTensorMode]
    run_and_save_rng_state.python_key_mode_table.pop(FakeTensorMode)

    @run_and_save_rng_state.py_impl(DispatchKey.PrivateUse1)
    def impl_npu(op, *args, **kwargs):
        # [seed, offset]
        return torch.empty(2, dtype=torch.int64, device='npu'), op(*args, **kwargs)

    @run_and_save_rng_state.py_impl(FakeTensorMode)
    def impl_fake_tensor_mode(op, *args, **kwargs):
        # Check device to call the right impl
        device = get_device_npu(args, kwargs)
        if device == "npu":
            return impl_npu(op, *args, **kwargs)
        return impl_backend_raw(op, *args, **kwargs)

    @run_with_rng_state.py_impl(DispatchKey.AutogradPrivateUse1)
    def impl_autograd_privateuse1(rng_state, op, *args, **kwargs):
        out = op(*args, **kwargs)
        return out


def patch_functionalize_rng_ops():
    def functionalize_rng_ops_new(joint_module, fw_module, bw_module, num_sym_nodes):
        # Unique id to generate name
        uid = itertools.count()

        def get_rng_ops(gmod):
            random_nodes = {}
            for node in gmod.graph.nodes:
                if (
                    node.op == "call_function"
                    and hasattr(node.target, "tags")
                    and torch.Tag.nondeterministic_seeded in node.target.tags
                ):
                    random_nodes[node.name] = node
            return random_nodes

        def get_device(node):
            """
            Check the example value of the node outputs to find the device type.
            """
            if "val" not in node.meta:
                return None

            candidates = node.meta["val"]
            if not isinstance(candidates, tuple):
                candidates = (candidates,)

            for candidate in candidates:
                if isinstance(candidate, torch.Tensor):
                    if candidate.device.type == "cuda":
                        return "cuda"
                    if candidate.device.type == "npu":
                        return "npu"

            return "cpu"

        def get_sample_rng_state(device):
            if device == "cuda":
                return torch.cuda.get_rng_state()
            if device == "npu":
                # [seed, offset]
                return torch.empty(2, dtype=torch.int64, device='npu')
            return torch.get_rng_state()

        # Step 1 - Construct a mapping of rng node between the fwd and its counterpart in bwd.
        joint_graph_rng_ops = get_rng_ops(joint_module)
        fw_graph_rng_ops = get_rng_ops(fw_module)
        bw_graph_rng_ops = get_rng_ops(bw_module)
        recomputable_rng_ops_map = dict()
        for node in joint_module.graph.nodes:
            if (
                partitioners.must_recompute(node)
                and hasattr(node.target, "tags")
                and torch.Tag.nondeterministic_seeded in node.target.tags
            ):
                base_node = joint_graph_rng_ops[node.name]
                fw_node = fw_graph_rng_ops[node.name]
                bw_node = bw_graph_rng_ops[node.name]
                recomputable_rng_ops_map[base_node] = {"fwd": fw_node, "bwd": bw_node}

        run_and_save_rng = torch._prims.rng_prims.run_and_save_rng_state
        run_with_rng = torch._prims.rng_prims.run_with_rng_state

        for node in bw_module.graph.nodes:
            if node.op == "placeholder" and "tangent" in node.name:
                bw_tangent_start_node = node
                break


        fw_rng_state_outputs = []
        for base_node, node_pair in recomputable_rng_ops_map.items():
            # Step 2 - Modify the fwd pass such that
            fw_node = node_pair["fwd"]
            bw_node = node_pair["bwd"]
            fw_graph = fw_module.graph
            with fw_graph.inserting_before(fw_node):
                functional_fw_node = fw_graph.create_node(
                    "call_function",
                    run_and_save_rng,
                    args=(fw_node.target, *fw_node.args),
                    kwargs=fw_node.kwargs
                )
                state = fw_graph.create_node("call_function", operator.getitem, args=(functional_fw_node, 0), kwargs={})
                rng_output = fw_graph.create_node("call_function", operator.getitem, args=(functional_fw_node, 1,), kwargs={})
                fw_node.replace_all_uses_with(rng_output)
                fw_graph.erase_node(fw_node)
                fw_rng_state_outputs.append(state)


            # Step 3 - Modify the bwd pass such that
            bw_graph = bw_module.graph
            with bw_graph.inserting_before(bw_tangent_start_node):
                state_name = f"rng_state_output_{next(uid)}"
                bw_rng_state_node = bw_graph.placeholder(state_name)
                bw_rng_state_node.meta["val"] = get_sample_rng_state(get_device(fw_node))

            with bw_graph.inserting_before(bw_node):
                rng_output = bw_graph.create_node(
                    "call_function",
                    run_with_rng,
                    args=(bw_rng_state_node, bw_node.target, *bw_node.args),
                    kwargs=bw_node.kwargs
                )

                bw_node.replace_all_uses_with(rng_output)
                bw_graph.erase_node(bw_node)


        # Add the rng states in the output of the fwd graph. AOT Autograd assumes
        # that symints are at the end of forward graph outputs. So, insert the new
        # rng states accordingly.
        fw_output_node = [node for node in fw_module.graph.nodes if node.op == "output"][0]
        fw_outputs = fw_output_node.args[0]
        sym_node_start_idx = len(fw_outputs) - num_sym_nodes
        outputs = fw_outputs[:sym_node_start_idx] + fw_rng_state_outputs + fw_outputs[sym_node_start_idx:]
        fw_module.graph.output(outputs)
        fw_module.graph.erase_node(fw_output_node)
        fw_module.recompile()
        bw_module.recompile()
        return fw_module, bw_module

    partitioners.functionalize_rng_ops = functionalize_rng_ops_new


def wrap_fx_proxy_cls_stream(
    target_cls, tx, proxy, example_value=None, ignore_subclass=False, **options
):
    example_value = get_fake_value(proxy.node, tx)
    proxy.node.meta['example_value'] = example_value
    return NPUStreamVariable(proxy, example_value, **options)


class NPUStreamVariable(CUDAStreamVariable):
    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        unimplemented("npu stream")


class NPUStreamContextVariable(CUDAStreamContextVariable):
    @staticmethod
    def create(tx, target_value, **kwargs):
        current_stream = wrap_fx_proxy_cls_stream(
            NPUStreamVariable,
            tx,
            tx.output.create_proxy(
                "call_function",
                torch_npu.npu.current_stream,
                (None,),
                {},
            ),
        )
        return NPUStreamContextVariable(
            target_values=[target_value],
            initial_values=[current_stream],
            **kwargs,
        )

    def enter(self, tx):
        # NPU stream generated inside of traced function
        if self.target_values[0].as_proxy() is not None:
            tx.output.create_proxy(
                "call_function",
                torch_npu.npu.set_stream,
                (self.target_values[0].as_proxy(),),
                {},
            )
        # NPU stream passed from outside of traced function
        else:
            stream = self.target_values[0].value
            tx.output.create_proxy(
                "call_function",
                torch._C._npu_setStream,
                (stream.stream_id, stream.device_index, stream.device_type),
                {},
            )
        torch_npu.npu.set_stream(self.target_values[0].value)

    def exit(self, tx, *args):
        tx.output.create_proxy(
            "call_function",
            torch_npu.npu.set_stream,
            (self.initial_values[0].as_proxy(),),
            {},
        )
        torch_npu.npu.set_stream(self.initial_values[0].value)

    def module_name(self):
        return "torch_npu.npu"


class NPUTorchVariable(TorchVariable):
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        if self.value in [
            torch.npu.amp.autocast,
            torch_npu.npu.amp.autocast,
            torch.npu.amp.autocast_mode.autocast,
            torch_npu.npu.amp.autocast_mode.autocast,
        ]:
            return NPUAutocastModeVariable.create(self.value, args, kwargs)
        elif self.value in [
            torch.npu.stream,
            torch_npu.npu.stream,
        ]:
            return NPUStreamContextVariable.create(tx, args[0], **options)
        elif self.value in [
            torch.npu.Stream,
            torch_npu.npu.Stream,
            torch.npu.streams.Stream,
            torch_npu.npu.streams.Stream,
        ]:
            return wrap_fx_proxy_cls_stream(
                NPUStreamVariable,
                tx,
                tx.output.create_proxy(
                    "call_function",
                    torch_npu.npu.streams.Stream,
                    (),
                    {},
                ),
                **options
            )
        else:
            return super().call_function(tx, args, kwargs)


class NPUAutocastModeVariable(AutocastModeVariable):
    @staticmethod
    def create(func, args, kwargs):
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

        var = AutocastModeVariable(
            target_values, initial_values=None, **kwargs)
        return var


def UserDefinedClassVariable__new__(cls, value, **kwargs):
    if value in [
        torch.npu.amp.autocast,
        torch_npu.npu.amp.autocast,
        torch.npu.amp.autocast_mode.autocast,
        torch_npu.npu.amp.autocast_mode.autocast,
        torch.npu.Stream,
        torch_npu.npu.Stream,
        torch.npu.streams.Stream,
        torch_npu.npu.streams.Stream,
        torch.device,
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
        return NPUTorchVariable(value, **kwargs)
    return cls.__new__raw(cls)


def SkipFilesVariable__new__(cls, value, **kwargs):
    if value in [
        torch.npu.stream,
        torch_npu.npu.stream,
    ]:
        return NPUTorchVariable(value, **kwargs)
    return cls.__new__raw(cls)


def TensorVariable_call_method(self, tx, name, args, kwargs):
    if (
        name == 'type'
        and self.dtype is not None
        and len(args) == 0
        and isinstance(self.device, torch.device)
        and self.device.type == 'npu'
    ):
        tensortype = next(
            k for k, v in tensortype_to_dtype.items() if self.dtype in v)
        constant_result = ConstantVariable(f"torch.npu.{tensortype.__name__}")

        if len(args) == 1:
            return constant_result.getitem_const(args[0])
        elif args:
            return TupleVariable([constant_result.getitem_const(a) for a in args])
        return constant_result
    else:
        return TensorVariable.call_method_raw(self, tx, name, args, kwargs)


def patch_dynamo_optimize():
    src_optimize = optimize

    def npu_optimize(*args, **kwargs):
        backend = None
        if 'backend' in kwargs.keys():
            backend = kwargs['backend']
        elif len(args) == 1:
            backend = args[0]

        backend_name = None
        if isinstance(backend, str):
            backend_name = backend
        elif isinstance(backend, _TorchCompileWrapper):
            backend_name = backend.compiler_name

        if isinstance(backend, _TorchCompileInductorWrapper) or \
            backend_name == 'inductor':
            patch_inductor_wrapper()

        if backend_name == 'npu':
            # Init torchair ahead of running model.
            _get_global_npu_backend()
        return src_optimize(*args, **kwargs)
    torch._dynamo.optimize = npu_optimize


def patch_inductor_wrapper():
    from torch._inductor.graph import GraphLowering
    from torch._inductor.codegen.common import get_wrapper_codegen_for_device
    from torch_npu.utils._error_code import ErrCode, pta_error

    if hasattr(GraphLowering, 'src_init_wrapper_code'):
        return

    src_init_wrapper_code = GraphLowering.init_wrapper_code

    def _check_wrapper_exist(self):
        device_types = self.device_types.copy()
        for buffer in self.buffers:
            device_types.add(buffer.get_device().type)
        device_types.discard("cpu")
        only_cpu = len(device_types) == 0
        device_type = "cpu" if only_cpu else device_types.pop()
        wrapper_code_gen_cls = get_wrapper_codegen_for_device(device_type)
        if wrapper_code_gen_cls is None:
            raise AssertionError(f"Device {device_type} not supported" + pta_error(ErrCode.NOT_SUPPORT))

    def new_init_wrapper_code(self):
        _check_wrapper_exist(self)
        return src_init_wrapper_code(self)

    GraphLowering.src_init_wrapper_code = src_init_wrapper_code
    GraphLowering.init_wrapper_code = new_init_wrapper_code


def add_dynamo_methods():
    UserDefinedClassVariable.__new__raw = UserDefinedClassVariable.__new__
    UserDefinedClassVariable.__new__ = UserDefinedClassVariable__new__
    SkipFilesVariable.__new__raw = SkipFilesVariable.__new__
    SkipFilesVariable.__new__ = SkipFilesVariable__new__
    TensorVariable.call_method_raw = TensorVariable.call_method
    TensorVariable.call_method = TensorVariable_call_method
    patch_higher_order_ops()
    patch_functionalize_rng_ops()
    patch_dynamo_optimize()
