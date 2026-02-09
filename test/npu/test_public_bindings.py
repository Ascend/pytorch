# Owner(s): ["module: autograd"]
import importlib
from typing import Callable
import inspect
import json
import os
import unittest
import warnings
from importlib import import_module
from itertools import chain
from pathlib import Path

import pkgutil
import torch
from torch.testing._internal.common_utils import TestCase, run_tests, IS_JETSON, IS_WINDOWS, IS_MACOS, skipIfTorchDynamo
from torch._utils_internal import get_file_path_2
import torch_npu
import torch_npu.testing


temp_filter = {
    "torch_npu.npu.amp.autocast_mode.Any",
    "torch_npu.npu.amp.autocast_mode.ErrCode",
    "torch_npu.npu.amp.autocast_mode.pta_error",
    "torch_npu.npu.amp.grad_scaler.BaseGradScaler",
    "torch_npu.npu.amp.grad_scaler.ErrCode",
    "torch_npu.npu.amp.grad_scaler.List",
    "torch_npu.npu.amp.grad_scaler.OptState",
    "torch_npu.npu.amp.grad_scaler.amp_definitely_not_available",
    "torch_npu.npu.amp.grad_scaler.defaultdict",
    "torch_npu.npu.amp.grad_scaler.pta_error",
    "torch_npu.profiler.analysis.prof_bean.node_info_bean.List",
    "torch_npu.profiler.analysis.prof_bean.node_info_bean.convert_us2ns",
    "torch_npu.testing.common_distributed.Any",
    "torch_npu.testing.common_distributed.Dict",
    "torch_npu.testing.common_distributed.Tuple",
    "torch_npu.testing.common_distributed.contextmanager",
    "torch_npu.testing.common_distributed.namedtuple",
    "torch_npu.testing.common_distributed.wraps",
    "torch_npu.testing.common_methods_invocations.List",
    "torch_npu.testing.common_methods_invocations.make_tensor",
    "torch_npu.testing.common_methods_invocations.partial",
    "torch_npu.testing.common_methods_invocations.sample_inputs_normal_common",
    "torch_npu.testing.common_methods_invocations.wraps",
    "torch_npu.testing.common_utils.List",
    "torch_npu.testing.common_utils.PathManager",
    "torch_npu.testing.common_utils.contextmanager",
    "torch_npu.testing.common_utils.product",
    "torch_npu.testing.common_utils.wraps",
    "torch_npu.testing.decorator.partialmethod",
    "torch_npu.testing.decorator.wraps",
    "torch_npu.testing.testcase.Number",
    "torch_npu.testing.testcase.OrderedDict",
    "torch_npu.testing.testcase.Sequence",
    "torch_npu.testing.testcase.TestResult",
    "torch_npu.testing.testcase.contextmanager",
    "torch_npu.testing.testcase.is_iterable",
    "torch_npu.testing.testcase.iter_indices",
    "torch_npu.testing.testcase.set_npu_device",
    "torch_npu.testing.testcase.strclass",
    "torch_npu.utils.collect_env.namedtuple",
    "torch_npu.utils.profiler.ErrCode",
    "torch_npu.utils.profiler.Optional",
    "torch_npu.utils.profiler.prof_error",
    "torch_npu.utils.flops_count.FlopsCounter",
    "torch_npu.npu_add_rms_norm",
    "torch_npu.npu_deep_norm",
    "torch_npu.npu_fast_gelu",
    "torch_npu.npu_fused_attention_layernorm_qkv_fwd",
    "torch_npu.npu_fused_attention_score_fwd",
    "torch_npu.npu_group_norm_silu",
    "torch_npu.npu_lstm_cell",
    "torch_npu.npu_masked_softmax_with_rel_pos_bias",
    "torch_npu.npu_moe_compute_expert_tokens",
    "torch_npu.npu_moe_finalize_routing",
    "torch_npu.npu_quantize",
    "torch_npu.npu_stride_copy",
    "torch_npu.fast_gelu",
    "torch_npu.npu_anchor_response_flags",
    "torch_npu.npu_anti_quant",
    "torch_npu.npu_batch_nms",
    "torch_npu.npu_bmmV2",
    "torch_npu.npu_bounding_box_decode",
    "torch_npu.npu_bounding_box_encode",
    "torch_npu.npu_broadcast",
    "torch_npu.npu_ciou",
    "torch_npu.npu_confusion_transpose",
    "torch_npu.npu_conv2d",
    "torch_npu.npu_conv3d",
    "torch_npu.npu_conv_transpose2d",
    "torch_npu.npu_convolution",
    "torch_npu.npu_convolution_transpose",
    "torch_npu.npu_deformable_conv2d",
    "torch_npu.npu_diou",
    "torch_npu.npu_dropout_with_add_softmax",
    "torch_npu.npu_dtype_cast",
    "torch_npu.npu_format_cast",
    "torch_npu.npu_fused_attention_score",
    "torch_npu.npu_giou",
    "torch_npu.npu_grid_assign_positive",
    "torch_npu.npu_gru",
    "torch_npu.npu_ifmr",
    "torch_npu.npu_incre_flash_attention",
    "torch_npu.npu_indexing",
    "torch_npu.npu_iou",
    "torch_npu.npu_layer_norm_eval",
    "torch_npu.npu_linear",
    "torch_npu.npu_lstm",
    "torch_npu.npu_masked_fill_range",
    "torch_npu.npu_max",
    "torch_npu.npu_min",
    "torch_npu.npu_mish",
    "torch_npu.npu_mm_all_reduce_base",
    "torch_npu.npu_multi_head_attention",
    "torch_npu.npu_nms_v4",
    "torch_npu.npu_nms_with_mask",
    "torch_npu.npu_normalize_batch",
    "torch_npu.npu_one_hot",
    "torch_npu.npu_pad",
    "torch_npu.npu_prompt_flash_attention",
    "torch_npu.npu_ps_roi_pooling",
    "torch_npu.npu_ptiou",
    "torch_npu.npu_reshape",
    "torch_npu.npu_roi_align",
    "torch_npu.npu_rotary_mul",
    "torch_npu.npu_rotated_box_decode",
    "torch_npu.npu_rotated_box_encode",
    "torch_npu.npu_rotated_iou",
    "torch_npu.npu_rotated_overlaps",
    "torch_npu.npu_scaled_masked_softmax",
    "torch_npu.npu_scatter",
    "torch_npu.npu_scatter_nd_update",
    "torch_npu.npu_sign_bits_pack",
    "torch_npu.npu_sign_bits_unpack",
    "torch_npu.npu_silu",
    "torch_npu.npu_slice",
    "torch_npu.npu_softmax_cross_entropy_with_logits",
    "torch_npu.one_",
}


def _find_all_importables(pkg):
    """Find all importables in the project.

    Return them in order.
    """
    return sorted(
        set(
            chain.from_iterable(
                _discover_path_importables(Path(p), pkg.__name__)
                for p in pkg.__path__
            ),
        ),
    )


def _discover_path_importables(pkg_pth, pkg_name):
    """Yield all importables under a given path and package.

    This is like pkgutil.walk_packages, but does *not* skip over namespace
    packages. Taken from init-py-required-for-pkgutil-walk-packages-in-python3
    """
    for dir_path, _d, file_names in os.walk(pkg_pth):
        pkg_dir_path = Path(dir_path)

        if pkg_dir_path.parts[-1] == '__pycache__':
            continue

        if all(Path(_).suffix != '.py' for _ in file_names):
            continue

        rel_pt = pkg_dir_path.relative_to(pkg_pth)
        pkg_pref = '.'.join((pkg_name, ) + rel_pt.parts)
        yield from (
            pkg_path
            for _, pkg_path, _ in pkgutil.walk_packages(
                (str(pkg_dir_path), ), prefix=f'{pkg_pref}.',
            )
        )


class TestPublicBindings(TestCase):
    def test_no_new_bindings(self):
        """
        This test aims to stop the introduction of new JIT bindings into torch._C
        whose names do not start with _. Such bindings are made available as
        torch.XXX, which may not be desirable.

        If your change causes this test to fail, add your new binding to a relevant
        submodule of torch._C, such as torch._C._jit (or other relevant submodule of
        torch._C). If your binding really needs to be available as torch.XXX, add it
        to torch._C and add it to the allowlist below.

        If you have removed a binding, remove it from the allowlist as well.
        """
        # This allowlist contains every binding in torch._C that is copied into torch at
        # the time of writing. It was generated with
        #
        #   {elem for elem in dir(torch._C) if not elem.startswith("_")}
        #
        torch_C_allowlist_superset = {
            "AggregationType",
            "AliasDb",
            "AnyType",
            "Argument",
            "ArgumentSpec",
            "AwaitType",
            "autocast_decrement_nesting",
            "autocast_increment_nesting",
            "AVG",
            "BenchmarkConfig",
            "BenchmarkExecutionStats",
            "Block",
            "BoolType",
            "BufferDict",
            "StorageBase",
            "CallStack",
            "Capsule",
            "ClassType",
            "clear_autocast_cache",
            "Code",
            "CompilationUnit",
            "CompleteArgumentSpec",
            "ComplexType",
            "ConcreteModuleType",
            "ConcreteModuleTypeBuilder",
            "cpp",
            "CudaBFloat16TensorBase",
            "CudaBoolTensorBase",
            "CudaByteTensorBase",
            "CudaCharTensorBase",
            "CudaComplexDoubleTensorBase",
            "CudaComplexFloatTensorBase",
            "CudaDoubleTensorBase",
            "CudaFloatTensorBase",
            "CudaHalfTensorBase",
            "CudaIntTensorBase",
            "CudaLongTensorBase",
            "CudaShortTensorBase",
            "DeepCopyMemoTable",
            "default_generator",
            "DeserializationStorageContext",
            "device",
            "DeviceObjType",
            "DictType",
            "DisableTorchFunction",
            "DisableTorchFunctionSubclass",
            "DispatchKey",
            "DispatchKeySet",
            "dtype",
            "EnumType",
            "ErrorReport",
            "ExcludeDispatchKeyGuard",
            "ExecutionPlan",
            "FatalError",
            "FileCheck",
            "finfo",
            "FloatType",
            "fork",
            "FunctionSchema",
            "Future",
            "FutureType",
            "Generator",
            "GeneratorType",
            "get_autocast_cpu_dtype",
            "get_autocast_dtype",
            "get_autocast_ipu_dtype",
            "get_default_dtype",
            "get_num_interop_threads",
            "get_num_threads",
            "Gradient",
            "Graph",
            "GraphExecutorState",
            "has_cuda",
            "has_cudnn",
            "has_lapack",
            "has_mkl",
            "has_mkldnn",
            "has_mps",
            "has_openmp",
            "has_spectral",
            "iinfo",
            "import_ir_module_from_buffer",
            "import_ir_module",
            "InferredType",
            "init_num_threads",
            "InterfaceType",
            "IntType",
            "SymFloatType",
            "SymBoolType",
            "SymIntType",
            "IODescriptor",
            "is_anomaly_enabled",
            "is_anomaly_check_nan_enabled",
            "is_autocast_cache_enabled",
            "is_autocast_cpu_enabled",
            "is_autocast_ipu_enabled",
            "is_autocast_enabled",
            "is_grad_enabled",
            "is_inference_mode_enabled",
            "JITException",
            "layout",
            "ListType",
            "LiteScriptModule",
            "LockingLogger",
            "LoggerBase",
            "memory_format",
            "merge_type_from_type_comment",
            "ModuleDict",
            "Node",
            "NoneType",
            "NoopLogger",
            "NumberType",
            "OperatorInfo",
            "OptionalType",
            "OutOfMemoryError",
            "ParameterDict",
            "parse_ir",
            "parse_schema",
            "parse_type_comment",
            "PyObjectType",
            "PyTorchFileReader",
            "PyTorchFileWriter",
            "qscheme",
            "read_vitals",
            "RRefType",
            "ScriptClass",
            "ScriptClassFunction",
            "ScriptDict",
            "ScriptDictIterator",
            "ScriptDictKeyIterator",
            "ScriptList",
            "ScriptListIterator",
            "ScriptFunction",
            "ScriptMethod",
            "ScriptModule",
            "ScriptModuleSerializer",
            "ScriptObject",
            "ScriptObjectProperty",
            "SerializationStorageContext",
            "set_anomaly_enabled",
            "set_autocast_cache_enabled",
            "set_autocast_cpu_dtype",
            "set_autocast_dtype",
            "set_autocast_ipu_dtype",
            "set_autocast_cpu_enabled",
            "set_autocast_ipu_enabled",
            "set_autocast_enabled",
            "set_flush_denormal",
            "set_num_interop_threads",
            "set_num_threads",
            "set_vital",
            "Size",
            "StaticModule",
            "Stream",
            "StreamObjType",
            "Event",
            "StringType",
            "SUM",
            "SymFloat",
            "SymInt",
            "TensorType",
            "ThroughputBenchmark",
            "TracingState",
            "TupleType",
            "Type",
            "unify_type_list",
            "UnionType",
            "Use",
            "Value",
            'set_autocast_gpu_dtype',
            'get_autocast_gpu_dtype',
            "vitals_enabled",
            "wait",
            "Tag",
            "set_autocast_xla_enabled",
            "set_autocast_xla_dtype",
            "get_autocast_xla_dtype",
            "is_autocast_xla_enabled",
        }
        torch_C_bindings = {elem for elem in dir(torch._C) if not elem.startswith("_")}

        # torch.TensorBase is explicitly removed in torch/__init__.py, so included here (#109940)
        explicitly_removed_torch_C_bindings = {
            "TensorBase",
        }
        torch_C_bindings = torch_C_bindings - explicitly_removed_torch_C_bindings

        # Check that the torch._C bindings are all in the allowlist. Since
        # bindings can change based on how PyTorch was compiled (e.g. with/without
        # CUDA), the two may not be an exact match but the bindings should be
        # a subset of the allowlist.
        difference = torch_C_bindings.difference(torch_C_allowlist_superset)
        msg = f"torch._C had bindings that are not present in the allowlist:\n{difference}"
        self.assertTrue(torch_C_bindings.issubset(torch_C_allowlist_superset), msg)

    @staticmethod
    def _is_mod_public(modname):
        split_strs = modname.split('.')
        for elem in split_strs:
            if elem.startswith("_"):
                return False
        return True

    @unittest.skipIf(IS_WINDOWS or IS_MACOS, "Inductor/Distributed modules hard fail on windows and macos")
    @skipIfTorchDynamo("Broken and not relevant for now")
    def test_modules_can_be_imported(self):
        failures = []
        for modname in _find_all_importables(torch):
            try:
                if "__main__" in modname:
                    continue
                import_module(modname)
            except Exception as e:
                # Some current failures are not ImportError
                failures.append((modname, type(e)))

        for modname in _find_all_importables(torch_npu):
            try:
                if "__main__" in modname or \
                        modname in ["torch_npu.dynamo.torchair.core._backend",
                                    "torch_npu.dynamo.torchair.core._torchair"]:
                    continue
                import_module(modname)
            except Exception as e:
                # Some current failures are not ImportError
                failures.append((modname, type(e)))

        # It is ok to add new entries here but please be careful that these modules
        # do not get imported by public code.
        private_allowlist = {
            "torch._inductor.codegen.cuda.cuda_kernel",
            "torch.onnx._internal.fx._pass",
            "torch.onnx._internal.fx.analysis",
            "torch.onnx._internal.fx.analysis.unsupported_nodes",
            "torch.onnx._internal.fx.decomposition_skip",
            "torch.onnx._internal.fx.diagnostics",
            "torch.onnx._internal.fx.fx_onnx_interpreter",
            "torch.onnx._internal.fx.fx_symbolic_graph_extractor",
            "torch.onnx._internal.fx.onnxfunction_dispatcher",
            "torch.onnx._internal.fx.op_validation",
            "torch.onnx._internal.fx.passes",
            "torch.onnx._internal.fx.passes._utils",
            "torch.onnx._internal.fx.passes.decomp",
            "torch.onnx._internal.fx.passes.functionalization",
            "torch.onnx._internal.fx.passes.modularization",
            "torch.onnx._internal.fx.passes.readability",
            "torch.onnx._internal.fx.passes.type_promotion",
            "torch.onnx._internal.fx.passes.virtualization",
            "torch.onnx._internal.fx.type_utils",
            "torch.testing._internal.common_distributed",
            "torch.testing._internal.common_fsdp",
            "torch.testing._internal.dist_utils",
            "torch.testing._internal.distributed.common_state_dict",
            "torch.testing._internal.distributed._shard.sharded_tensor",
            "torch.testing._internal.distributed._shard.test_common",
            "torch.testing._internal.distributed._tensor.common_dtensor",
            "torch.testing._internal.distributed.ddp_under_dist_autograd_test",
            "torch.testing._internal.distributed.distributed_test",
            "torch.testing._internal.distributed.distributed_utils",
            "torch.testing._internal.distributed.fake_pg",
            "torch.testing._internal.distributed.multi_threaded_pg",
            "torch.testing._internal.distributed.nn.api.remote_module_test",
            "torch.testing._internal.distributed.rpc.dist_autograd_test",
            "torch.testing._internal.distributed.rpc.dist_optimizer_test",
            "torch.testing._internal.distributed.rpc.examples.parameter_server_test",
            "torch.testing._internal.distributed.rpc.examples.reinforcement_learning_rpc_test",
            "torch.testing._internal.distributed.rpc.faulty_agent_rpc_test",
            "torch.testing._internal.distributed.rpc.faulty_rpc_agent_test_fixture",
            "torch.testing._internal.distributed.rpc.jit.dist_autograd_test",
            "torch.testing._internal.distributed.rpc.jit.rpc_test",
            "torch.testing._internal.distributed.rpc.jit.rpc_test_faulty",
            "torch.testing._internal.distributed.rpc.rpc_agent_test_fixture",
            "torch.testing._internal.distributed.rpc.rpc_test",
            "torch.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture",
            "torch.testing._internal.distributed.rpc_utils",
            "torch.utils.tensorboard._caffe2_graph",
            "torch._inductor.codegen.cuda.cuda_template",
            "torch._inductor.codegen.cuda.gemm_template",
            "torch._inductor.runtime.triton_helpers",
            "torch.ao.pruning._experimental.data_sparsifier.lightning.callbacks.data_sparsity",
            "torch.backends._coreml.preprocess",
            "torch.contrib._tensorboard_vis",
            "torch.distributed._composable",
            "torch.distributed._functional_collectives",
            "torch.distributed._functional_collectives_impl",
            "torch.distributed._shard",
            "torch.distributed._sharded_tensor",
            "torch.distributed._sharding_spec",
            "torch.distributed._spmd.api",
            "torch.distributed._spmd.batch_dim_utils",
            "torch.distributed._spmd.comm_tensor",
            "torch.distributed._spmd.data_parallel",
            "torch.distributed._spmd.distribute",
            "torch.distributed._spmd.experimental_ops",
            "torch.distributed._spmd.parallel_mode",
            "torch.distributed._tensor",
            "torch.distributed._tools.fsdp_ilp",
            "torch.distributed.algorithms._checkpoint.checkpoint_wrapper",
            "torch.distributed.algorithms._optimizer_overlap",
            "torch.distributed.rpc._testing.faulty_agent_backend_registry",
            "torch.distributed.rpc._utils",
            "torch.ao.pruning._experimental.data_sparsifier.benchmarks.dlrm_utils",
            "torch.ao.pruning._experimental.data_sparsifier.benchmarks.evaluate_disk_savings",
            "torch.ao.pruning._experimental.data_sparsifier.benchmarks.evaluate_forward_time",
            "torch.ao.pruning._experimental.data_sparsifier.benchmarks.evaluate_model_metrics",
            "torch.ao.pruning._experimental.data_sparsifier.lightning.tests.test_callbacks",
            "torch.csrc.jit.tensorexpr.scripts.bisect",
            "torch.csrc.lazy.test_mnist",
            "torch.distributed._shard.checkpoint._fsspec_filesystem",
            "torch.distributed._tensor.examples.visualize_sharding_example",
            "torch.distributed.checkpoint._fsspec_filesystem",
            "torch.distributed.examples.memory_tracker_example",
            "torch.testing._internal.distributed.rpc.fb.thrift_rpc_agent_test_fixture",
            "torch.utils._cxx_pytree",
            "torch.utils.tensorboard._convert_np",
            "torch.utils.tensorboard._embedding",
            "torch.utils.tensorboard._onnx_graph",
            "torch.utils.tensorboard._proto_graph",
            "torch.utils.tensorboard._pytorch_graph",
            "torch.utils.tensorboard._utils",
            "torch.distributed._tools.sac_ilp",
            "torch_npu.dynamo.torchair._tf_concrete_graph.fx2tf_converter",
            "torch_npu.dynamo.torchair.core._abi_compat_ge_apis",
            "torch_npu.dynamo.torchair.core._backend",
            "torch_npu.dynamo.torchair.core._torchair",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.prims.slice",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.prims.slice_in_dim",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.aten.bernoulli",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.aten.multinomial",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.aten.native_dropout",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.aten.rand",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.aten.randint",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.aten.randn",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.aten.randperm",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.aten.uniform",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.custom.npu_dynamic_quant",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.custom.fused_infer_attention_score",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.experimental.hcom_allgather",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.experimental.hcom_allreduce",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.experimental.hcom_alltoall",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.experimental.hcom_broadcast",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.experimental.hcom_reducescatter",
            "torch_npu.dynamo.torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce",
            "torch_npu.dynamo.torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.custom.npu_selu_backward",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_ir_by_protoc_3_13_pb2",
            "torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.custom.npu_dequant_bias",
            "torch_npu.utils.collect_hccl_info",
            "torch_npu.op_plugin.meta._meta_registrations",
            "torch_npu._inductor",
            "torch_npu._inductor.codegen",
            "torch_npu._inductor.config",
            "torch_npu._inductor.decomposition",
            "torch_npu._inductor.lowering",
            "torch_npu._inductor.lowering_fx",
            "torch_npu._inductor.npu_choices",
            "torch_npu._inductor.npu_device",
            "torch_npu._inductor.npu_fusion_attention_graph",
            "torch_npu._inductor.npu_triton_helpers",
            "torch_npu._inductor.npu_triton_heuristics",
            "torch_npu._inductor.npu_static_kernel",
            "torch_npu._inductor.runtime",
            "torch_npu._inductor.utils",
            "torch_npu._inductor.codegen._sizevars",
            "torch_npu._inductor.codegen.cpp_wrapper",
            "torch_npu._inductor.codegen.ir",
            "torch_npu._inductor.codegen.ir_fx",
            "torch_npu._inductor.codegen.kernel_analysis",
            "torch_npu._inductor.codegen.npu_kernel_features",
            "torch_npu._inductor.codegen.scheduling",
            "torch_npu._inductor.codegen.split_tiling",
            "torch_npu._inductor.codegen.tile_generator",
            "torch_npu._inductor.codegen.triton",
            "torch_npu._inductor.codegen.triton_utils",
            "torch_npu._inductor.codegen.cpp_utils",
            "torch_npu._inductor.codegen.wrapper",
            "torch_npu._inductor.codecache",
            "torch_npu._inductor.cpp_builder",
            "torch_npu._inductor.fx_passes.joint_graph",
            "torch_npu._inductor.ir",
            "torch_npu._inductor.graph",
            "torch_npu._inductor.lowering_op_list",
            "torch_npu._inductor.shape_handling",
            "torch_npu.op_plugin.atb._atb_meta_registrations",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.build_info",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.cache",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.codecache",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.config",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.triton",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.inductor_patch",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.codegen",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.mlir_compiler",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.npu_decomp",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.npu_inductor_plugin",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.npu_lowering",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.npu_meta",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.npu_patch_deprecated",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.npu_stream",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.torch_mlir_patch",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.utils",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.codegen.cpp_wrapper",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.codegen.mlir",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.codegen.wrapper",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.inductor_patch.fake_tensor",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.inductor_patch.ir",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.inductor_patch.lowering",
            "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.inductor_patch.scheduler",
        }

        # No new entries should be added to this list.
        # All public modules should be importable on all platforms.
        public_allowlist = {
            "torch.distributed.algorithms.ddp_comm_hooks",
            "torch.distributed.algorithms.model_averaging.averagers",
            "torch.distributed.algorithms.model_averaging.hierarchical_model_averager",
            "torch.distributed.algorithms.model_averaging.utils",
            "torch.distributed.checkpoint",
            "torch.distributed.constants",
            "torch.distributed.distributed_c10d",
            "torch.distributed.elastic.agent.server",
            "torch.distributed.elastic.rendezvous",
            "torch.distributed.fsdp",
            "torch.distributed.launch",
            "torch.distributed.launcher",
            "torch.distributed.nn",
            "torch.distributed.nn.api.remote_module",
            "torch.distributed.optim",
            "torch.distributed.optim.optimizer",
            "torch.distributed.rendezvous",
            "torch.distributed.rpc.api",
            "torch.distributed.rpc.backend_registry",
            "torch.distributed.rpc.constants",
            "torch.distributed.rpc.internal",
            "torch.distributed.rpc.options",
            "torch.distributed.rpc.rref_proxy",
            "torch.distributed.elastic.rendezvous.etcd_rendezvous",
            "torch.distributed.elastic.rendezvous.etcd_rendezvous_backend",
            "torch.distributed.elastic.rendezvous.etcd_store",
            "torch.distributed.rpc.server_process_global_profiler",
            "torch.distributed.run",
            "torch.distributed.tensor.parallel",
            "torch.distributed.utils",
            "torch.utils.tensorboard",
            "torch.utils.tensorboard.summary",
            "torch.utils.tensorboard.writer",
            "torch.ao.quantization.experimental.fake_quantize",
            "torch.ao.quantization.experimental.linear",
            "torch.ao.quantization.experimental.observer",
            "torch.ao.quantization.experimental.qconfig",
        }

        errors = []
        for mod, excep_type in failures:
            if mod in public_allowlist:
                continue

            if mod in private_allowlist:
                continue

            errors.append(f"{mod} failed to import with error {excep_type}")
        self.assertEqual("", "\n".join(errors))

    # AttributeError: module 'torch.distributed' has no attribute '_shard'
    @unittest.skipIf(IS_WINDOWS or IS_JETSON or IS_MACOS, "Distributed Attribute Error")
    @skipIfTorchDynamo("Broken and not relevant for now")
    def test_correct_module_names(self):
        '''
        An API is considered public, if  its  `__module__` starts with `torch.`
        and there is no name in `__module__` or the object itself that starts with “_”.
        Each public package should either:
        - (preferred) Define `__all__` and all callables and classes in there must have their
         `__module__` start with the current submodule's path. Things not in `__all__` should
          NOT have their `__module__` start with the current submodule.
        - (for simple python-only modules) Not define `__all__` and all the elements in `dir(submod)` must have their
          `__module__` that start with the current submodule.
        '''
        failure_list = []

        def _get_test_torch_version():
            torch_npu_version = torch_npu.__version__
            version_list = torch_npu_version.split('.')
            if len(version_list) > 2:
                return f'v{version_list[0]}.{version_list[1]}'
            else:
                raise RuntimeError("Invalid torch_npu version.")
        
        try:
            file_abspath = os.path.abspath(__file__)
            air_path = 'third_party/torchair/torchair/tests/st/allowlist_for_publicAPI.json'
            with open(
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_abspath))), air_path)) as json_file_torchair:
                allow_dict_torchair = json.load(json_file_torchair)
                update_allow_dict_torchair = {f"torch_npu.dynamo.{key}": value for key, value in allow_dict_torchair.items()}
        except Exception:
            update_allow_dict_torchair = {}
            warnings.warn("if you are debugging UT file in clone repo, please recursively update the torchair submodule")
        
        try:
            file_abspath = os.path.abspath(__file__)
            op_plugin_path = 'third_party/op-plugin/test/allowlist_for_publicAPI.json'
            with open(
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_abspath))), op_plugin_path)) as json_file_op_plugin:
                version_tag = _get_test_torch_version()
                allow_dict_info = json.load(json_file_op_plugin)
                allow_dict_op_plugin = {}
                if "torch_npu" in allow_dict_info and "all_version" in allow_dict_info["torch_npu"]:
                    allow_dict_op_plugin["torch_npu"] = allow_dict_info["torch_npu"]["all_version"]
                    if version_tag in allow_dict_info["torch_npu"] and allow_dict_info["torch_npu"][version_tag]:
                        allow_dict_op_plugin["torch_npu"].extend(allow_dict_info["torch_npu"][version_tag])

        except Exception as e:
            allow_dict_op_plugin = {}
            warnings.warn(f"{e}")
       
        with open(get_file_path_2(os.path.dirname(os.path.dirname(__file__)),
                                  'allowlist_for_publicAPI.json')) as json_file:
            # no new entries should be added to this allow_dict.
            # New APIs must follow the public API guidelines.
            allow_dict = json.load(json_file)
            # Because we want minimal modifications to the `allowlist_for_publicAPI.json`,
            # we are adding the entries for the migrated modules here from the original
            # locations.
            for modname in allow_dict["being_migrated"]:
                if modname in allow_dict:
                    allow_dict[allow_dict["being_migrated"][modname]] = allow_dict[modname]
        with open(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'deprecated_apis.json')) as json_file:
            deprecated_dict = json.load(json_file)
                    
        if update_allow_dict_torchair:
            allow_dict.update(update_allow_dict_torchair)
        
        if allow_dict_op_plugin and "torch_npu" in allow_dict_op_plugin:
            if "torch_npu" in allow_dict:
                allow_dict["torch_npu"].extend(allow_dict_op_plugin["torch_npu"])
            else:
                allow_dict.update(allow_dict_op_plugin["torch_npu"])

        
        def test_module(modname):
            try:
                if "__main__" in modname or \
                        modname in ["torch_npu.dynamo.torchair.core._backend",
                                    "torch_npu.dynamo.torchair.core._torchair"]:
                    return
                mod = importlib.import_module(modname)
            except Exception:
                # It is ok to ignore here as we have a test above that ensures
                # this should never happen
                return

            if not self._is_mod_public(modname):
                return

            # verifies that each public API has the correct module name and naming semantics
            def check_one_element(elem, modname, mod, *, is_public, is_all):
                obj = getattr(mod, elem)
                if not (isinstance(obj, (Callable, torch.dtype)) or inspect.isclass(obj)):
                    return
                elem_module = getattr(obj, '__module__', None)
                # Only used for nice error message below
                why_not_looks_public = ""
                if elem_module is None:
                    why_not_looks_public = "because it does not have a `__module__` attribute"
                # If a module is being migrated from foo.a to bar.a (that is entry {"foo": "bar"}),
                # the module's starting package would be referred to as the new location even
                # if there is a "from foo import a" inside the "bar.py".
                modname = allow_dict["being_migrated"].get(modname, modname)
                elem_modname_starts_with_mod = elem_module is not None and \
                    elem_module.startswith(modname) and \
                    '._' not in elem_module
                if not why_not_looks_public and not elem_modname_starts_with_mod:
                    why_not_looks_public = f"because its `__module__` attribute (`{elem_module}`) is not within the " \
                        f"torch library or does not start with the submodule where it is defined (`{modname}`)"
                # elem's name must NOT begin with an `_` and it's module name
                # SHOULD start with it's current module since it's a public API
                looks_public = not elem.startswith('_') and elem_modname_starts_with_mod
                if not why_not_looks_public and not looks_public:
                    why_not_looks_public = f"because it starts with `_` (`{elem}`)"

                if is_public != looks_public:
                    # Skip some APIs which don't meet the guidelines for public API until they are fixed.
                    if f"{modname}.{elem}" in temp_filter or \
                            modname.startswith("torch_npu.dynamo.torchair.ge_concrete_graph"):
                        return

                    if ((modname in allow_dict and elem in allow_dict[modname]) or
                        (modname in deprecated_dict and elem in deprecated_dict[modname])):
                        return

                    if is_public:
                        why_is_public = f"it is inside the module's (`{modname}`) `__all__`" if is_all else \
                            "it is an attribute that does not start with `_` on a module that " \
                            "does not have `__all__` defined"
                        fix_is_public = f"remove it from the modules's (`{modname}`) `__all__`" if is_all else \
                            f"either define a `__all__` for `{modname}` or add a `_` at the beginning of the name"
                    else:
                        assert is_all
                        why_is_public = f"it is not inside the module's (`{modname}`) `__all__`"
                        fix_is_public = f"add it from the modules's (`{modname}`) `__all__`"

                    if looks_public:
                        why_looks_public = "it does look public because it follows the rules from the doc above " \
                            "(does not start with `_` and has a proper `__module__`)."
                        fix_looks_public = "make its name start with `_`"
                    else:
                        why_looks_public = why_not_looks_public
                        if not elem_modname_starts_with_mod:
                            fix_looks_public = "make sure the `__module__` is properly set and points to a submodule "\
                                f"of `{modname}`"
                        else:
                            fix_looks_public = "remove the `_` at the beginning of the name"

                    failure_list.append(f"# {modname}.{elem}:")
                    is_public_str = "" if is_public else " NOT"
                    failure_list.append(f"  - Is{is_public_str} public: {why_is_public}")
                    looks_public_str = "" if looks_public else " NOT"
                    failure_list.append(f"  - Does{looks_public_str} look public: {why_looks_public}")
                    # Swap the str below to avoid having to create the NOT again
                    failure_list.append("  - You can do either of these two things to fix this problem:")
                    failure_list.append(f"    - To make it{looks_public_str} public: {fix_is_public}")
                    failure_list.append(f"    - To make it{is_public_str} look public: {fix_looks_public}")

            if hasattr(mod, '__all__'):
                public_api = mod.__all__
                all_api = dir(mod)
                for elem in all_api:
                    check_one_element(elem, modname, mod, is_public=elem in public_api, is_all=True)
            else:
                all_api = dir(mod)
                for elem in all_api:
                    if not elem.startswith('_'):
                        check_one_element(elem, modname, mod, is_public=True, is_all=False)

        for modname in _find_all_importables(torch):
            test_module(modname)

        for modname in _find_all_importables(torch_npu):
            test_module(modname)

        test_module('torch')
        test_module('torch_npu')

        msg = "All the APIs below do not meet our guidelines for public API from " \
              "pytorch/wiki/Public-API-definition-and-documentation.\n"
        msg += "Make sure that everything that is public is expected (in particular that the module " \
            "has a properly populated `__all__` attribute) and that everything that is supposed to be public " \
            "does look public (it does not start with `_` and has a `__module__` that is properly populated)."
        msg += "\n\nFull list:\n"
        msg += "\n".join(map(str, failure_list))

        # empty lists are considered false in python
        self.assertTrue(not failure_list, msg)


if __name__ == '__main__':
    run_tests()
