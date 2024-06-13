# Owner(s): ["module: autograd"]
import pkgutil
import importlib
from typing import Callable
import inspect
import json
import os
import unittest
from importlib import import_module

import torch
import torch_npu
import torch_npu.testing
from torch.testing._internal.common_utils import TestCase, run_tests, IS_JETSON, IS_WINDOWS

tempFilter = {
    "torch_npu.contrib.FastBatchNorm1d",
    "torch_npu.contrib.FastBatchNorm2d",
    "torch_npu.contrib.FastBatchNorm3d",
    "torch_npu.contrib.FastSyncBatchNorm",
    "torch_npu.contrib.module.FastBatchNorm1d",
    "torch_npu.contrib.module.FastBatchNorm2d",
    "torch_npu.contrib.module.FastBatchNorm3d",
    "torch_npu.contrib.module.FastSyncBatchNorm",
    "torch_npu.npu.amp.autocast_mode.Any",
    "torch_npu.npu.amp.autocast_mode.ErrCode",
    "torch_npu.npu.amp.autocast_mode.pta_error",
    "torch_npu.npu.amp.grad_scaler.Cuda_GradScaler",
    "torch_npu.npu.amp.grad_scaler.ErrCode",
    "torch_npu.npu.amp.grad_scaler.List",
    "torch_npu.npu.amp.grad_scaler.OptState",
    "torch_npu.npu.amp.grad_scaler.amp_definitely_not_available",
    "torch_npu.npu.amp.grad_scaler.defaultdict",
    "torch_npu.npu.amp.grad_scaler.pta_error",
    "torch_npu.npu.amp.sharded_grad_scaler.Dict",
    "torch_npu.npu.amp.sharded_grad_scaler.ErrCode",
    "torch_npu.npu.amp.sharded_grad_scaler.GradScaler",
    "torch_npu.npu.amp.sharded_grad_scaler.List",
    "torch_npu.npu.amp.sharded_grad_scaler.OptState",
    "torch_npu.npu.amp.sharded_grad_scaler.Optional",
    "torch_npu.npu.amp.sharded_grad_scaler.ProcessGroup",
    "torch_npu.npu.amp.sharded_grad_scaler.SGD",
    "torch_npu.npu.amp.sharded_grad_scaler.Union",
    "torch_npu.npu.amp.sharded_grad_scaler.defaultdict",
    "torch_npu.npu.amp.sharded_grad_scaler.npu_check_overflow",
    "torch_npu.npu.amp.sharded_grad_scaler.pta_error",
    "torch_npu.profiler.analysis.prof_bean.node_info_bean.NodeInfoBean",
    "torch_npu.profiler.analysis.prof_common_func.time_range_calculator.dataclass",
    "torch_npu.profiler.analysis.prof_view.prof_db_parse.fwk_api_db_parser.CannNodeLaunchApiOri",
    "torch_npu.profiler.analysis.prof_view.prof_db_parse.fwk_api_db_parser.PythonTraceApiDataOri",
    "torch_npu.profiler.analysis.prof_view.prof_db_parse.fwk_api_db_parser.TaskQueueDataOri",
    "torch_npu.profiler.analysis.prof_view.prof_db_parse.fwk_api_db_parser.TorchOpDataOri",
    "torch_npu.profiler.analysis.prof_view.prof_db_parse.memory_db_parser.GeOpMemRecordsOri",
    "torch_npu.profiler.profiler.supported_activities",
    "torch_npu.testing.common_distributed.Any",
    "torch_npu.testing.common_distributed.Dict",
    "torch_npu.testing.common_distributed.Tuple",
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
    "torch_npu.utils.profiler.ErrCode",
    "torch_npu.utils.profiler.Optional",
    "torch_npu.utils.profiler.prof_error",
    "torch_npu.npu_add_rms_norm",
    "torch_npu.npu_deep_norm",
    "torch_npu.npu_fast_gelu",
    "torch_npu.npu_fused_attention_layernorm_qkv_fwd",
    "torch_npu.npu_fused_attention_score_fwd",
    "torch_npu.npu_group_norm_silu",
    "torch_npu.npu_lstm_cell",
    "torch_npu.npu_masked_softmax_with_rel_pos_bias",
    "torch_npu.npu_mm_all_reduce_base",
    "torch_npu.npu_quant_update_scatter",
    "torch_npu.npu_trans_quant_param",
    "torch_npu.npu_moe_compute_expert_tokens",
    "torch_npu.one_",
    "torch_npu.utils.collect_env.main",
    "torch_npu.utils.collect_env.namedtuple"
}


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

    def test_modules_can_be_imported(self):
        failures = []
        for _, modname, _ in pkgutil.walk_packages(path=torch.__path__, prefix=torch.__name__ + '.'):
            try:
                if "__main__" in modname:
                    continue
                import_module(modname)
            except Exception as e:
                # Some current failures are not ImportError
                failures.append((modname, type(e)))

        for _, modname, _ in pkgutil.walk_packages(path=torch_npu.__path__, prefix=torch_npu.__name__ + '.'):
            try:
                if "__main__" in modname:
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
            "torch.onnx._internal.fx.diagnostics",
            "torch.onnx._internal.fx.fx_onnx_interpreter",
            "torch.onnx._internal.fx.fx_symbolic_graph_extractor",
            "torch.onnx._internal.fx.onnxfunction_dispatcher",
            "torch.onnx._internal.fx.op_validation",
            "torch.onnx._internal.fx.passes",
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
            "torch.testing._internal.distributed.pipe_with_ddp_test",
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
            "torch._inductor.triton_helpers",
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
            "torch.distributed.algorithms._checkpoint.checkpoint_wrapper",
            "torch.distributed.algorithms._optimizer_overlap",
            "torch.distributed.rpc._testing.faulty_agent_backend_registry",
            "torch.distributed.rpc._utils",
            "torch.utils.tensorboard"
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
            "torch.distributed.pipeline.sync",
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
            "torch_npu.utils.collect_hccl_info"
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
    @unittest.skipIf(IS_WINDOWS or IS_JETSON, "Distributed Attribute Error")
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
        with open(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'allowlist_for_publicAPI.json')) as json_file:
            # no new entries should be added to this allow_dict.
            # New APIs must follow the public API guidelines.
            allow_dict = json.load(json_file)
            # Because we want minimal modifications to the `allowlist_for_publicAPI.json`,
            # we are adding the entries for the migrated modules here from the original
            # locations.
            for modname in allow_dict["being_migrated"]:
                if modname in allow_dict:
                    allow_dict[allow_dict["being_migrated"][modname]] = allow_dict[modname]

        def test_module(modname):
            try:
                if "__main__" in modname:
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
                if not (isinstance(obj, Callable) or inspect.isclass(obj)):
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
                    if f"{modname}.{elem}" in tempFilter:
                        return

                    if modname in allow_dict and elem in allow_dict[modname]:
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

        for _, modname, ispkg in pkgutil.walk_packages(path=torch.__path__, prefix=torch.__name__ + '.'):
            test_module(modname)

        for _, modname, ispkg in pkgutil.walk_packages(path=torch_npu.__path__, prefix=torch_npu.__name__ + '.'):
            test_module(modname)

        test_module('torch')
        test_module('torch_npu')

        msg = "All the APIs below do not meet our guidelines for public API from " \
              "wiki/Public-API-definition-and-documentation.\n"
        msg += "Make sure that everything that is public is expected (in particular that the module " \
            "has a properly populated `__all__` attribute) and that everything that is supposed to be public " \
            "does look public (it does not start with `_` and has a `__module__` that is properly populated)."
        msg += "\n\nFull list:\n"
        msg += "\n".join(map(str, failure_list))

        # empty lists are considered false in python
        self.assertTrue(not failure_list, msg)


if __name__ == '__main__':
    run_tests()
