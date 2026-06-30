# Owner(s): ["module: unknown"]
import os
import subprocess
import sys
import textwrap
import unittest
from packaging import version

import torch_npu
from torch_npu.testing.testcase import run_tests, TestCase


ENABLE_2_7_1_TEST = (
    version.parse(torch_npu.__version__).base_version == "2.7.1"
)

REQUIRED_C_EXTENSION_CHILDREN = [
    "_profiler",
    "_distributed_c10d",
    "_cd",
    "_logging",
    "_flops_count",
]

EXPECTED_LOADED_MODULES = [
    "torch_npu.npu",
    "torch_npu.npu.amp",
    "torch_npu.npu.aclnn",
    "torch_npu.optim",
    "torch_npu.dynamo",
    "torch_npu._logging",
    "torch_npu._afd",
    "torch_npu.profiler",
    "torch_npu.distributed",
    "torch_npu.distributed.rpc",
    "torch_npu.distributed.nn",
    "torch_npu.distributed.nn.functional",
    "torch_npu.distributed._symmetric_memory",
    "torch_npu.op_plugin",
    "torch_npu.op_plugin.meta",
    "torch_npu.op_plugin.meta._meta_registrations",
    "torch_npu.asd.checksum",
    "torch_npu.utils._dynamo",
    "torch_npu.utils.custom_ops",
    "torch_npu.utils.patch_getenv",
    "torch_npu.utils.syncbatchnorm",
]

EXPECTED_NOT_LOADED_MODULES = [
    "torch_npu._C._afd",
]

EXPECTED_TOP_LEVEL_ATTRS = [
    "npu",
    "optim",
    "dynamo",
    "_afd",
    "profiler",
    "op_plugin",
    "utils",
]

LAZY_TOP_LEVEL_APIS = [
    "HiFloat8Tensor",
    "erase_stream",
    "matmul_checksum",
]

AFD_OPS = [
    "attention_worker_scheduler_",
    "attention_worker_scheduler",
    "ffn_worker_scheduler_",
    "ffn_worker_scheduler",
]


class TestTorchNpuBootstrap(TestCase):
    def _run_python(self, code: str, *, optional: bool = False):
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            text=True,
            capture_output=True,
            env=os.environ.copy(),
        )

        if proc.returncode == 0:
            return

        message = (
            f"subprocess failed with return code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

        if optional:
            self.skipTest(message)

        self.fail(message)

    def test_01_import_order_compatibility(self):
        cases = [
            "import torch_npu",
            "import torch\nimport torch_npu",
            "import torch_npu\nimport torch",
            "import torch_npu\nimport torch_npu",
        ]

        for code in cases:
            self._run_python(code)

    def test_02_import_state_snapshot(self):
        self._run_python(
            f"""
            import sys
            import torch
            import torch_npu
            import torch_npu._C as C

            required_c_children = {REQUIRED_C_EXTENSION_CHILDREN!r}
            expected_loaded = {EXPECTED_LOADED_MODULES!r}
            expected_not_loaded = {EXPECTED_NOT_LOADED_MODULES!r}
            expected_top_attrs = {EXPECTED_TOP_LEVEL_ATTRS!r}

            assert hasattr(torch, "npu"), "torch.npu is not registered"
            assert torch.npu is torch_npu.npu

            assert hasattr(torch.Tensor, "npu"), "torch.Tensor.npu is missing"
            assert hasattr(torch.nn.Module, "npu"), "torch.nn.Module.npu is missing"

            for name in required_c_children:
                assert hasattr(C, name), f"torch_npu._C.{{name}} is missing"

            # Old behavior: AFD is exposed as torch_npu._afd, not torch_npu._C._afd.
            assert hasattr(C, "_afd") is False

            missing_modules = [
                name for name in expected_loaded if name not in sys.modules
            ]
            assert not missing_modules, (
                f"init-time modules changed, missing: {{missing_modules}}"
            )

            unexpected_modules = [
                name for name in expected_not_loaded if name in sys.modules
            ]
            assert not unexpected_modules, (
                f"unexpected eager modules loaded: {{unexpected_modules}}"
            )

            missing_attrs = [
                name for name in expected_top_attrs if not hasattr(torch_npu, name)
            ]
            assert not missing_attrs, (
                f"torch_npu top-level attrs changed, missing: {{missing_attrs}}"
            )

            import torch_npu.npu
            import torch_npu.npu.aclnn

            assert torch_npu.npu is not None
            assert torch_npu.npu.aclnn is not None

            # _op_plugin_docs is imported for side effect, then removed from top-level.
            assert "torch_npu._op_plugin_docs" in sys.modules
            assert not hasattr(torch_npu, "_op_plugin_docs")
            """
        )

    def test_03_public_exports_snapshot(self):
        self._run_python(
            f"""
            import torch
            import torch_npu
            import torch_npu._C as C
            from torch_npu.utils.exposed_api import public_npu_functions

            lazy_names = {LAZY_TOP_LEVEL_APIS!r}

            for name in lazy_names:
                assert name in torch_npu.__all__, f"{{name}} is missing from __all__"
                assert name in dir(torch_npu), f"{{name}} is missing from dir(torch_npu)"
                assert name not in torch_npu.__dict__, (
                    f"{{name}} should not be cached before lazy access"
                )

                value = getattr(torch_npu, name)

                assert value is not None
                assert name in torch_npu.__dict__, (
                    f"{{name}} was not cached after lazy access"
                )

            available_public_ops = []
            missing_from_torch_npu = []
            missing_from_all = []
            missing_torch_alias = []

            for name in public_npu_functions:
                if not hasattr(torch.ops.npu, name):
                    continue

                available_public_ops.append(name)

                if not hasattr(torch_npu, name):
                    missing_from_torch_npu.append(name)

                if name not in torch_npu.__all__:
                    missing_from_all.append(name)

                if not hasattr(torch, name):
                    missing_torch_alias.append(name)

            assert available_public_ops, "no available public torch.ops.npu ops found"
            assert not missing_from_torch_npu, (
                f"some public ops are missing from torch_npu: "
                f"{{missing_from_torch_npu[:20]}}"
            )
            assert not missing_from_all, (
                f"some public ops are missing from torch_npu.__all__: "
                f"{{missing_from_all[:20]}}"
            )
            assert not missing_torch_alias, (
                f"some public ops are missing deprecated torch aliases: "
                f"{{missing_torch_alias[:20]}}"
            )

            dtype_names = [
                name
                for name in dir(C._cd.DType)
                if not name.startswith("_") and name not in ["_dir", "name"]
            ]

            missing_dtype = []
            mismatch_dtype = []

            for name in dtype_names:
                if not hasattr(torch_npu, name):
                    missing_dtype.append(name)
                    continue

                exported = getattr(torch_npu, name)
                source = getattr(C._cd.DType, name)

                # Pybind objects may not preserve Python identity across getattr calls.
                if exported != source and repr(exported) != repr(source):
                    mismatch_dtype.append((name, repr(exported), repr(source)))

            assert not missing_dtype, (
                f"some DType symbols are missing from torch_npu: {{missing_dtype}}"
            )
            assert not mismatch_dtype, (
                f"some DType symbols do not match torch_npu._C._cd.DType: "
                f"{{mismatch_dtype[:10]}}"
            )
            """
        )

    def test_04_framework_registration_snapshot(self):
        self._run_python(
            """
            import torch_npu
            import torch.distributed as dist
            import torch.distributed.rpc as rpc
            import torch.distributed.tensor  # noqa: F401
            from torch._dynamo.device_interface import get_interface_for_device
            from torch._dynamo.backends.registry import _BACKENDS
            from torch._inductor.codegen.common import device_op_overrides_dict

            iface = get_interface_for_device("npu")
            assert iface is not None

            assert "npu" in _BACKENDS, "npu dynamo backend is not registered"
            assert "npugraph_ex" in _BACKENDS, (
                "npugraph_ex dynamo backend is not registered"
            )

            assert "npu" in device_op_overrides_dict
            assert device_op_overrides_dict.get("npu") is not None

            assert "hccl" in dist.Backend.backend_list
            assert "lccl" in dist.Backend.backend_list

            names = [
                name for name in dir(rpc.BackendType)
                if "NPU" in name or "TENSORPIPE" in name
            ]
            assert hasattr(rpc, "BackendType")
            assert "NPU_TENSORPIPE" in names
            """,
            optional=True,
        )

    def test_05_runtime_lazy_init_semantics(self):
        self._run_python(
            """
            import torch
            import torch_npu

            assert torch_npu.npu.is_initialized() is False, (
                "import torch_npu unexpectedly triggered NPU lazy init"
            )

            torch.npu.is_available()
            torch.npu.device_count()

            assert torch_npu.npu.is_initialized() is False, (
                "availability query unexpectedly triggered NPU lazy init"
            )
            """
        )

        self._run_python(
            """
            import sys
            import torch_npu

            assert torch_npu.npu.is_initialized() is False

            if torch_npu.npu.device_count() <= 0:
                sys.exit(0)

            torch_npu.npu.get_device_properties(0)

            assert torch_npu.npu.is_initialized() is True, (
                "runtime NPU API did not trigger lazy init"
            )
            """
        )

        self._run_python(
            """
            import torch_npu

            assert torch_npu.npu.is_initialized() is False

            torch_npu.npu.init()

            assert torch_npu.npu.is_initialized() is True
            """
        )

    def test_06_component_behavior_snapshot(self):
        self._run_python(
            f"""
            import os
            import sys
            import torch_npu
            import torch_npu._C as C
            import torch_npu._afd
            import torch_npu.utils as utils
            import torch_npu.utils.asd_detector as asd_detector
            import torch_npu.utils.patch_getenv as patch_getenv

            afd_ops = {AFD_OPS!r}

            # patch_getenv behavior.
            assert os.getenv is patch_getenv._patched_getenv
            assert os.environ.get is patch_getenv._patched_environ_get

            # ASD compatibility APIs.
            for module_name in [
                "torch_npu.utils._asd_detector",
                "torch_npu.utils.asd_detector",
            ]:
                assert module_name in sys.modules, (
                    f"{{module_name}} is not loaded after import torch_npu"
                )

            for api_name in ["set_asd_loss_scale", "register_asd_hook"]:
                assert hasattr(utils, api_name), (
                    f"torch_npu.utils.{{api_name}} is missing"
                )
                assert hasattr(asd_detector, api_name), (
                    f"torch_npu.utils.asd_detector.{{api_name}} is missing"
                )

                utils_api = getattr(utils, api_name)
                detector_api = getattr(asd_detector, api_name)

                assert callable(utils_api)
                assert callable(detector_api)
                assert utils_api is detector_api

            # AFD compatibility behavior.
            assert hasattr(C, "_afd") is False
            assert "torch_npu._afd" in sys.modules
            assert "torch_npu._C._afd" not in sys.modules

            try:
                import torch_npu._C._afd  # noqa: F401
                raise AssertionError("import torch_npu._C._afd should fail")
            except ModuleNotFoundError:
                pass

            for name in afd_ops:
                assert hasattr(torch_npu._afd, name), (
                    f"torch_npu._afd.{{name}} is missing"
                )
            """
        )

    def test_07_distributed_patch_behavior(self):
        self._run_python(
            """
            import sys
            import torch
            import torch_npu
            import torch.distributed as dist
            import torch.distributed.distributed_c10d as c10d
            import torch.distributed.launcher.api as launcher_api
            from torch.distributed.fsdp import sharded_grad_scaler
            from torch.distributed.fsdp._fully_shard import _fsdp_collectives
            from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
                FSDPParamGroup,
            )
            from torch_npu.distributed.fsdp._add_fsdp_patch import (
                _patched_finalize_backward,
            )
            from torch_npu.npu.amp.sharded_grad_scaler import _ShardedGradScaler

            assert torch._C._distributed_c10d._verify_params_across_processes is (
                torch_npu.distributed._verify_params_across_processes
            )

            assert torch._C._distributed_c10d.ProcessGroup._get_sequence_number_for_group is (
                torch_npu.distributed.distributed_c10d._hccl_get_sequence_number_for_group
            )

            assert c10d._add_ephemeral_timeout_for_all_pgs is (
                torch_npu.distributed.distributed_c10d._hccl_add_ephemeral_timeout_for_all_pgs
            )

            assert dist.batch_isend_irecv is (
                torch_npu.distributed.distributed_c10d._batch_isend_irecv
            )
            assert c10d.batch_isend_irecv is (
                torch_npu.distributed.distributed_c10d._batch_isend_irecv
            )

            assert dist.gather is torch_npu.distributed.distributed_c10d._gather
            assert c10d.gather is torch_npu.distributed.distributed_c10d._gather

            assert dist.gather_object is torch_npu.distributed.distributed_c10d._gather_object
            assert c10d.gather_object is torch_npu.distributed.distributed_c10d._gather_object

            assert dist.is_hccl_available is torch_npu.distributed.is_hccl_available
            assert dist.reinit_process_group is torch_npu.distributed.reinit_process_group

            assert callable(c10d.rendezvous)
            assert callable(launcher_api._get_addr_and_port)

            assert sharded_grad_scaler.ShardedGradScaler is _ShardedGradScaler
            assert FSDPParamGroup.finalize_backward is _patched_finalize_backward
            """
        )
    
    def test_08_top_level_unsupported_dtype_compatibility(self):
        self._run_python(
            """
            import torch
            import torch_npu

            # Regression test for external packages such as MindSpeed.
            # They access torch_npu.unsupported_dtype directly after importing torch_npu.
            expected_unsupported_dtype = [
                torch.quint8,
                torch.quint4x2,
                torch.quint2x4,
                torch.qint32,
                torch.qint8,
            ]

            unsupported_dtype = torch_npu.unsupported_dtype

            assert unsupported_dtype == expected_unsupported_dtype
            assert "unsupported_dtype" in dir(torch_npu)
            assert "unsupported_dtype" in torch_npu.__dict__

            # Simulate MindSpeed-style dtype filtering.
            valid_dtype_names = []
            for name, attr in torch.__dict__.items():
                if isinstance(attr, torch.dtype) and attr not in torch_npu.unsupported_dtype:
                    valid_dtype_names.append(name)

            assert valid_dtype_names, "no valid torch dtype found"
            """
        )

    def test_09_legacy_submodule_attribute_compatibility(self):
        self._run_python(
            """
            import sys
            import torch_npu

            # Old behavior: importing torch_npu also exposed these child modules
            # as attributes on their parent packages.

            assert "torch_npu.asd.checksum" in sys.modules
            assert hasattr(torch_npu, "asd")
            assert hasattr(torch_npu.asd, "checksum")
            assert torch_npu.asd.checksum is sys.modules["torch_npu.asd.checksum"]
            assert hasattr(torch_npu.asd.checksum, "_matmul_checksum")

            assert "torch_npu.distributed._symmetric_memory" in sys.modules
            assert hasattr(torch_npu.distributed, "_symmetric_memory")
            assert torch_npu.distributed._symmetric_memory is (
                sys.modules["torch_npu.distributed._symmetric_memory"]
            )

            assert "torch_npu.utils.syncbatchnorm" in sys.modules
            assert hasattr(torch_npu.utils, "syncbatchnorm")
            assert torch_npu.utils.syncbatchnorm is (
                sys.modules["torch_npu.utils.syncbatchnorm"]
            )
            """
        )

    def test_10_legacy_top_level_distributed_api_compatibility(self):
        self._run_python(
            """
            import torch_npu
            from torch.distributed.fsdp import sharded_grad_scaler
            from torch_npu._C._distributed_c10d import ParallelStore
            from torch_npu.npu.amp.sharded_grad_scaler import _ShardedGradScaler

            # Old behavior: these names were visible on torch_npu top-level
            # due to module-scope imports in the old monolithic __init__.py.
            assert hasattr(torch_npu, "ParallelStore")
            assert torch_npu.ParallelStore is ParallelStore
            assert "ParallelStore" not in torch_npu.__all__

            assert hasattr(torch_npu, "_ShardedGradScaler")
            assert torch_npu._ShardedGradScaler is _ShardedGradScaler
            assert "_ShardedGradScaler" not in torch_npu.__all__

            # The FSDP patch behavior should still be preserved.
            assert sharded_grad_scaler.ShardedGradScaler is _ShardedGradScaler
            """
        )

    def test_11_import_does_not_trigger_device_count(self):
        cases = [
            "import torch; import torch_npu",
            "import torch_npu",
        ]

        for import_code in cases:
            self._run_python(
                f"""
                import os

                {import_code}

                assert torch_npu.npu.is_initialized() is False, (
                    "import torch_npu unexpectedly triggered NPU lazy init"
                )

                # Regression test for import-time low-level device probing.
                # If import torch_npu has already called low-level NPU device count,
                # changing ASCEND_RT_VISIBLE_DEVICES here will no longer take effect.
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "32"

                raw_count = torch_npu._C._npu_getDeviceCount()
                assert raw_count == 0, (
                    f"import torch_npu unexpectedly triggered low-level NPU "
                    f"device probing, raw_count={{raw_count}}"
                )

                assert torch_npu.npu.is_initialized() is False
                """
            )

    @unittest.skipIf(not ENABLE_2_7_1_TEST, "only for torch_npu 2.7.1")
    def test_12_distributed_enable_symm_mem_for_group(self):
        self._run_python(
            """
            import torch
            import torch_npu
            import torch.distributed as dist
            import torch.distributed.nn.functional as F

            # 1. symmetric memory patch check
            assert hasattr(dist._symmetric_memory, "enable_symm_mem_for_group"), (
                "missing symmetric memory patch"
            )
            assert (
                dist._symmetric_memory.enable_symm_mem_for_group
                is not None
            )
            assert (
                dist._symmetric_memory.enable_symm_mem_for_group
                is torch_npu.distributed._symmetric_memory._enable_symm_mem_for_group
            )
            """
        )

    def test_13_distributed_allgatherbase_backward(self):
        self._run_python(
            """
            import torch
            import torch_npu
            import torch.distributed as dist
            import torch.distributed.nn.functional as F

            # 1. AllGatherBase backward patch check
            assert hasattr(F._AllGatherBase, "backward"), (
                "missing AllGatherBase backward patch"
            )
            assert (
                F._AllGatherBase.backward
                is torch_npu.distributed.nn.functional._allgather_base_backward_hccl
            )
            # 2. ensure patch not default torch impl
            import inspect
            assert "torch_npu" in str(F._AllGatherBase.backward.__module__)
            """
        )

if __name__ == "__main__":
    run_tests()
