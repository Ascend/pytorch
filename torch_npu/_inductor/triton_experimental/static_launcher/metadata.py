# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from dataclasses import dataclass
from enum import Enum


NPU_STATIC_LAUNCH_SCHEMA_VERSION = 1
_SUPPORTED_TARGET_PREFIXES = (
    "Ascend910B",
    "Ascend910D",
    "Ascend910_93",
    "Ascend910_95",
    "Ascend950",
)


class NPUHostArgsLayout(str, Enum):
    """Triton-Ascend host-argument ABIs consumed by the CANN kernel binary.

    Reference implementations:
      * 3.2.2: third_party/ascend/backend/driver.py::generate_npu_wrapper_src
      * 3.6.0: third_party/ascend/backend/driver.py::make_launcher

    Keep version detection in adapter.py and the byte layout in
    runtime.cpp::StaticNpuKernel::BuildPackedLayout.
    """

    TRITON_ASCEND_3_2 = "triton_ascend_3_2"
    TRITON_ASCEND_3_6 = "triton_ascend_3_6"


@dataclass(frozen=True)
class NPULaunchMetadata:
    schema_version: int
    host_args_layout: NPUHostArgsLayout
    target_arch: str
    mix_mode: str
    parallel_mode: str
    enable_simt: bool
    shared_mem_dynamic_size: int
    is_pure_simt: bool
    target_support_ffts: bool
    num_ctas: int
    cluster_dims: tuple[int, int, int]
    workspace_size: int
    lock_num: int
    device_print_enabled: bool

    @property
    def trailing_pointer_count(self) -> int:
        if self.host_args_layout == NPUHostArgsLayout.TRITON_ASCEND_3_2:
            # 3.2 has no unconditional tail slot; its conditional DTData mode
            # is rejected by static-launch eligibility checks.
            return 0
        if self.host_args_layout == NPUHostArgsLayout.TRITON_ASCEND_3_6:
            # 3.6 order after gridX/Y/Z:
            #   non-pure SIMT: DTData
            #   pure SIMT: global_scratch, profile_scratch, DTData
            # Static launch rejects device print and does not allocate scratch,
            # so all of these pointer-sized ABI slots remain zero-initialized.
            return 3 if self.is_pure_simt else 1
        raise RuntimeError(
            f"Unsupported NPU static launcher host args layout: {self.host_args_layout!r}"
        )

    def validate(self) -> None:
        if self.schema_version != NPU_STATIC_LAUNCH_SCHEMA_VERSION:
            raise RuntimeError(
                "Unsupported NPU static launcher metadata schema: "
                f"{self.schema_version}"
            )
        if not isinstance(self.host_args_layout, NPUHostArgsLayout):
            raise RuntimeError(
                "Unsupported NPU static launcher host args layout: "
                f"{self.host_args_layout!r}"
            )
        if self.mix_mode not in ("aic", "aiv"):
            raise RuntimeError(
                f"Unsupported NPU static launcher mix mode: {self.mix_mode}"
            )
        if not self.target_arch.startswith(_SUPPORTED_TARGET_PREFIXES):
            raise RuntimeError(
                "NPU static launcher requires an Ascend 910B-or-newer "
                f"target, got {self.target_arch!r}"
            )
        if not self.parallel_mode:
            raise RuntimeError("NPU static launcher parallel_mode is missing")
        if self.shared_mem_dynamic_size < 0:
            raise RuntimeError("shared_mem_dynamic_size must be non-negative")
        if self.workspace_size < 0 or self.lock_num < 0:
            raise RuntimeError("workspace_size and lock_num must be non-negative")
        if self.is_pure_simt and not self.enable_simt:
            raise RuntimeError("is_pure_simt requires enable_simt")
        if self.num_ctas != 1 or self.cluster_dims != (1, 1, 1):
            raise RuntimeError(
                "NPU static launcher only supports num_ctas=1 and "
                "cluster_dims=(1, 1, 1)"
            )


__all__ = [
    "NPUHostArgsLayout",
    "NPULaunchMetadata",
    "NPU_STATIC_LAUNCH_SCHEMA_VERSION",
]
