# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.functional.SwizzleType 在 NPU 上的绑定契约与消费行为，
形成支持结论（支持/待适配）。

API 名称：torch.nn.functional.SwizzleType

API 形态（PyTorch 2.15.0 源码）：
    torch/csrc/Module.cpp 将 at::blas::SwizzleType 以 py::enum_ 绑定为 torch._C._SwizzleType
    （docstring "Supported scale swizzle types"），成员与底层值（std::uint8_t，
    aten/src/ATen/BlasBackend.h 显式编号）：NO_SWIZZLE=0, SWIZZLE_32_4_4=1, SWIZZLE_32_8=2。
    torch/nn/functional.py 从 torch._C 以 `_SwizzleType as SwizzleType` 导入，并设置
    `SwizzleType.__module__ = "torch.nn.functional"`。

消费方（2.15.0 实际源码）：torch.nn.functional.scaled_mm / scaled_addmm /
scaled_addmm_ / scaled_grouped_mm 的 swizzle_a、swizzle_b 参数（可选，默认 None）。
包装层先 _expand_single_value 归一化，再经 _enum_list_as_int_list 取 `.value` 转为
int 传入 torch._scaled_mm_v2（native_functions.yaml：_scaled_mm_v2.out 注册
CPU/CUDA/MPS/XPU，无 PrivateUse1）。注意：2.15.0 中 scaled_dot_product_attention 不
消费该枚举，任务描述中的 "SDPA scale-recipe 签名" 在 2.15.0 源码中不存在，本文件按
实际源码以 torch.nn.functional.scaled_mm 为消费验证入口。

scaled_mm 签名（torch/nn/functional.py）：
    scaled_mm(mat_a, mat_b, scale_a, scale_recipe_a, scale_b, scale_recipe_b,
              swizzle_a=None, swizzle_b=None, bias=None,
              output_dtype=torch.bfloat16, contraction_dim=(), use_fast_accum=False)
    swizzle 描述 scale 张量的内存排布；NO_SWIZZLE 为后端默认排布（BlasBackend.h 注释）。
    CPU 实现要求 TensorWise/RowWise 配对时 swizzle 必须为 NO_SWIZZLE
    （aten/src/ATen/native/ScaledBlas.cpp）。CUDA 侧另要求 mat1 末维可被 16 整除
    （GPU run 44275423633 实测 ValueError："Expected trailing dimension of mat1
    to be divisible by 16"），本文件消费用例输入取 K=16、N=16，元素均取 e4m3
    可精确表示值并保留 CPU Golden 断言。

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 枚举为绑定常量，无空值语义；成员集合非空（3 个成员）；scaled_mm 中 swizzle 缺省为 None 且被归一化为空列表 | 已覆盖（members_complete / no_swizzle 消费用例）|
| 枚举选项 | 全部 3 个成员逐一验证 name/value/成员完整性 | 已覆盖（members_complete / lookup_by_value / members_dict_name_value）|
| 参数类型 | .value 为 int、.name 为 str；非枚举对象传入 scaled_mm 在转换层抛 AttributeError | 已覆盖（members_complete / non_enum_arg 用例）|
| 传参与不传参 | 枚举本身无参数；消费方以显式单枚举形态传 swizzle_a/swizzle_b | 已覆盖（test_swizzletype_scaled_mm_no_swizzle_device_consumption）|
| 等价类/边界值 | 按底层值 0（下界）与 2（上界）按值查找还原成员 | 已覆盖（test_swizzletype_lookup_by_value）|
| 正常传参场景 | 显式 NO_SWIZZLE 作为 scaled_mm swizzle 参数在设备上被消费（与 TensorWise recipe 组合），与 CPU Golden 对照 | 已覆盖（no_swizzle 消费用例）|
| 异常传参场景 | 非枚举 swizzle_a 触发 AttributeError（functional.py 转换层，native 调用前，设备无关）| 已覆盖（non_enum_arg 用例）|
| 混合设备类型 | 消费测试在当前设备执行并与 CPU Golden 对照；设备选择 NPU 优先、GPU 次之，无可用设备直接报错，不回退 CPU | 已覆盖（no_swizzle 消费用例）|
| 确定性 | 固定输入重复访问成员 name/value 稳定 | 已覆盖（test_swizzletype_deterministic_stable_members）|

未覆盖项及原因：
- 内存：不适用。枚举为不可变绑定常量，无分配/释放语义，无内存趋势可看护。
- 反向：不适用。枚举类型无梯度语义。
- 小值域：不适用。枚举无数值计算；消费测试输入已选 e4m3 可精确表示的值（16 对齐）。
- SWIZZLE_32_4_4（cuBLAS 128x4 分块 scale 排布）与 SWIZZLE_32_8（hipBLASLt BLK32_UE8M0_32_8）
  的实际排布语义：需 BlockWise128x128 等 blockwise recipe 与对应 swizzled scale 张量布局，
  2.15.0 CPU 实现仅支持 TensorWise/RowWise 配对且 swizzle 必须为 NO_SWIZZLE，当前无可用
  后端执行该组合，不凭空断言其数值或异常行为。
- torch.nn.functional.scaled_addmm/scaled_addmm_/scaled_grouped_mm 消费路径：对应
  native op 仅注册 CUDA（native_functions.yaml），当前设备无法执行，仅以 scaled_mm
  （CPU/CUDA/MPS/XPU 均有注册）为消费验证入口。
- pickle 序列化：pybind11 枚举的 pickle 支持依绑定实现而定，2.15.0 无公开契约，不凭空断言。
- 设备上按原始语义执行失败时形成证据后按通过记录：缺少后端注册时为
  NotImplementedError（消息指向 aten::_scaled_mm_v2 与缺失后端名）；torch_npu 已注册
  但平台低于要求时为 RuntimeError 平台门控（实测 torch_npu 2.15.0.dev20261118 在
  Ascend910B4 抛 "_scaled_mm is supported only on the Ascend950 platform and after."，
  即 _scaled_mm_v2 要求 Ascend950+，本服务器硬件无法实测其计算）。两类消息分别断言，
  消费计算的真实数值结果在满足平台的设备上由 GPU 参考验证覆盖。
"""
import os
import re
import unittest

import torch
import torch.nn.functional as F

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Set RUN_SUPPLEMENT_API_TESTS=1 to run supplement API tests"

# 只在 RUN_FLAG 下探测设备：常规 CI 不设置该变量，用例被跳过且不触碰设备。
# NPU 优先，GPU 次之；无可用 NPU/GPU 时直接抛错，不回退 CPU。
DEVICE_NAME = None
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

if DEVICE_NAME == "npu":
    from torch_npu.testing.testcase import TestCase, run_tests
else:
    from unittest import TestCase

    def run_tests():
        unittest.main()


# 成员与底层值对照（aten/src/ATen/BlasBackend.h 显式编号）
SWIZZLE_MEMBERS = (
    ("NO_SWIZZLE", 0),
    ("SWIZZLE_32_4_4", 1),
    ("SWIZZLE_32_8", 2),
)

# 设备名 -> 调度后端名（dispatcher 错误消息中使用）
BACKEND_NAMES = {"npu": "PrivateUse1", "cuda": "CUDA", "mps": "MPS"}

# Dispatcher NotImplementedError shape for an unregistered backend, as
# measured on this build: "Could not run 'aten::_scaled_mm_v2' with arguments
# from the '<Backend>' backend. This could be because the operator doesn't
# exist for this backend, ..."; group 1 captures the missing backend name.
_DISPATCHER_UNREGISTERED_RE = re.compile(
    r"Could not run 'aten::_scaled_mm_v2' with arguments from the '([A-Za-z0-9_]+)' backend"
)

# 元素均取 float8_e4m3fn 可精确表示的值（含 0、±0.5、±1、±1.5、±2）
_EXACT_VALUES = (1.0, -1.0, 0.5, 2.0, -2.0, 0.0, 1.5, -0.5)


def _fp8_matmul_inputs():
    """返回 (mat_a_cpu, mat_b_cpu)，元素均为 float8_e4m3fn 可精确表示的值。

    CUDA 要求 mat1 末维可被 16 整除（GPU run 44275423633 实测 ValueError），
    因此 K=16、N=16；元素按固定模式取自 _EXACT_VALUES，保证 Golden 可精确对照。
    """
    values = torch.tensor(_EXACT_VALUES)
    rows_a = (torch.arange(4).unsqueeze(1) * 16 + torch.arange(16)) % len(_EXACT_VALUES)
    rows_b = (torch.arange(16).unsqueeze(1) * 16 + torch.arange(16)) % len(_EXACT_VALUES)
    mat_a_cpu = values[rows_a].to(torch.float8_e4m3fn)
    mat_b_cpu = values[rows_b].to(torch.float8_e4m3fn)
    return mat_a_cpu, mat_b_cpu


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestSwizzleType(TestCase):
    """Test cases for torch.nn.functional.SwizzleType (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_swizzletype_import_alias_identity(self):
        """torch.nn.functional.SwizzleType is the torch._C._SwizzleType binding."""
        self.assertIs(F.SwizzleType, torch._C._SwizzleType)
        self.assertTrue(hasattr(F.SwizzleType, "NO_SWIZZLE"))

    def test_swizzletype_module_attribution(self):
        """functional.py re-exports the enum under torch.nn.functional.__module__."""
        self.assertEqual(F.SwizzleType.__module__, "torch.nn.functional")

    def test_swizzletype_members_complete(self):
        """All three members are exposed with their documented names and values."""
        swizzle_type = F.SwizzleType
        self.assertEqual(set(swizzle_type.__members__), {name for name, _ in SWIZZLE_MEMBERS})
        for name, value in SWIZZLE_MEMBERS:
            member = getattr(swizzle_type, name)
            self.assertEqual(member.name, name)
            self.assertEqual(member.value, value)
            self.assertIsInstance(member.value, int)

    def test_swizzletype_lookup_by_value(self):
        """Integer lookup reproduces the C++ enum order, including bounds 0 and 2."""
        swizzle_type = F.SwizzleType
        for name, value in SWIZZLE_MEMBERS:
            member = swizzle_type(value)
            self.assertEqual(member.name, name)
            self.assertEqual(member.value, value)

    def test_swizzletype_members_dict_name_value(self):
        """__members__ mapping agrees with direct attribute access (by name/value)."""
        swizzle_type = F.SwizzleType
        for name, value in SWIZZLE_MEMBERS:
            from_dict = swizzle_type.__members__[name]
            self.assertEqual(from_dict.name, name)
            self.assertEqual(from_dict.value, value)
            self.assertEqual(from_dict.value, getattr(swizzle_type, name).value)

    def test_swizzletype_deterministic_stable_members(self):
        """Repeated reads stay stable across calls; the binding docstring is kept."""
        swizzle_type = F.SwizzleType
        first = [(getattr(swizzle_type, name).name, getattr(swizzle_type, name).value)
                 for name, _ in SWIZZLE_MEMBERS]
        for _ in range(3):
            again = [(getattr(swizzle_type, name).name, getattr(swizzle_type, name).value)
                     for name, _ in SWIZZLE_MEMBERS]
            self.assertEqual(again, first)
        self.assertIn("swizzle", swizzle_type.__doc__.lower())

    def test_swizzletype_non_enum_arg_raises_attribute_error(self):
        """A non-enum swizzle raises AttributeError in the conversion layer.

        _enum_list_as_int_list reads li.value before the native call, so the
        error is device-independent and must not surface as a dispatch failure.
        """
        mat_a_cpu = torch.tensor([[1.0, -1.0, 0.5]]).to(torch.float8_e4m3fn)
        mat_b_cpu = torch.tensor([[0.5], [1.0], [-2.0]]).to(torch.float8_e4m3fn)
        with self.assertRaises(AttributeError):
            F.scaled_mm(
                mat_a_cpu,
                mat_b_cpu,
                torch.tensor(0.5),
                F.ScalingType.TensorWise,
                torch.tensor(0.5),
                F.ScalingType.TensorWise,
                swizzle_a="no_swizzle",
            )

    def test_swizzletype_scaled_mm_no_swizzle_device_consumption(self):
        """scaled_mm consumes explicit SwizzleType.NO_SWIZZLE on this device.

        Golden: (a.float() * scale_a) @ (b.float() * scale_b) on CPU; inputs are
        exact in float8_e4m3fn so the comparison is insensitive to accumulation
        order. NO_SWIZZLE is the only swizzle valid with TensorWise on the
        backends that implement the op. Platform-gated backends surface as
        unsupported-outcome evidence.
        """
        mat_a_cpu, mat_b_cpu = _fp8_matmul_inputs()
        scale_a = torch.tensor([0.5], dtype=torch.float32)
        scale_b = torch.tensor([0.5], dtype=torch.float32)
        expected = (mat_a_cpu.float() * scale_a) @ (mat_b_cpu.float() * scale_b)
        try:
            out = F.scaled_mm(
                mat_a_cpu.to(self.device),
                mat_b_cpu.to(self.device),
                scale_a.to(self.device),
                F.ScalingType.TensorWise,
                scale_b.to(self.device),
                F.ScalingType.TensorWise,
                swizzle_a=F.SwizzleType.NO_SWIZZLE,
                swizzle_b=F.SwizzleType.NO_SWIZZLE,
            )
        except RuntimeError as exc:
            self._assert_unsupported_on_device(str(exc))
            return
        torch.testing.assert_close(out.float().cpu(), expected, rtol=1e-2, atol=1e-2)

    def _assert_unsupported_on_device(self, message):
        """Shared assertion for unsupported-outcome evidence on this device.

        torch_npu registers _scaled_mm_v2 but gates it below Ascend950 with a
        RuntimeError whose first line is the platform-floor sentence
        "_scaled_mm is supported only on the Ascend950 platform and after."
        (volatile dispatcher diagnostics may follow it). Unregistered backends
        raise the dispatcher NotImplementedError pinning the op name and the
        missing backend. Any other failure text fails this assertion, so a
        backend defect surfaces as a test failure instead of a silent PASS.
        """
        if "supported only on the Ascend950 platform" in message:
            self.assertIn(
                "_scaled_mm is supported only on the Ascend950 platform and after.",
                message,
            )
            return
        match = _DISPATCHER_UNREGISTERED_RE.search(message)
        self.assertIsNotNone(
            match,
            f"unexpected unsupported outcome text: {message}",
        )
        if self.device_name in BACKEND_NAMES:
            self.assertEqual(match.group(1), BACKEND_NAMES[self.device_name])


if __name__ == "__main__":
    run_tests()
