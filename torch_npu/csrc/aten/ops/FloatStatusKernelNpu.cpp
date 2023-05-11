#include "torch_npu/csrc/framework/graph/util/GraphModeGuard.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"

namespace at_npu {
namespace native {

const int FLOAT_STATUS_OP_DIMS_SIZE = 8;

at::Tensor NPUNativeFunctions::npu_alloc_float_status(const at::Tensor& self) {
  auto options = at::TensorOptions(c10::DeviceType::PrivateUse1).dtype(at::kFloat);
  at::Tensor result = at::empty({FLOAT_STATUS_OP_DIMS_SIZE}, options);
  OpCommand cmd;
  cmd.Name("NPUAllocFloatStatus")
      .Output(result)
      .Run();

  return result;
}

at::Tensor NPUNativeFunctions::npu_get_float_status(const at::Tensor& self) {
  OpCommand cmd;
  if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1) {
    at::Tensor out_tensor = at::empty({FLOAT_STATUS_OP_DIMS_SIZE}, self.options().dtype(at::kInt));
    cmd.Name("NPUGetFloatStatusV2")
      .Output(out_tensor)
      .Run();

    return out_tensor;
  } else {
    at::Tensor out_tensor = at::empty({FLOAT_STATUS_OP_DIMS_SIZE}, self.options());
    cmd.Name("NPUGetFloatStatus")
      .Input(self)
      .Output(out_tensor)
      .Run();

    return self;
  }
}

at::Tensor NPUNativeFunctions::npu_clear_float_status(const at::Tensor& self) {
  GraphModeGuard mode_guard(c10_npu::ModeKind::SINGLE_OP_MODE);
  at::Tensor result = at::empty({FLOAT_STATUS_OP_DIMS_SIZE}, self.options());
  OpCommand cmd;
  if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1) {
    cmd.Name("NPUClearFloatStatusV2")
      .Run();
  } else {
    cmd.Name("NPUClearFloatStatus")
      .Input(self)
      .Output(result)
      .Run();
  }
  return result;
}

} // namespace native
} // namespace at_npu
