#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::isfinite(const at::Tensor& self_ex) {
  at::Tensor self = self_ex;
  if (torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.npu_format_ !=
      ACL_FORMAT_ND) {
    self = NPUNativeFunctions::npu_format_cast(self_ex, ACL_FORMAT_ND);
  }
  if (self.scalar_type() == at::ScalarType::Half) {
    self = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }
  auto outputSize = self.sizes();
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, self.options().dtype(at::kBool), ACL_FORMAT_ND);
  OpCommand cmd;
  cmd.Name("IsFinite")
      .Input(self)
      .Output(result)
      .Run();
  return result;
}

} // namespace native
} // namespace at_npu