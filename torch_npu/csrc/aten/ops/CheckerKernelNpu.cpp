#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

void NPUNativeFunctions::check_memory_overlaps(at::TensorList inputs, at::TensorList outputs) {
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);
}

bool NPUNativeFunctions::check_match(const at::Tensor& self) {
  return NpuUtils::check_match(&self);
}

}
} // namespace at_npu
