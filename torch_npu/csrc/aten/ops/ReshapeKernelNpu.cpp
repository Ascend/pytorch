#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPURunMode.h"
#include <c10/core/TensorImpl.h>

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::npu_reshape_out(
    const at::Tensor& src,
    at::IntArrayRef shape,
    bool can_refresh,
    at::Tensor& result) {
  if (c10_npu::NpuRunMode::IsGraphMode()) {
    std::vector<int64_t> out_strides = at::detail::defaultStrides(shape);
    if (result.sizes() != shape || result.strides() != out_strides) {
      auto allow_flag =
          result.unsafeGetTensorImpl()->allow_tensor_metadata_change();
      result.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(true);
      StorageDescHelper::SetDesc(result, shape, out_strides);
      result.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(allow_flag);
    }

    OpCommand cmd;
    cmd.Name("Reshape")
        .InputWithoutContiguous(src)
        .Input(shape, at::kLong)
        .Output(result)
        .Run();
  } else if (can_refresh) {
    StorageDescHelper::SetDesc(
        result,
        array_to_small_vector(result.sizes()),
        array_to_small_vector(result.strides()));
  } else {
    copy_d2d_by_memcpy(
        result,
        src,
        c10::multiply_integers(torch_npu::NPUBridge::GetNpuStorageImpl(result)->get_npu_desc().storage_sizes_));
  }
  return result;
}

at::Tensor NPUNativeFunctions::npu_reshape(const at::Tensor& self, at::IntArrayRef shape, bool can_refresh) {
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      shape, self.options(), CalcuOpUtil::GetTensorNpuFormat(self));
  NPUNativeFunctions::npu_reshape_out(self, shape, can_refresh, result);

  return result;
}

} // namespace native
} // namespace at_npu