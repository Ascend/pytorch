#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& nonzero_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  c10::SmallVector<int64_t, N> output_sync_idx = {0};
  OpCommand cmd;
  cmd.Sync(output_sync_idx)
    .Name("NonZero")
    .Input(self)
    .Output(result)
    .Attr("transpose", false)
    .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::nonzero_out(const at::Tensor& self, at::Tensor& result) {
  auto outputSize = nonzero_npu_max_output_size(self);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(self),
      at::ScalarType::Long,
      outputSize);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self](at::Tensor& result){nonzero_out_npu_nocheck(result, self);})
   .Call(result);
}

at::Tensor NPUNativeFunctions::nonzero(const at::Tensor& self) {
  // calculate the output size
  auto outputSize = nonzero_npu_max_output_size(self);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(
      outputSize, self.options().dtype(at::kLong), self);

  // calculate the output result of the NPU
  nonzero_out_npu_nocheck(result, self);
  return result;
}

} // namespace native
} // namespace at_npu
