#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& searchsorted_out_npu_nocheck(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    at::Tensor& result) {
  at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
  OpCommand cmd;
  cmd.Name("SearchSorted")
     .Input(sorted_sequence)
     .Input(self)
     .Attr("dtype", scalar_type)
     .Attr("right", right)
     .Output(result)
     .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::searchsorted_out(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<at::Tensor>& sorter_opt,
    at::Tensor& result) {
  at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
  OpPreparation::CheckOut(
      {sorted_sequence, self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(self),
      scalar_type,
      self.sizes());
  searchsorted_out_npu_nocheck(sorted_sequence, self, out_int32, right, result);
  return result;
}

at::Tensor NPUNativeFunctions::searchsorted(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<at::Tensor>& sorter_opt) {
  at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
  at::Tensor result = OpPreparation::ApplyTensor(self.sizes(), self.options().dtype(scalar_type), self);
  searchsorted_out_npu_nocheck(sorted_sequence, self, out_int32, right, result);
  return result;
}

at::Tensor NPUNativeFunctions::searchsorted(
    const at::Tensor& sorted_sequence,
    const at::Scalar& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<at::Tensor>& sorter_opt) {
  at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
  at::Tensor selfOp = CalcuOpUtil::CopyScalarToDevice(self, sorted_sequence.scalar_type());
  selfOp = selfOp.unsqueeze(0);
  at::Tensor result = OpPreparation::ApplyTensor({}, sorted_sequence.options().dtype(scalar_type), sorted_sequence);
  searchsorted_out_npu_nocheck(sorted_sequence, selfOp, out_int32, right, result);
  return result;
}
} // namespace native
} // namespace at_npu
