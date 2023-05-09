#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
namespace {
c10::SmallVector<int64_t, SIZE> diag_npu_output_size(
    const at::Tensor& self,
    int64_t diagonal) {
  c10::SmallVector<int64_t, SIZE> shape;
  if (self.dim() == 1) {
    shape.emplace_back(self.size(0) + diagonal);
    shape.emplace_back(self.size(0) + diagonal);
    return shape;
  }
  int64_t m = self.size(0);
  int64_t n = self.size(1);
  if (m == n) {
    shape.emplace_back(m - diagonal);
  } else if (m < n) {
    shape.emplace_back(diagonal <= n - m ? m : n - diagonal);
  } else {
    shape.emplace_back(n - diagonal);
  }
  return shape;
}
} // namespace

at::Tensor& diag_out_npu_nocheck(
    const at::Tensor& self, 
    int64_t diagonal,
    at::Tensor& result) {
  OpCommand cmd;
  if (self.dim() == 1) {
    cmd.Name("Diag");
  } else {
    cmd.Name("DiagPart");
  }
  cmd.Input(self)
    .Output(result)
    .Attr("diagonal", diagonal)
    .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::diag_out(
    const at::Tensor& self, 
    int64_t diagonal, 
    at::Tensor& result) {
  TORCH_CHECK((self.dim() == 1) || (self.dim() == 2),
              "Value should be a 1-dimensional tensor or 2-dimensional tensor, but got ", self.dim());
  diagonal = make_wrap_dim(diagonal, self.dim());
  TORCH_CHECK((self.dim() == 1) || (self.dim() == 2 && diagonal <= self.size(0) && diagonal <= self.size(1)),
              "If the value is 2-dimensional tensor, the diagonal shoule less than shape.Diagonal is ", diagonal);

  auto outputSize = diag_npu_output_size(self, diagonal);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(self),
      self.scalar_type(),
      outputSize);
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self, &diagonal](at::Tensor& result){diag_out_npu_nocheck(self, diagonal, result);})
   .Call(result);
}

at::Tensor NPUNativeFunctions::diag(
    const at::Tensor& self, 
    int64_t diagonal) {
  TORCH_CHECK((self.dim() == 1) || (self.dim() == 2),
              "Value should be a 1-dimensional tensor or 2-dimensional tensor, but got ", self.dim());
  diagonal = make_wrap_dim(diagonal, self.dim());
  TORCH_CHECK((self.dim() == 1) || (self.dim() == 2 && diagonal <= self.size(0) && diagonal <= self.size(1)),
              "If the value is 2-dimensional tensor, the diagonal shoule less than shape.Diagonal is ", diagonal);

  auto outputSize = diag_npu_output_size(self, diagonal);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  diag_out_npu_nocheck(self, diagonal, result);
  return result;
}

} // namespace native
} // namespace at_npu
