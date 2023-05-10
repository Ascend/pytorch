#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& quantize_per_tensor_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    at::ScalarType dtype) {
  string dtypeStr = "torch.qint8";
  if (dtype == at::ScalarType::QUInt8) {
    dtypeStr = "torch.quint8";
  } else if (dtype == at::ScalarType::QInt32) {
    dtypeStr = "torch.qint32";
  }
  OpCommand cmd;
  cmd.Name("Quantize")
     .Input(self)
     .Input(scales)
     .Input(zero_points)
     .Output(result)
     .Attr("axis", (int64_t)1)
     .Attr("dtype", dtypeStr)
     .Run();

  return result;
}

at::Tensor NPUNativeFunctions::quantize_per_tensor(
    const at::Tensor& self,
    double scale,
    int64_t zero_point,
    at::ScalarType dtype) {
  float scaleFloat = static_cast<float>(scale);
  auto outputSize = input_same_output_size(self);
  auto outputDtype = at::kInt;
  if (dtype == at::ScalarType::QInt8) {
    outputDtype = at::kChar;
  } else if (dtype == at::ScalarType::QUInt8) {
    outputDtype = at::kByte;
  } else if (dtype == at::ScalarType::QInt32) {
    outputDtype = at::kInt;
  }
  at::Tensor scaleTensor = OpPreparation::ApplyTensorWithFormat(
      {1},
      self.options().dtype(at::kFloat),
      CalcuOpUtil::GetTensorNpuFormat(self));
  scaleTensor[0] = scaleFloat;
  at::Tensor zpTensor = OpPreparation::ApplyTensorWithFormat(
      {1},
      self.options().dtype(at::kInt),
      CalcuOpUtil::GetTensorNpuFormat(self));
  zpTensor[0] = zero_point;
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      self.options().dtype(outputDtype),
      CalcuOpUtil::GetTensorNpuFormat(self));
  quantize_per_tensor_out_nocheck(result, self, scaleTensor, zpTensor, dtype);
  return result;
}

} // namespace native
} // namespace at_npu