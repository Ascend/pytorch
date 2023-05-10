#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_expand_outplace(
    const at::Tensor &to_expand1,
    const at::Tensor &to_expand2,
    const at::Tensor &to_expand3,
    const char *api_name) {
  for (auto& t : {to_expand1, to_expand2, to_expand3}) {
    if (!t.defined()) {
      AT_ERROR(api_name, "(...) called with an undefined Tensor");
    }
  }

  if (to_expand1.sizes().equals(to_expand2.sizes()) && to_expand1.sizes().equals(to_expand3.sizes())) {
    return std::make_tuple(to_expand1, to_expand2, to_expand3);
  }

  auto expanded_size12 = broadcast_ops_npu_output_size(to_expand1, to_expand2);
  auto expanded_size = broadcast_ops_npu_output_size(expanded_size12, to_expand3.sizes());

  return std::make_tuple(
      to_expand1.expand(expanded_size, true),
      to_expand2.expand(expanded_size, true),
      to_expand3.expand(expanded_size, true));
}

at::Tensor& NPUNativeFunctions::where_out(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  at::Tensor self_, other_;
  if (self.dtype() != other.dtype()) {
    auto result_type = at::native::result_type(self, other);
    self_ = self.to(result_type);
    other_ = other.to(result_type);
  } else {
    self_ = self;
    other_ = other;
  }
  if (condition.scalar_type() != at::ScalarType::Byte && condition.scalar_type() != at::ScalarType::Bool) {
    AT_ERROR("Expected condition to have ScalarType Byte, but got ScalarType ",
        toString(condition.scalar_type()));
  }

  OpCommand cmd;
  cmd.Name("Select")
    .Input(condition)
    .Input(self_)
    .Input(other_)
    .Output(out)
    .Run();

  return out;
}

at::Tensor NPUNativeFunctions::where(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  at::Tensor b_condition, b_self, b_other;
  std::tie(b_condition, b_self, b_other) = npu_expand_outplace(condition, self, other, "where_npu");
  at::Tensor ret = OpPreparation::ApplyTensor(b_self);
  NPUNativeFunctions::where_out(b_condition, b_self, b_other, ret);
  return ret;
}

at::SmallVector<int64_t, SIZE> where_npu_output_size(const at::Tensor& condition){
  int64_t dim = condition.dim();
  at::Tensor boolSelf = NPUNativeFunctions::npu_dtype_cast(condition, at::ScalarType::Bool);
  at::Tensor intSelf = NPUNativeFunctions::npu_dtype_cast(boolSelf, at::ScalarType::Int);
  at::Tensor coutNonzeroSelf = at::sum(intSelf, at::ScalarType::Int);
  int64_t nonzeroNum = coutNonzeroSelf.item().toInt();
  at::SmallVector<int64_t, SIZE> outputSize = {nonzeroNum, dim};
  return outputSize;
}


vector<at::Tensor> NPUNativeFunctions::where(const at::Tensor& condition) {
  at::Tensor formatCastOfCondition = condition;
  if (torch_npu::NPUBridge::GetNpuStorageImpl(condition)->npu_desc_.npu_format_ !=
    ACL_FORMAT_ND) {
    formatCastOfCondition = NPUNativeFunctions::npu_format_cast(formatCastOfCondition, ACL_FORMAT_ND);
  }
  if (condition.scalar_type() == at::ScalarType::Half) {
    formatCastOfCondition = NPUNativeFunctions::npu_dtype_cast(formatCastOfCondition, at::ScalarType::Float);
  }

  // calculate the output size
  auto outputSize = where_npu_output_size(formatCastOfCondition);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, formatCastOfCondition.options().dtype(at::kLong), ACL_FORMAT_ND);

  OpCommand cmd;
  cmd.Name("NonZero")
    .Input(formatCastOfCondition)
    .Output(result)
    .Run();
  result = result.transpose(1, 0);
  std::vector<at::Tensor> chunkResult = result.chunk(result.size(0), 0);
  std::vector<at::Tensor> squeezeResult;
  for(int64_t i = 0; i < chunkResult.size(); i++){
    squeezeResult.push_back(chunkResult[i].squeeze(0));
  }

  return squeezeResult;
}

} // namespace native
} // namespace at_npu