#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& range_out_nocheck(
    at::Scalar start,
    at::Scalar end,
    at::Scalar step,
    at::Tensor& result) {
  // generate x assistant tensor
  int value = result.size(0);
  c10::SmallVector<int64_t, N> tmp_vector = {};
  for (int i = 0; i < value; i++) {
    tmp_vector.emplace_back(i);
  }

  OpCommand cmd;
  cmd.Name("RangeD")
     .Input(tmp_vector, result.scalar_type())
     .Output(result)
     .Attr("start", start)
     .Attr("limit", end)
     .Attr("delta", step)
     .Run();

  return result;
}

at::Tensor NPUNativeFunctions::range(
    const at::Scalar& start,
    const at::Scalar& end,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  c10::TensorOptions option = c10::TensorOptions().dtype(dtype_opt)
                                          .device(device_opt)
                                          .layout(layout_opt)
                                          .pinned_memory(pin_memory_opt);
  return at::range(start, end, 1, option);
}

at::Tensor NPUNativeFunctions::range(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  c10::TensorOptions option = c10::TensorOptions().dtype(dtype_opt)
                                          .device(device_opt)
                                          .layout(layout_opt)
                                          .pinned_memory(pin_memory_opt);

  float start_value = CalcuOpUtil::GetScalarFloatValue(start);
  float end_value = CalcuOpUtil::GetScalarFloatValue(end);
  float step_value = CalcuOpUtil::GetScalarFloatValue(step);

  TORCH_CHECK(step_value > 0 || step_value < 0, "step must be nonzero");
  TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) || ((step_value < 0) && (end_value <= start_value)),
      "upper bound and larger bound inconsistent with step sign");

  auto outputSize = range_npu_output_size(start_value, end_value, step_value);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(outputSize, option, ACL_FORMAT_NCHW);
  return range_out_nocheck(start, end, step, result);
}

at::Tensor& NPUNativeFunctions::range_out(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& result) {
  float start_value = CalcuOpUtil::GetScalarFloatValue(start);
  float end_value = CalcuOpUtil::GetScalarFloatValue(end);
  float step_value = CalcuOpUtil::GetScalarFloatValue(step);

  TORCH_CHECK(step_value > 0 || step_value < 0, "step must be nonzero");
  TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) || ((step_value < 0) && (end_value <= start_value)),
      "upper bound and larger bound inconsistent with step sign");

  auto outputSize = range_npu_output_size(start_value, end_value, step_value);
  OpPreparation::CheckOut(
      { },
      result,
      ACL_FORMAT_NCHW,
      result.scalar_type(),
      outputSize);

  return range_out_nocheck(start, end, step, result);
}

} // namespace native
} // namespace at