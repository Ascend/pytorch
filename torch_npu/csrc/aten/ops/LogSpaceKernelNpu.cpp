#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& logspace_out_npu_nocheck(
    at::Scalar start,
    at::Scalar end,
    int64_t steps,
    double base,
    at::Tensor& result) {
  if (steps < 0){
    TORCH_CHECK("please input steps > 0");
  }
  if (base <= 0) {
    printf("if base<=0, please input intenger start, end, (end-start)/(steps-1)");
  }
  at::Tensor inputs;
  if (result.scalar_type() == at::ScalarType::Half) {
    inputs = NPUNativeFunctions::npu_dtype_cast(at::arange(0, steps, at::device(c10::DeviceType::PrivateUse1)), at::kHalf);
  } else if (result.scalar_type() == at::ScalarType::Float) {
    inputs = at::arange(0, steps, at::device(c10::DeviceType::PrivateUse1).dtype(at::kFloat));
  }
  int64_t dtype = 0;
  if (result.scalar_type() == at::ScalarType::Half) {
    dtype = 0;
  } else if (result.scalar_type() == at::ScalarType::Float) {
    dtype = 1;
  } else {
    TORCH_CHECK("only support float32 and float16");
  }
  OpCommand cmd;
  cmd.Name("LogSpaceD")
      .Input(inputs)
      .Output(result)
      .Attr("start", start)
      .Attr("end", end)
      .Attr("steps", steps)
      .Attr("base", static_cast<float>(base))
      .Attr("dtype", dtype)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::logspace_out(
    const at::Scalar& start,
    const at::Scalar& end,
    int64_t steps,
    double base,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      { },
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      {steps});
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({},{result})
    .Func([&start, &end, &steps, &base](at::Tensor& result)
    {logspace_out_npu_nocheck(start, end, steps, base, result);})
    .Call(result);
  return result;
}

at::Tensor NPUNativeFunctions::logspace(
    const at::Scalar& start,
    const at::Scalar& end,
    int64_t steps,
    double base,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  auto device = device_or_default(device_opt);
  at::TensorOptions options;
  options = options.dtype(dtype_opt)
                   .layout(layout_opt)
                   .device(device)
                   .pinned_memory(pin_memory_opt);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat({steps}, options, ACL_FORMAT_ND);
  return logspace_out_npu_nocheck(start, end, steps, base, result);
}
}
}
