#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "c10/npu/npu_log.h"
#include <iostream>

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> det_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
    return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> det_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {

  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> det_npu_attr(const Tensor& self) {
    SmallVector<NPUAttrDesc, N> attrs = {};

  return attrs;
}

Tensor& det_out_npu(Tensor& result, const Tensor& self) {
  // constructs the input and output NPUTensorDesc
  auto inputs = det_npu_input({self});
  auto outputs = det_npu_output({result});

  // constructs the attr of the NPUAttrDesc
  auto attrs = det_npu_attr(self);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Det", inputs, outputs, attrs);

  return result;
}

Tensor det_npu(const Tensor& self) {
  // calculate the output size

  auto outputSize = det_npu_output_size(self);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
  outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  // calculate the output result of the NPU
  det_out_npu(result, self);
  return result;
}
} // namespace native
} // namespace at
