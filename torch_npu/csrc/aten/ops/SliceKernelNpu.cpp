#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    at::Tensor &NPUNativeFunctions::npu_slice_out(
        const at::Tensor &self,
        c10::IntArrayRef offsets,
        c10::IntArrayRef size,
        at::Tensor &result)
    {

      c10::SmallVector<int64_t, N> offsetVec = array_to_small_vector(offsets);
      c10::SmallVector<int64_t, N> sizeVec = array_to_small_vector(size);
      OpCommand cmd;
      cmd.Name("Slice")
          .Input(self)
          .Input(offsetVec)
          .Input(sizeVec)
          .Output(result)
          .Run();
      return result;
    }

    at::Tensor NPUNativeFunctions::npu_slice(const at::Tensor &self, c10::IntArrayRef offsets, c10::IntArrayRef size)
    {
      // calculate the output size
      c10::SmallVector<int64_t, SIZE> outputSize =
          CalcuOpUtil::ConvertIntArrayRefToSmallVector(size);
      // construct the output at::Tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);

      // calculate the output result of the NPU
      npu_slice_out(self, offsets, size, result);

      return result;
    }

  } // namespace native
} // namespace at