#include <ATen/record_function.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {
    at::Tensor &NPUNativeFunctions::npu_transpose_out(
        const at::Tensor &self,
        at::IntArrayRef perm,
        bool require_contiguous,
        at::Tensor &result
        )
    {
      OpCommand cmd;
      if (require_contiguous) {
        // Any tensor-view(discontiguous) Input Tensor from users should be transformed to be contiguous here.
        cmd.Name("Transpose")
          .Input(self)
          .Input(perm)
          .Output(result)
          .Run();
      } else {
      // For permute-opt in trans-contiguous, it accepts transposed(discontiguous) Input Tensor.
      cmd.Name("Transpose")
          .InputWithoutContiguous(self)
          .Input(perm)
          .Output(result)
          .Run();
      }
      return result;
    }

    at::Tensor NPUNativeFunctions::npu_transpose(const at::Tensor &self, at::IntArrayRef perm, bool require_contiguous)
    {
      auto outputSize = transpose_npu_output_size(self, perm);
      at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
      NPUNativeFunctions::npu_transpose_out(self, perm, require_contiguous, result);

      return result;
    }


  } // namespace native
} // namespace at_npu