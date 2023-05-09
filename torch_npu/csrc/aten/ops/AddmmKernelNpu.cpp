#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    at::Tensor &NPUNativeFunctions::addmm_out(
        const at::Tensor &self,
        const at::Tensor &mat1,
        const at::Tensor &mat2,
        const at::Scalar &beta,
        const at::Scalar &alpha,
        at::Tensor &result)
    {
      // mat1*alpha
      at::Tensor mulResult = at::mul(mat1, alpha);

      // mulmat1 mm mat2
      at::Tensor mmResult = at::mm(mulResult, mat2);

      // matmul*alpha+self*beta
      at::add_out(result, mmResult, self, beta);

      return result;
    }

    at::Tensor NPUNativeFunctions::addmm(
        const at::Tensor &self,
        const at::Tensor &mat1,
        const at::Tensor &mat2,
        const at::Scalar &beta,
        const at::Scalar &alpha)
    {
      // calculate the output size
      auto outputSize = addmm_npu_output_size(self, mat1, mat2, beta, alpha);

      // add算子支持NZ与1维且该轴能被16整除的ND相加，直接得到NZ result
      int64_t resFormat = (self.dim() == 1 && self.size(0) % 16 == 0 && self.scalar_type() == at::kHalf) ? ACL_FORMAT_FRACTAL_NZ : ACL_FORMAT_ND;
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(outputSize, self.options(), resFormat);

      // calculate the output result of the NPU
      NPUNativeFunctions::addmm_out(self, mat1, mat2, beta, alpha, result);

      return result;
    }

    at::Tensor &NPUNativeFunctions::addmm_(
        at::Tensor &self,
        const at::Tensor &mat1,
        const at::Tensor &mat2,
        const at::Scalar &beta,
        const at::Scalar &alpha)
    {
      c10::SmallVector<at::Tensor, N> inputs = {self, mat1, mat2};
      c10::SmallVector<at::Tensor, N> outputs = {self};
      CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);
      if (!NpuUtils::check_match(&self))
      {
        at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
        at::Tensor result =
            NPUNativeFunctions::addmm_out(contiguousSelf, mat1, mat2, beta, alpha, contiguousSelf);
        NpuUtils::format_fresh_view(self, result);
      }
      else
      {
        NPUNativeFunctions::addmm_out(self, mat1, mat2, beta, alpha, self);
      }

      return self;
    }

  } // namespace native
} // namespace at_npu
