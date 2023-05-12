#include "torch_npu/csrc/framework/utils/OpPipe.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace at_npu
{
  namespace native
  {

    OpPipeWithDefinedOut &OpPipeWithDefinedOut::CheckMemory(const std::initializer_list<at::Tensor> &inputs, const std::initializer_list<at::Tensor> &outputs)
    {
      OpPreparation::CheckMemory(inputs, outputs);
      return *this;
    }

    at::Tensor &OpPipeWithDefinedOut::Call(at::Tensor &dst)
    {
      if (!NpuUtils::check_match(&dst))
      {
        at::Tensor contigTensor = NpuUtils::format_contiguous(dst);
        this->func(contigTensor);
        NpuUtils::format_fresh_view(dst, contigTensor);
      }
      else
      {
        this->func(dst);
      }
      return dst;
    }

    OpPipeWithApplyOut &OpPipeWithApplyOut::ApplyOutputSameAs(const at::Tensor &src)
    {
      this->dst = OpPreparation::ApplyTensor(src);
      return *this;
    }

    at::Tensor &OpPipeWithApplyOut::Call()
    {
      this->func(this->dst);
      return this->dst;
    }

  } // namespace native
} // namespace at_npu