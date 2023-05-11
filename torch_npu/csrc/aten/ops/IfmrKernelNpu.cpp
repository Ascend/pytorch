#include <torch/csrc/autograd/custom_function.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
using namespace torch::autograd;

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::npu_ifmr(
    const at::Tensor& data,
    const at::Tensor& data_min,
    const at::Tensor& data_max,
    const at::Tensor& cumsum,
    const double min_percentile=0.999999,
    const double max_percentile=0.999999,
    const double search_start=0.7,
    const double search_end=1.3,
    const double search_step=0.01,
    const bool with_offset=true) {
  at::Tensor scale = OpPreparation::ApplyTensorWithFormat(data_min, ACL_FORMAT_NCHW);
  at::Tensor offset = OpPreparation::ApplyTensorWithFormat(data_min, ACL_FORMAT_NCHW);

  std::vector<float> tmp;
  tmp.push_back(static_cast<float>(search_start));
  tmp.push_back(static_cast<float>(search_end));
  at::ArrayRef<float> searchRange(tmp);

  OpCommand cmd;
  cmd.Name("IFMR")
      .Input(data)
      .Input(data_min)
      .Input(data_max)
      .Input(cumsum)
      .Attr("min_percentile", static_cast<float>(min_percentile))
      .Attr("max_percentile", static_cast<float>(max_percentile))
      .Attr("search_range", searchRange)
      .Attr("search_step", static_cast<float>(search_step))
      .Attr("with_offset", with_offset)
      .Output(scale)
      .Output(offset)
      .Run();

  return std::tie(scale, offset);
}

} // namespace native
} // namespace at_npu
