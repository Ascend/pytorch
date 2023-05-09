#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& channel_shuffle_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, int64_t groups) {
  OpCommand cmd;
  cmd.Name("ShuffleChannel")
     .Input(self)
     .Output(result)
     .Attr("group", groups)
     .Run();
  return result;
}

at::Tensor NPUNativeFunctions::channel_shuffle(const at::Tensor& self, int64_t groups) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  channel_shuffle_out_npu_nocheck(result, self, groups);
  return result;
}

} // namespace native
} // namespace at_npu
