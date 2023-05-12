#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::put_(
    at::Tensor& self,
    const at::Tensor& index,
    const at::Tensor& source,
    bool accumulate) {
  TORCH_CHECK(index.numel() == source.numel(), "source should have the same number of elements as index");
  if (source.numel() == 0) {
    return self;
  }
  c10::SmallVector<at::Tensor, N> inputs = {self};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);

  at::Tensor selfFlatten = NpuUtils::format_contiguous(self.reshape(-1));
  at::Tensor indexFlatten = index.reshape({-1, 1});
  at::Tensor sourceFlatten = source.reshape(-1);

  OpCommand cmd;
  accumulate ? cmd.Name("ScatterNdAdd") : cmd.Name("ScatterNdUpdate");
  cmd.Input(selfFlatten)
     .Input(indexFlatten)
     .Input(sourceFlatten)
     .Output(selfFlatten)
     .Attr("use_locking", false)
     .Run();

  self.copy_(selfFlatten);
  return self;
}
}  // namespace native
}  // namespace at_npu