#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::npu_iou(
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
    int64_t mode) {
  at::Tensor bboxesFP16 = bboxes;
  if (bboxes.scalar_type() != at::ScalarType::Half) {
    bboxesFP16 = NPUNativeFunctions::npu_dtype_cast(bboxes, at::kHalf);
  }
  at::Tensor gtboxesFP16 = gtboxes;
  if (gtboxes.scalar_type() != at::ScalarType::Half) {
    gtboxesFP16 = NPUNativeFunctions::npu_dtype_cast(gtboxes, at::kHalf);
  }

  auto outputSize = {gtboxes.size(0), bboxes.size(0)};
  at::Tensor overlap = OpPreparation::ApplyTensorWithFormat(
      bboxesFP16,
      outputSize,
      CalcuOpUtil::GetTensorNpuFormat(bboxes));
  string modeStr = "iou";
  if (mode == 1) {
    modeStr = "iof";
  }
  OpCommand cmd;
  cmd.Name("Iou")
      .Input(bboxesFP16)
      .Input(gtboxesFP16)
      .Output(overlap)
      .Attr("mode", modeStr)
      .Attr("eps", static_cast<float>(0.01))
      .Run();
  if (overlap.scalar_type() != bboxes.scalar_type()) {
    overlap = NPUNativeFunctions::npu_dtype_cast(overlap, bboxes.scalar_type());
  }
  return overlap;
}

at::Tensor NPUNativeFunctions::npu_ptiou(
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
    int64_t mode) {
  return NPUNativeFunctions::npu_iou(bboxes, gtboxes, mode);
}

} // namespace native
} // namespace at_npu