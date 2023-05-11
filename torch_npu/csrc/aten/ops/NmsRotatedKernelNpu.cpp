#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
tuple<at::Tensor, at::Tensor> NPUNativeFunctions::npu_nms_rotated(const at::Tensor& dets, const at::Tensor& scores,
                                                                  double iouThreshold, double scoreThreshold,
                                                                  int64_t maxOutputSize, int64_t mode) {
  // the Op only support fp32 currently!
  auto originDtype = dets.scalar_type();
  at::Tensor detsCast = dets;
  at::Tensor scoresCast = scores;
  at::Tensor labels = at::zeros({}, scores.options().dtype(at::kInt));
  if (originDtype != at::ScalarType::Float) {
    detsCast = NPUNativeFunctions::npu_dtype_cast(dets, at::kFloat);
    scoresCast = NPUNativeFunctions::npu_dtype_cast(scores, at::kFloat);
  }
  c10::SmallVector<int64_t, SIZE> selectedIndexSize = {dets.size(0)};
  at::Tensor selectedBox = OpPreparation::ApplyTensor(detsCast);
  at::Tensor selectedIndex = OpPreparation::ApplyTensor(selectedIndexSize, dets.options().dtype(at::kInt), dets);

  c10::SmallVector<int64_t, N> output_sync_idx = {0, 1};
  OpCommand cmd;
  cmd.Sync(output_sync_idx)
      .Name("RotatedNMS")
      .Input(detsCast)
      .Input(scoresCast)
      .Input(labels)
      .Output(selectedBox)
      .Output(selectedIndex)
      .Attr("iou_threshold", (float)iouThreshold)
      .Attr("score_threshold", (float)scoreThreshold)
      .Attr("max_output_size", maxOutputSize)
      .Attr("mode", mode)
      .Run();

  at::Tensor selectedNum =
      OpPreparation::ApplyTensor({1}, scores.options().dtype(at::kInt), scores).fill_(selectedIndex.size(0));
  return std::tie(selectedIndex, selectedNum);
}

}  // namespace native
}  // namespace at_npu