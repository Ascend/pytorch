#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& bounding_box_encode_npu_nocheck(
    const at::Tensor& anchor_box,
    const at::Tensor& ground_truth_box,
    c10::SmallVector<float, SIZE> means,
    c10::SmallVector<float, SIZE> stds,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("BoundingBoxEncode")
       .Input(anchor_box)
       .Input(ground_truth_box)
       .Output(result)
       .Attr("means", means)
       .Attr("stds", stds)
       .Run();
  return result;
}

at::Tensor NPUNativeFunctions::npu_bounding_box_encode(
    const at::Tensor& anchor_box,
    const at::Tensor& ground_truth_box,
    double means0,
    double means1,
    double means2,
    double means3,
    double stds0,
    double stds1,
    double stds2,
    double stds3) {
  at::Tensor result = OpPreparation::ApplyTensor(anchor_box, {anchor_box.size(0), 4});
  c10::SmallVector<float, SIZE> means = {
      static_cast<float>(means0),
      static_cast<float>(means1),
      static_cast<float>(means2),
      static_cast<float>(means3)};
  c10::SmallVector<float, SIZE> stds = {
      static_cast<float>(stds0),
      static_cast<float>(stds1),
      static_cast<float>(stds2),
      static_cast<float>(stds3)};
  bounding_box_encode_npu_nocheck(
      anchor_box, ground_truth_box, means, stds, result);
  return result;
}

} // namespace native
} // namespace at_npu
