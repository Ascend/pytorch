#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
 
namespace at_npu {
namespace native {

at::Tensor& rotated_iou_npu_nocheck(
    at::Tensor& iou,
    const at::Tensor& boxes,
    const at::Tensor& query_boxes,
    bool trans,
    int64_t mode,
    bool is_cross,
    double v_threshold,
    double e_threshold) {
  string mode_str = (mode == 0) ? "iou" : "iof";   

  OpCommand cmd;
  cmd.Name("RotatedIou")
      .Input(boxes)
      .Input(query_boxes)
      .Output(iou)
      .Attr("trans", trans)
      .Attr("mode", mode_str)
      .Attr("is_cross", is_cross)
      .Attr("value", static_cast<float>(v_threshold))
      .Attr("value", static_cast<float>(e_threshold))
      .Run();
  return iou;
}

at::Tensor NPUNativeFunctions::npu_rotated_iou(
    const at::Tensor& boxes,
    const at::Tensor& query_boxes,
    bool trans,
    int64_t mode,
    bool is_cross,
    double v_threshold,
    double e_threshold) {
  TORCH_CHECK(boxes.ndimension() == 3 && query_boxes.ndimension() == 3);
      
  auto origin_dtype = boxes.scalar_type();
 
  at::Tensor boxesOk = boxes.permute({0, 2, 1});
  if (boxesOk.scalar_type() == at::kHalf){
    boxesOk = NPUNativeFunctions::npu_dtype_cast(boxesOk, at::kFloat);
  }
  at::Tensor query_boxesOk = query_boxes.permute({0, 2, 1});
  if (query_boxesOk.scalar_type() == at::kHalf){
    query_boxesOk = NPUNativeFunctions::npu_dtype_cast(query_boxesOk, at::kFloat);
  }

  int64_t B = boxesOk.size(0);
  int64_t N = boxesOk.size(-1);
  int64_t K = query_boxesOk.size(-1);
 
  c10::SmallVector<int64_t, SIZE> output_size({B, N, K});
  at::Tensor iou = OpPreparation::ApplyTensor(boxesOk, output_size);
 
  rotated_iou_npu_nocheck(iou, boxesOk, query_boxesOk, trans, mode, is_cross, v_threshold, e_threshold);
  iou = NPUNativeFunctions::npu_dtype_cast(iou, origin_dtype);
  return iou;
} 
} // namespace native
} // namespace at_npu