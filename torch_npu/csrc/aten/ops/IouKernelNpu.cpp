// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
      CalcuOpUtil::get_tensor_npu_format(bboxes));
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