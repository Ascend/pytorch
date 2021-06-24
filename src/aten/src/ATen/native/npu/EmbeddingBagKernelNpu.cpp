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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

namespace {
SmallVector<int64_t, SIZE> _embedding_bag_npu_output_size(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets) {
  SmallVector<int64_t, SIZE> outputSize = {};
  if (indices.dim() == 1) {
    outputSize = {offsets.size(0), weight.size(1)};
  } else {
    outputSize = {indices.size(0), weight.size(1)};
  }
  return outputSize;
} // _embedding_bag_npu_output_size

string get_mode_str(bool mode) {
  string modeStr = "mean";
  if (mode == 0) {
    modeStr = "sum";
  } else if (mode == 1) {
    modeStr = "mean";
  } else {
    modeStr = "max";
  }
  return modeStr;
} // get_mode_str

} // namespace

tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_npu(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const Tensor& per_sample_weights,
    bool include_last_offset) {
  auto outputSize = _embedding_bag_npu_output_size(weight, indices, offsets);

  Tensor output = OpPreparation::ApplyTensorWithFormat(outputSize, weight.options(), ACL_FORMAT_ND);

  Tensor indicesCopy = indices;
  if (!(indices.dtype() == at::kInt)) {
    indicesCopy = indicesCopy.to(at::kInt);
  }

  string modeStr = get_mode_str(mode);

  OpCommand cmd;
  cmd.Name("EmbeddingBag")
      .Input(weight)
      .Input(indicesCopy);
  if (offsets.defined()) {
    Tensor offsetsCopy = offsets;
    if (!(offsets.dtype() == at::kInt)) {
      offsetsCopy = offsetsCopy.to(at::kInt);
    }
    cmd.Input(offsetsCopy);
  }
  if (per_sample_weights.defined()) {
    cmd.Input(per_sample_weights);
  }
  cmd.Output(output)
      .Attr("mode", modeStr)
      .Attr("scale_grad_by_freq", scale_grad_by_freq)
      .Attr("sparse", sparse)
      .Attr("include_last_offset", include_last_offset)
      .Run();
  
  return std::tie(output, output, output, output);
}

} // namespace native
} // namespace at