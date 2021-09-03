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

Tensor confusion_transpose_npu(
    const Tensor& self,
    IntArrayRef perm,
    IntArrayRef shape,
    bool transpose_first) {
  SmallVector<int64_t, SIZE> output_size;
  if (transpose_first){
    output_size = array_to_small_vector(shape);
  } else {
    for (int i = 0; i < perm.size(); i++){
      output_size.emplace_back(shape[perm[i]]);
    }
  }

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self, output_size);
  OpCommand cmd;
  cmd.Name("ConfusionTransposeD")
      .Input(self)
      .Output(result)
      .Attr("perm", perm)
      .Attr("shape", shape)
      .Attr("transpose_first", transpose_first)
      .Run();

  return result;
}

} // namespace native
} // namespace at