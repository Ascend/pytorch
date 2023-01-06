// Copyright (c) 2020 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::_pad_packed_sequence(
    const at::Tensor& input,
    const at::Tensor& batchSizes,
    bool batchFirst,
    const at::Scalar& paddingValue,
    int64_t totalLength) {
  if (totalLength > 0) {
    TORCH_CHECK(totalLength >= batchSizes.size(0),
        "Expected total_length to be at least the length of the longest "
        "sequence in input, but got total_length=", totalLength, " and "
        "max sequence length being ", batchSizes.size(0));
  }

  // 输入shape为[B*T, *], 计算B和T
  auto batchSizesCpu = batchSizes.to("cpu");
  int64_t* batchSizeVec = batchSizesCpu.data_ptr<int64_t>();
  auto batchsize = batchSizeVec[0];
  auto timesize = batchSizes.size(0);

  // 构造输出pad后的tensor, [B, T, *] 或 [T, B, *]
  at::SmallVector<int64_t, N> shape;
  shape.emplace_back(timesize);
  shape.emplace_back(batchsize);

  for (int i = 1; i < input.dim(); i++) {
    shape.emplace_back(input.size(i));
  }

  auto output = input.reshape(shape);

  if (batchFirst) {
    output = output.transpose(0,1);
  }
  output = output.contiguous();

  // 构造输出timesizes
  auto batchsizes = at::empty({batchsize}, batchSizesCpu.options());
  auto batchsizesVec = batchsizes.data_ptr<int64_t>();
  int64_t last = timesize - 1;
  for (int bi = 0; bi < batchsize; bi++) {
    for (int ti = last; ti >= 0; ti--) {
      if (batchSizeVec[ti] > bi ) {
        batchsizesVec[bi] = (ti + 1);
        last = ti;
        break;
      }
    }
  }

  return std::tie(output, batchsizes);
}

} // namespace native
} // namespace at_npu
