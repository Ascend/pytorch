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
    const at::Tensor& data,
    const at::Tensor& _batchSizes,
    bool batchFirst,
    at::Scalar paddingValue,
    int64_t totalLength) {
  at::Tensor output = data;
  auto batchSizesT = _batchSizes.contiguous().to("cpu");

  int64_t * batchSizes = batchSizesT.data_ptr<int64_t>();
  int64_t maxBatchSize = batchSizes[0];
  int64_t maxRealSeqLength = batchSizesT.size(0);
  int64_t maxSeqLength = maxRealSeqLength;
  if (totalLength > 0) {
    TORCH_CHECK(totalLength >= maxSeqLength,
        "Expected total_length to be at least the length of the longest "
        "sequence in input, but got total_length=", totalLength, " and "
        "max sequence length being ", maxSeqLength);
    maxSeqLength = totalLength;
  }

  at::Tensor lengthsT = at::empty(maxBatchSize, batchSizesT.options().device(at::kCPU));
  int64_t * lengths = lengthsT.data_ptr<int64_t>() + maxBatchSize - 1;
  int64_t prevBatchSize = maxBatchSize;
  for (int64_t i = 0; i <= maxRealSeqLength; ++i) {
    int64_t batchSize = i != maxRealSeqLength ? batchSizes[i] : 0;
    int64_t dec = prevBatchSize - batchSize;
    if (dec > 0) {
      for (int64_t j = 0; j < dec; ++j) {
        *lengths = i;
        lengths--;
      }
    }
    prevBatchSize = batchSize;
  }
  if (batchFirst) {
    output = data.transpose(0, 1);
  }
  return std::tie(output, lengthsT);
}

} // namespace native
} // namespace at_npu