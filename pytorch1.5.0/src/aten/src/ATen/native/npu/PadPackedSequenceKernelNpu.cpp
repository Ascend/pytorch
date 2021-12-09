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

#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

std::tuple<Tensor, Tensor> _pad_packed_sequence_npu(const Tensor& data, const Tensor& _batchSizes, bool batchFirst, Scalar paddingValue, int64_t totalLength) {
  Tensor output = data;
  auto batchSizesT = _batchSizes.contiguous();

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

  at::Tensor lengthsT = at::empty(maxBatchSize, batchSizesT.options());
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

}}
