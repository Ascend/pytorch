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

std::tuple<Tensor, Tensor> _pack_padded_sequence_npu(const Tensor& _input, const Tensor& _lengths, bool batch_first) {
  auto input = batch_first ? _input.transpose(0, 1) : _input;
  auto lengths_t = _lengths.contiguous();

  int64_t batchSize = input.size(1);
  int64_t * lengths = lengths_t.data_ptr<int64_t>();
  TORCH_CHECK(input.numel() > 0, "Cannot pack empty tensors.");
  TORCH_CHECK(lengths_t.size(0) == batchSize,
      "Expected `len(lengths)` to be equal to batch_size, but got ", lengths_t.size(0),
      " (batch_size=", batchSize, ")");
  TORCH_CHECK(lengths[batchSize - 1] > 0,
      "Length of all samples has to be greater than 0, but found an element "
      "in 'lengths' that is <= 0");
  for(auto i = 0; i < batchSize - 1; i++) {
    if (lengths[batchSize - 1 - i] > lengths[batchSize - 2 - i]) {
      // NB: enforce_sorted is implemented at a Python level, but the sortedness
      // check lives here. If enforce_sorted=False then this error should never
      // get called.
      AT_ERROR("`lengths` array must be sorted in decreasing order when "
               "`enforce_sorted` is True. You can pass `enforce_sorted=False` "
               "to pack_padded_sequence and/or pack_sequence to sidestep this "
               "requirement if you do not need ONNX exportability.");
    }
  }

  at::Tensor batchSizesT = at::empty(lengths[0], _lengths.options());
  int64_t * batchSizes = batchSizesT.data_ptr<int64_t>();

  int64_t prevL = 0;
  for (int64_t i = 0; i < batchSize; ++i) {
    int64_t l = lengths[batchSize - 1 - i];
    if (l > prevL) {
      auto currentBatchSize = batchSize - i;
      for (int64_t j = 0; j < (l - prevL); ++j) {
        *batchSizes = currentBatchSize;
        batchSizes++;
      }
      prevL = l;
    }
    TORCH_CHECK(l >= prevL);
  }
  
  // input must have 2 dim for  rnn
  int64_t lastDim = _input.size(2);
  Tensor inputDim2 = _input.contiguous().view({-1, lastDim});
  
  return std::tie(inputDim2, batchSizesT);
}

}}
