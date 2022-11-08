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

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::_pack_padded_sequence(
    const at::Tensor& input,
    const at::Tensor& lengths,
    bool batch_first) {
  // 获取B、T的大小，输入是[T, B, *] or [B, T, *]
  auto batchsize = batch_first ? input.size(0) : input.size(1);
  auto timesize = batch_first ? input.size(1) : input.size(0);

  // 与CPU对齐输入条件
  TORCH_CHECK(input.numel() > 0, "Cannot pack empty tensors.");

  TORCH_CHECK(lengths.size(0) == batchsize,
      "Expected 'len(lengths)' to be equal to batch_size, but got ", lengths.size(0),
      " (batch_size=", batchsize, ")");

  auto lengthsVec = lengths.contiguous().data_ptr<int64_t>();
  TORCH_CHECK(lengthsVec[batchsize - 1] > 0,
      "Length of all samples has to be greater than 0, but found an element "
      "in 'lengths' that is <= 0");

  // 根据TMG决策暂时采用适配规避方案一，保留有效T0内的填充, [B*T0, *]
  auto output = batch_first ? input.transpose(0, 1) : input;
  auto len = lengthsVec[0];
  if (len < timesize) {
    vector<int> tmp_vector = {};
    for (int i = 0; i < len; i++) {
      tmp_vector.emplace_back(i);
    }
    auto index = at::from_blob(tmp_vector.data(), {len}, at::kInt);
    index = CalcuOpUtil::copy_tensor_host_to_device(index);
    output = NPUNativeFunctions::index_select(output, 0, index);
    timesize = len;
  }

  at::SmallVector<int64_t, N> shape;
  shape.emplace_back(batchsize * timesize);
  for (int i = 2; i < input.dim(); i++) {
    shape.emplace_back(input.size(i));
  }

  output = output.contiguous();
  output = output.reshape(shape);

  // 计算输出的batch_size
  at::Tensor batchsizes = at::empty({timesize}, lengths.options());
  auto batchsizeVec = batchsizes.data_ptr<int64_t>();
  int64_t last = batchsize - 1;
  for (int ti = 0; ti < timesize; ti++) {
    for (int bi = last; bi >= 0 ; bi--) {
      if (lengthsVec[bi] > ti ) {
        batchsizeVec[ti] = (bi + 1);
        last = bi;
        break;
      }
    }
  }

  return std::tie(output, batchsizes);
}

}
}
