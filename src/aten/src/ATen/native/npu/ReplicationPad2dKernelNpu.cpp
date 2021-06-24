// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
//
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"

namespace at {
namespace native {
using namespace at::native::npu;
SmallVector<NPUTensorDesc, N> replication_pad2d_npu_input(SmallVector<Tensor, N> inputs) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputs);
}

SmallVector<NPUTensorDesc, N> replication_pad2d_npu_output(const SmallVector<Tensor, N> &outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}
SmallVector<NPUAttrDesc, N> replication_pad2d_npu_attr(const Tensor& input, IntArrayRef paddingSize) {   
  int64_t pad_l = 0;
  int64_t pad_r = 0;
  int64_t pad_t = 0;
  int64_t pad_b = 0;
  int64_t pad_zeros = 0;

  TORCH_CHECK(paddingSize.size() == 4, "padding size is expected to be 4");

  pad_l = paddingSize[0];
  pad_r = paddingSize[1];
  pad_t = paddingSize[2];
  pad_b = paddingSize[3];

  SmallVector<int64_t, SIZE> vectorInt;
  SmallVector<SmallVector<int64_t, SIZE>, SIZE> vectorVectorInt;
  SmallVector<IntArrayRef, SIZE> vectorListInt;
  SmallVector<int64_t, SIZE> paddingsVector = array_to_small_vector(paddingSize);
  paddingsVector.resize(input.dim(), 0);

  for (int i = 0; i < paddingsVector.size(); i ++) {
    if (i<2) {
      vectorInt.emplace_back(pad_zeros);
      vectorInt.emplace_back(pad_zeros);   
    }
    else if (i == 2) {
       vectorInt.emplace_back(pad_l);
       vectorInt.emplace_back(pad_r);
    }
    else {
      vectorInt.emplace_back(pad_t);
      vectorInt.emplace_back(pad_b);
    }
    vectorVectorInt.emplace_back(vectorInt);
    vectorInt.clear();
    vectorListInt.emplace_back(IntArrayRef(vectorVectorInt.back()));
  }
  int64_t constant_values = 0;
  // string mode = "constant";
  string mode = "edge";
  bool padding_contiguous = true;
  NPUAttrDesc npuAttrConstantValues = NPUAttrDesc("constant_values", constant_values);
  NPUAttrDesc npuAttrMode = NPUAttrDesc("mode", mode);
  NPUAttrDesc npuAttrPaddingContiguous = NPUAttrDesc("padding_contiguous", padding_contiguous);
  NPUAttrDesc npuAttrPadding = NPUAttrDesc("paddings", vectorListInt);
  SmallVector<NPUAttrDesc, N> attrs = {
      npuAttrPadding,
      npuAttrConstantValues,
      npuAttrMode,
      npuAttrPaddingContiguous
  };
  return attrs;
}

Tensor& replication_pad2d_out_npu(Tensor& out, const Tensor& self, IntArrayRef padding) {
  //constructs the input and output NPUTensorDesc
  auto inputs = replication_pad2d_npu_input({self});
  auto outputs = replication_pad2d_npu_output({out});

  //constructs the attr of the NPUAttrDesc
  auto attrs = replication_pad2d_npu_attr(self, padding);
    
  //executing the NPU operator 
  CalcuOpUtil::execute_npu_operate("PadV3D", inputs, outputs, attrs);

  return out;
}

Tensor replication_pad2d_npu(const Tensor& self, IntArrayRef padding) {
  //calculate the output size
  auto outputSize = replication_pad2d_npu_output_size(self, padding);
  //construct the output tensor of the NPU
  Tensor out = at::empty_with_format(outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  //calculate the output result of the NPU
  replication_pad2d_out_npu(out, self, padding);

  return out;
}
}
} // namespace at::native
