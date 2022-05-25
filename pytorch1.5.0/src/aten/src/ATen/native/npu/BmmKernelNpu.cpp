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
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "c10/npu/OptionsManager.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& bmm_out_npu(Tensor& result, const Tensor& self, const Tensor& mat2) {
  Tensor contiguousResult = result.is_contiguous() ? result : result.contiguous();

  Tensor contiguousSelf = self;
  Tensor contiguousMat2 = mat2;
  bool isSelfT = CalcuOpUtil::is_transpose_last_two_dims(self);
  bool isMat2T = CalcuOpUtil::is_transpose_last_two_dims(mat2);

  if(!isSelfT){
    contiguousSelf = NpuUtils::format_contiguous_add_copy_optimize(self);
  }
  if(!isMat2T){
    contiguousMat2 = NpuUtils::format_contiguous_add_copy_optimize(mat2);
  }

  auto func1 = [&contiguousSelf]() {
      bool pass = false;
      return std::tie(pass, contiguousSelf);
  };
  auto func2 = [&contiguousMat2]() {
      bool pass = false;
      return std::tie(pass, contiguousMat2);
  };

  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("BatchMatMul")
      .InputWithFunc(func1)
      .InputWithFunc(func2)
      .Output(contiguousResult)
      .Attr("adj_x1", isSelfT)
      .Attr("adj_x2", isMat2T)
      .Run();

  if (!result.is_contiguous()) {
    result.copy_(contiguousResult);
  }
  return result;
}

Tensor bmm_npu(const Tensor& self, const Tensor& mat2) {
  // calculate the output size
  auto outputSize = {self.size(0), self.size(1), mat2.size(2)};
  // construct the output tensor of the NPU
  Tensor result;

  // TODO(ASCEND): 检查是否指定mm输出为NCHW。待NLP模型总体策略制定后删去
  if ((self.scalar_type() == ScalarType::Half) && !c10::npu::OptionsManager::CheckSwitchMMOutputEnable()) {
    // check is 16-algined with high-performance
    auto isAligin = [&]() {
      return (!(static_cast<uint64_t>(self.size(1)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(self.size(2)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(mat2.size(1)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(mat2.size(2)) & 0x0000000F));
    };
    // There is a data trampling problem in non-aligned scenes. For the time being, only aligned scenes are supported.
    if (env::CheckMmBmmNDEnable() && FormatHelper::IsBaseFormatType(self) &&
        FormatHelper::IsBaseFormatType(mat2) && isAligin()) {
      result = at::empty_with_format(outputSize, self.options());
    } else {
      result = at::empty_with_format(
          outputSize, self.options(), ACL_FORMAT_FRACTAL_NZ);
    }
  } else {
    result = at::empty_with_format(outputSize, self.options(), ACL_FORMAT_ND);
  }
  // calculate the output result of the NPU
  bmm_out_npu(result, self, mat2);
  return result;
}
} // namespace native
} // namespace at
