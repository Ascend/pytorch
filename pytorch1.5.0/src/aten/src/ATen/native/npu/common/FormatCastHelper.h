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

#ifndef __NATIVE_NPU_COMMON_FORMAT_CAST_HELPER__
#define __NATIVE_NPU_COMMON_FORMAT_CAST_HELPER__

#include <ATen/ATen.h>

namespace at {
namespace native {
namespace npu {

class FormatCastHelper {
public:
  static bool IsSameGroupType(const Tensor& src, const Tensor& dst);
  static void format_cast_as_base_format(const Tensor& src, aclFormat format);
  using FormatCastFunc = std::function<Tensor(Tensor&, const Tensor&)>;
  static bool format_cast_between_group(Tensor& dst, const Tensor& src, FormatCastFunc format_cast_inside_group);
  // this interface is similar to CastBackToOriFormat, but CastBackToOriFormat may have overload problem.
  static Tensor ApplyBaseFormatTensorBy(const Tensor& src);
  static Tensor& CovertSelfToBaseFormat(Tensor& src);
private:
  // help function of format_cast_between_group
  static void base_format_cast_nocheck(const Tensor& dst, const Tensor& src);
}; // class FormatCastHelper


} // namespace npu
} // namespace native
} // namespace at

#endif // __NATIVE_NPU_COMMON_FORMAT_CAST_HELPER__