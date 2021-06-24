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

#ifndef __NATIVE_NPU_UTILS_NPU_TYPE_PROPERIES__
#define __NATIVE_NPU_UTILS_NPU_TYPE_PROPERIES__


#include <ATen/ATen.h>

namespace at { namespace native { namespace npu {

struct ResultTypeState {
  c10::ScalarType dimResult = ScalarType::Undefined;
  c10::ScalarType wrappedResult = ScalarType::Undefined;
  c10::ScalarType zeroResult = ScalarType::Undefined;
};

ResultTypeState update_result_type_state(const Tensor& tensor, const ResultTypeState& in_state);
ScalarType result_type(const ResultTypeState& state);
ScalarType result_type(ScalarType a, ScalarType b);

}}}

#endif // __NATIVE_NPU_UTILS_NPU_TYPE_PROPERIES__