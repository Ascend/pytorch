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

#include <ATen/utils/OverflowUtils.h>

namespace at {

  at::Tensor GetCopyValue(at::Tensor& value) {
    if (!value.has_storage()) {
      return value;
    }
    return value.detach().clone();
  }

  vector<at::Tensor> GetCopyValue(vector<at::Tensor>& value) {
    vector<at::Tensor> list;
    list.reserve(value.size());
    for (auto &tensor : value) {
      list.emplace_back(GetCopyValue(tensor));
    }
    return list;
  }

  OverflowUtil::OverflowUtil() {
  }

  OverflowUtil::~OverflowUtil() {
  }

  bool OverflowUtil::CheckOverflowNpu() {
    auto options = at::TensorOptions(at::kNPU).dtype(at::kFloat);
    Tensor tmp = at::empty({8}, options);
    auto floatStatus = at::npu_alloc_float_status(tmp);

    auto result = at::npu_get_float_status(floatStatus);
    if (floatStatus.cpu()[0].item().toInt() != 0) {
      return true;
    }
    return false;
  }

  void OverflowUtil::ClearOverflowNpu() {
    auto options = at::TensorOptions(at::kNPU).dtype(at::kFloat);
    Tensor tmp = at::empty({8}, options);
    auto floatStatus = at::npu_alloc_float_status(tmp);
    auto result = at::npu_clear_float_status(floatStatus);
    return;
  }
}
