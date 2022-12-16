// Copyright (c) 2022 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/utils/NpuDataDumpMgr.h"

#include <algorithm>
#include <map>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
void NpuDataDumpMgr::DatadumpEnqueue(const at::TensorList &inputs,
                                     const at::TensorList &outputs,
                                     const string &opName) {
  if (!enableFlag_) {
    return;
  }
  int idx = NpuDataDumpMgr::GetDatadumpOpIdx(opName);
  if (idx < 0) {
    return;
  }
  ASCEND_LOGI("Datadump enque: %s", opName.c_str());
  enableFlag_ = false;
  string tensorName = std::to_string(idx) + '_' + opName;
  if (!inputs.empty()) {
    at_npu::native::NPUNativeFunctions::npu_enque_tensor(inputs,
                                                         tensorName + "_input");
  }
  if (!outputs.empty()) {
    at_npu::native::NPUNativeFunctions::npu_enque_tensor(
        outputs, tensorName + "_output");
  }
  enableFlag_ = true;
}

void NpuDataDumpMgr::EnableDatadump(
    const c10::SmallVector<std::string, N> &opWhiteList) {
  ASCEND_LOGI("Datadump enable.");
  opWhiteList_ = opWhiteList;
  enableFlag_ = true;
}
void NpuDataDumpMgr::DisableDatadump() {
  ASCEND_LOGI("Datadump disable.");
  enableFlag_ = false;
}

bool NpuDataDumpMgr::IsDatadumpEnable() const { return enableFlag_; }

int NpuDataDumpMgr::GetDatadumpOpIdx(const std::string &opName) {
  if (opWhiteList_.empty() ||
      (std::find(opWhiteList_.begin(), opWhiteList_.end(), opName) !=
       opWhiteList_.end())) {
    return index_++;
  }
  return -1;
}
}  // namespace native
}  // namespace at_npu
