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

class NpuDataDumpMgr {
public:
  int GetDatadumpOpIdx(const std::string &opName) {
    if (opWhiteList_.empty() ||
        (std::find(opWhiteList_.begin(), opWhiteList_.end(), opName) !=
          opWhiteList_.end())) {
      return index_++;
    }
    return -1;
  }

  void Enable(const std::vector<std::string> &opWrites) {
    opWhiteList_ = opWrites;
    enableFlag_ = true;
  }

  void Enable() { enableFlag_ = true; }

  bool IsEnable() const { return enableFlag_; }

  void Disable() { enableFlag_ = false; }

private:
  bool enableFlag_ = false;
  c10::SmallVector<std::string, N> opWhiteList_;
  int index_ = 0;
};

static NpuDataDumpMgr instance;

int DatadumpInputsEnqueue(const at::TensorList &tensors, const string &opName) {
  if (!instance.IsEnable()) {
    return -1;
  }
  int idx = instance.GetDatadumpOpIdx(opName);
  if ((idx < 0) || tensors.empty()) {
    return idx;
  }
  std::string tensorName = std::to_string(idx) + '_' + opName + "_input";
  instance.Disable();
  at_npu::native::NPUNativeFunctions::npu_enque_tensor(tensors, tensorName);
  instance.Enable();
  return idx;
}

void DatadumpOutputsEnqueue(const at::TensorList &tensors, const string &opName,
                            int idx) {
  if ((idx < 0) || tensors.empty()) {
    return;
  }
  std::string tensorName = std::to_string(idx) + '_' + opName + "_output";
  instance.Disable();
  at_npu::native::NPUNativeFunctions::npu_enque_tensor(tensors, tensorName);
  instance.Enable();
}

void EnableDatadump(const std::vector<std::string> &opWrites) {
  instance.Enable(opWrites);
}

void DisableDatadump() { instance.Disable(); }

bool IsDatadumpEnable() { instance.IsEnable(); }

}  // namespace native
}  // namespace at_npu
