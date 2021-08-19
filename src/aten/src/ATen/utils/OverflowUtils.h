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
#pragma once

#include <ATen/utils/DumpUtils.h>

namespace at {

at::Tensor GetCopyValue(at::Tensor& value);
vector<at::Tensor> GetCopyValue(vector<at::Tensor>& value);

class OverflowUtil {

 public:
  ~OverflowUtil();

  static OverflowUtil* GetInstance() {
    static OverflowUtil instance;
    return &instance;
  }

  void SetCheckSwitch(bool flag) {
    isCheckSwitchOn = flag;
    return;
  }

  bool IsCheckSwitchOn() {
    return isCheckSwitchOn;
  }

  void SetCheckFlag(bool flag) {
    isChecking = flag;
    return;
  }

  bool GetCheckFlag() {
    return isChecking;
  }

  void Lock() {
    mu_.lock();
  }

  void Unlock() {
    mu_.unlock();
  }

  bool CheckOverflowNpu();
  void ClearOverflowNpu();

 private:
  OverflowUtil();
  bool isChecking = false;
  bool isCheckSwitchOn = false;
  std::recursive_mutex mu_;
};

} // namespace c10
