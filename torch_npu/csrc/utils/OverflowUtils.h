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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace torch_npu {
namespace utils {

class OverflowUtil {
public:
  ~OverflowUtil();

  static OverflowUtil *GetInstance() {
    static OverflowUtil instance;
    return &instance;
  }

  void EnableOverflowNpu();
  bool CheckOverflowNpu();
  void ClearOverflowNpu();

private:
  OverflowUtil();
  bool hasOverflow = false;
};

}
}
