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
#pragma once

#include <cinttypes>
#include <c10/macros/Export.h>

namespace at {
namespace native {

class GlobalStep
{
 public:
  static GlobalStep& Instance();
  void GlobalStepInc();
  int64_t GetGlobalStep() const;
  void SetStartFuzzCompileStep(const int64_t step);
  int64_t GetStartFuzzCompileStep() const;
  ~GlobalStep() = default;

 private:  
  int64_t GLOBAL_STEP;
  int64_t START_FUZZ_COMPILE_STEP;
  GlobalStep(int64_t globalstep, int64_t startstep) { 
    GLOBAL_STEP = globalstep;
    START_FUZZ_COMPILE_STEP = startstep; 
  }
};
TORCH_NPU_API bool check_fuzz_enable();
}
}