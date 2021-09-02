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

#ifndef __DYNAMIC_LOG_UTIL__
#define __DYNAMIC_LOG_UTIL__

#include <string.h>
#include <chrono>
#include <iostream>

#include "c10/npu/OptionsManager.h"

using namespace std;

namespace at {
namespace native {
namespace npu {

class DynamicLogUtil {
 public:
  DynamicLogUtil() {
    isLogEnable = c10::npu::OptionsManager::CheckDynamicLogEnable();
  }
  ~DynamicLogUtil() = default;
  void PrintLog(long long int step, string key, string opInfo) {
    if (isLogEnable) {
      time_t now = time(0);
      char* time = ctime(&now);
      char time_[25] = "";
      memcpy(time_, time, strlen(time) - 1);

      auto endTime = chrono::system_clock::now();
      auto duration =
          chrono::duration_cast<chrono::microseconds>(endTime - startTime);
      auto usedTime = double(duration.count()) *
          chrono::microseconds::period::num / chrono::microseconds::period::den;

      printf(
          "[%s]step=%lld. %s. %s. used time=%lfs\n",
          time_,
          step,
          key.c_str(),
          opInfo.c_str(),
          usedTime);
    }
  }

  void SetStartTime() {
    startTime = chrono::system_clock::now();
  }

 private:
  bool isLogEnable = false;
  std::chrono::system_clock::time_point startTime;
};
} // namespace npu
} // namespace native
} // namespace at

#endif