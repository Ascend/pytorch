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

#include "DebugDynamic.h"
#include "c10/npu/OptionsManager.h"
#include <regex>
#include "c10/npu/npu_log.h"
#include <string>
#include <iostream>
#include <chrono>
#include <set>
#include <fstream>

namespace at {
namespace native {
namespace npu {
using namespace std;

DebugDynamic::DebugDynamic() {
  configPath = c10::npu::OptionsManager::CheckDisableDynamicPath();
  if (configPath != "") {
    ConfigFileRead(configInfo);
  } else {
    NPU_LOGD("dynamic config file path is empty!");
  }
}

bool DebugDynamic::CheckInConfig(const string& opName) {
  if (configPath == "") {
    return false;
  }

  // check securse path
  std::string pattern{"/[\\w]+/[\\w]+.conf"};
  std::regex re(pattern);
  bool ret = std::regex_match(configPath, re);
  if (ret == false) {
    NPU_LOGD("configPath path Fails, path %s", (char*)configPath.c_str());
    return false;
  }

  if (configInfo.find(opName) != configInfo.end()) {
    NPU_LOGD("Check SetConfigInfo disable op is %s", (char*)opName.c_str());
    return true;
  }

  return false;
}

void DebugDynamic::ConfigFileRead(set<string>& configInfo) {
  string path = "";
  if (configPath.length() != 0) {
    path = configPath;
  } else {
    return;
  }

  ifstream configFile;
  configFile.open(path.c_str());
  string str_line;
  if (configFile.is_open()) {
    while (!configFile.eof()) {
      getline(configFile, str_line);
      if (str_line.find('#') == 0) {
        continue;
      }

      configInfo.insert(str_line);
    }
  } else {
    NPU_LOGD("dynamic config wrong path %s", path.c_str());
    return;
  }
}
} // namespace npu
} // namespace native
} // namespace at