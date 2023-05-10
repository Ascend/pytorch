// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at_npu
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __PLUGIN_NATIVE_UTILS_JITCOMPILELIST__
#define __PLUGIN_NATIVE_UTILS_JITCOMPILELIST__

#include <string>
#include <set>

using std::string;
using std::vector;

namespace at_npu {
namespace native {

class ForceJitCompileList {
public:
  static ForceJitCompileList& GetInstance();
  void RegisterJitlist(const std::string& blacklist);
  bool Inlist(const std::string& opName) const;
  void DisplayJitlist() const;
  ~ForceJitCompileList() = default;
private:
  ForceJitCompileList() {}
  std::set<std::string> jit_list_;
};

}
}

#endif