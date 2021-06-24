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

#include <string>
#include <set>
using std::string;
using std::vector;

namespace at {
namespace native {
namespace npu {

class FuzzyCompileBlacklist {
public:
  static FuzzyCompileBlacklist& GetInstance() {
    static FuzzyCompileBlacklist fuzzy_black_list;
      return fuzzy_black_list;
  }
  void RegisterBlacklist(const std::string blacklist);
  bool IsInBlacklist(const std::string opName) const;
  void DisplayBlacklist() const;
  ~FuzzyCompileBlacklist() = default;
private:
  FuzzyCompileBlacklist() {}
  std::set<std::string> black_list_;
};


}
}
}