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


#ifndef __C10_NPU_OPTION_REGISTER_H__
#define __C10_NPU_OPTION_REGISTER_H__

#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>

namespace c10 {
namespace npu {

class OptionInterface {
public:
  OptionInterface() {}
  ~OptionInterface() {}
  void Set(const std::string& in) {this->val = in;}
  std::string Get() {return val;}
private:
  std::string val;
};


namespace register_options {
class OptionRegister {
public:
  static OptionRegister* GetInstance() {
    static OptionRegister instance;
    return &instance;
  }

  void Register(const std::string& name, ::std::unique_ptr<OptionInterface>& ptr) {
    std::lock_guard<std::mutex> lock(mu_);
    registry.emplace(name, std::move(ptr));
  }

  void Set(const std::string name, const std::string& val) {
    auto itr = registry.find(name);
    if (itr != registry.end()) {
      itr->second->Set(val);
    } else {
      AT_ERROR("invalid npu option name:", name);
    }
  }

  std::string Get(const std::string name) {
    auto itr = registry.find(name);
    if (itr != registry.end()) {
        return itr->second->Get();
    }
    return ""; // default value
  }

private:
  OptionRegister() {}
  mutable std::mutex mu_;
  mutable std::unordered_map<std::string, ::std::unique_ptr<OptionInterface>> registry;
};


class OptionInterfaceBuilder {
public:
  OptionInterfaceBuilder(const std::string& name, ::std::unique_ptr<OptionInterface>& ptr) {
    OptionRegister::GetInstance()->Register(name, ptr);
  }
};

} // namespace register_options


#define REGISTER_ENV_VARIABLE(name)                                         \
  REGISTER_OPTION(name, OptionInterface)

#define REGISTER_OPTION(name, optimization)                                 \
  REGISTER_OPTION_UNIQ(name, name, optimization)

#define REGISTER_OPTION_UNIQ(id, name, optimization)                        \
    auto options_interface_##id =                                           \
        ::std::unique_ptr<OptionInterface>(new optimization());             \
    static register_options::OptionInterfaceBuilder                         \
        register_options_interface_##id(#name, options_interface_##id);


} // namespace npu
} // namespace c10

#endif // __C10_NPU_OPTION_REGISTER_H__