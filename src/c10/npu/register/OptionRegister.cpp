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

#include <algorithm>
#include "OptionRegister.h"
#include "c10/util/Exception.h"

namespace c10 {
namespace npu {

OptionInterface::OptionInterface(OptionCallBack callback) {
  this->callback = callback;
}

void OptionInterface::Set(const std::string& in) {
  this->val = in;
  if (this->callback != nullptr) {
    this->callback(in);
  }
}

std::string OptionInterface::Get() {
  return val;
}


namespace register_options {
OptionRegister* OptionRegister::GetInstance() {
  static OptionRegister instance;
  return &instance;
}

void OptionRegister::Register(const std::string& name,
                            ::std::unique_ptr<OptionInterface>& ptr) {
  std::lock_guard<std::mutex> lock(mu_);
  registry.emplace(name, std::move(ptr));
}

void OptionRegister::Set(const std::string& name, const std::string& val) {
  auto itr = registry.find(name);
  if (itr != registry.end()) {
    itr->second->Set(val);
  } else {
    AT_ERROR("invalid npu option name:", name);
  }
}

c10::optional<std::string> OptionRegister::Get(const std::string& name) {
  auto itr = registry.find(name);
  if (itr != registry.end()) {
    return itr->second->Get();
  }
  return c10::nullopt; // default value
}

OptionInterfaceBuilder::OptionInterfaceBuilder(
    const std::string& name,
    ::std::unique_ptr<OptionInterface>& ptr,
    const std::string& type) {
  OptionRegister::GetInstance()->Register(name, ptr);

  // init the value if env variable.
  if (type == "env") {
    std::string env_name = name;
    std::transform(env_name.begin(), env_name.end(), env_name.begin(), ::toupper);
    char* env_val = std::getenv(env_name.c_str());
    if (env_val != nullptr) {
      std::string val(env_val);
      OptionRegister::GetInstance()->Set(name, val);
    }
  }
}
} // namespace register_options

void SetOption(const std::string& key, const std::string& val) {
  register_options::OptionRegister::GetInstance()->Set(key, val);
}

void SetOption(const std::map<std::string, std::string>& options) {
  for (auto item : options) {
    SetOption(item.first, item.second);
  }
}

c10::optional<std::string> GetOption(const std::string& key) {
  return register_options::OptionRegister::GetInstance()->Get(key);
}

} // namespace c10
} // namespace npu
