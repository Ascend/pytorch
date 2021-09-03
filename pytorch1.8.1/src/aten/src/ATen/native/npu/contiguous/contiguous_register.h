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

#ifndef __NATIVE_NPU_CONTIGUOUS_CONTIGUOUS_REGISTER__
#define __NATIVE_NPU_CONTIGUOUS_CONTIGUOUS_REGISTER__

#include <mutex>
#include <string>
#include <map>
#include "ATen/ATen.h"
#include <c10/util/Optional.h>
#include "ATen/native/npu/frame/FormatHelper.h"
#include "ATen/native/npu/frame/StorageDescHelper.h"

namespace at {
namespace native {
namespace npu {

class ContiguousOpt {
 public:
  ContiguousOpt() {}
  virtual ~ContiguousOpt() {}
  virtual bool Optimizer(const Tensor& src, Tensor& self) = 0;
  virtual bool CanOptimizer(const Tensor& src) {
    return false;
  }
};

namespace register_opt {
class CopyOptRegister {
 public:
  static CopyOptRegister* GetInstance() {
    static CopyOptRegister instance;
    return &instance;
  }
  void Register(std::string name, ::std::unique_ptr<ContiguousOpt>& ptr) {
    std::lock_guard<std::mutex> lock(mu_);
    registry.emplace(name, std::move(ptr));
  }

  bool CanOptimize(std::string name, const Tensor& src) {
    auto itr = registry.find(name);
    if (itr != registry.end()) {
      return itr->second->CanOptimizer(src);
    }
    return false;
  }

  bool Run(std::string name, const Tensor& src, Tensor& self) {
    auto itr = registry.find(name);
    if (itr != registry.end()) {
      return itr->second->Optimizer(src, self);
    }
    return false;
  }

 private:
  CopyOptRegister() {}
  mutable std::mutex mu_;
  mutable std::map<std::string, ::std::unique_ptr<ContiguousOpt>> registry;
}; // class CopyOptRegister

class CopyOptBuilder {
 public:
  CopyOptBuilder(std::string name, ::std::unique_ptr<ContiguousOpt>& ptr) {
    CopyOptRegister::GetInstance()->Register(name, ptr);
  }
}; // class CopyOptBuilder
} // namespace register_opt

#define REGISTER_COPY_OPT(name, optimization) \
  REGISTER_COPY_OPT_UNIQ(name, name, optimization)
#define REGISTER_COPY_OPT_UNIQ(id, name, optimization)                       \
  auto copy_opt_##id = ::std::unique_ptr<ContiguousOpt>(new optimization()); \
  static register_opt::CopyOptBuilder register_copy_opt##id(                 \
      #name, copy_opt_##id);

} // namespace npu
} // namespace native
} // namespace at

#endif