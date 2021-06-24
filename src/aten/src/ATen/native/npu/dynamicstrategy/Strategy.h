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
#ifndef __NATIVE_NPU_UTILS_DYNAMIC_STRATEGY__
#define __NATIVE_NPU_UTILS_DYNAMIC_STRATEGY__

#include <ATen/native/npu/frame/FormatHelper.h>
#include <ATen/native/npu/frame/InputInfoLib.h>
#include <c10/util/SmallVector.h>
#include <third_party/acl/inc/acl/acl_base.h>
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
namespace npu {

class DescStrategyBase {
 public:
  virtual ~DescStrategyBase(){};

  void CreateDefaultDescInfo(
      const aclTensorDesc** descs,
      int num,
      int64_t* storageDims,
      aclFormat* storageFormats,
      SmallVector<FormatShape, N>& inShape,
      SmallVector<FormatShape, N>& inStorageShape);

  virtual void CreateInputDescInfo(
      ACL_PARAMS& params,
      DynamicCompileShape& compileShape) = 0;

  virtual void CreateOutputDescInfo(
      ACL_PARAMS& params,
      DynamicCompileShape& compileShape) = 0;
};

namespace register_dynamic_shape {
class DynamicOptRegister {
 public:
  ~DynamicOptRegister() = default;
  static DynamicOptRegister* GetInstance() {
    static DynamicOptRegister instance;
    return &instance;
  }
  void Register(std::string name, ::std::unique_ptr<DescStrategyBase>& ptr) {
    std::lock_guard<std::mutex> lock(mu_);
    registry.emplace(name, std::move(ptr));
  }

  void CreateDynmaicDescInfo(
      const std::string& name,
      ACL_PARAMS& params,
      DynamicCompileShape& compileShape) {
    auto itr = registry.find(name);
    if (itr != registry.end()) {
      itr->second->CreateOutputDescInfo(params, compileShape);
      itr->second->CreateInputDescInfo(params, compileShape);
      return;
    }

    itr = registry.find("Default");
    itr->second->CreateOutputDescInfo(params, compileShape);
    itr->second->CreateInputDescInfo(params, compileShape);
    return;
  }

 private:
  DynamicOptRegister() {}
  mutable std::mutex mu_;
  mutable std::map<std::string, ::std::unique_ptr<DescStrategyBase>> registry;
}; // class DynamicOptRegister

class DynamicOptBuilder {
 public:
  DynamicOptBuilder(
      std::string name,
      ::std::unique_ptr<DescStrategyBase>& ptr) {
    DynamicOptRegister::GetInstance()->Register(name, ptr);
  }
  ~DynamicOptBuilder() {}
}; // class DynamicOptBuilder
} // namespace register_dynamic_shape

#define REGISTER_DYNAMIC_SHAPE_OPT(name, optimization) \
  REGISTER_DYNAMIC_SHAPE_OPT_UNIQ(name, name, optimization)
#define REGISTER_DYNAMIC_SHAPE_OPT_UNIQ(id, name, optimization) \
  auto dynamic_shape_##id =                                     \
      ::std::unique_ptr<DescStrategyBase>(new optimization());  \
  static register_dynamic_shape::DynamicOptBuilder              \
      register_dynamic_shape_opt##id(#name, dynamic_shape_##id);
} // namespace npu
} // namespace native
} // namespace at

#endif