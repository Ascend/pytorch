/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_OP_IMPL_SPACE_REGISTRY_V2_H_
#define INC_OP_IMPL_SPACE_REGISTRY_V2_H_

#include "register/op_impl_registry.h"
#include "register/op_impl_kernel_registry.h"
#include "base/registry/opp_package_utils.h"

namespace gert {
using char_t = char;
class OpImplRegistryHolder;
class OpImplSpaceRegistryImpl;
using OpImplSpaceRegistryImplPtr = std::shared_ptr<OpImplSpaceRegistryImpl>;

class OpImplSpaceRegistryV2 {
 public:
  OpImplSpaceRegistryV2();

  ~OpImplSpaceRegistryV2() = default;

  ge::graphStatus AddSoToRegistry(const OppSoDesc& so_desc);

  const OpImplKernelRegistry::OpImplFunctionsV2* GetOpImpl(const char_t* op_type) const;

 private:
  friend class OpImplSpaceRegistry;
  // 兼容实现，cann-graph-engine仓后续使用OpImplSpaceRegistryEntry中的public接口
  ge::graphStatus AddRegistry(const std::shared_ptr<OpImplRegistryHolder>& registry_holder);
  OpImplKernelRegistry::OpImplFunctionsV2* CreateOrGetOpImpl(const char_t* op_type);
  OpImplSpaceRegistryImplPtr impl_;
};

using OpImplSpaceRegistryV2Ptr = std::shared_ptr<OpImplSpaceRegistryV2>;
using OpImplSpaceRegistryV2Array =
    std::array<OpImplSpaceRegistryV2Ptr, static_cast<size_t>(OppImplVersionTag::kVersionEnd)>;

class DefaultOpImplSpaceRegistryV2Impl;
using DefaultOpImplSpaceRegistryV2ImplPtr = std::shared_ptr<DefaultOpImplSpaceRegistryV2Impl>;

class DefaultOpImplSpaceRegistryV2 {
 public:
  DefaultOpImplSpaceRegistryV2(const DefaultOpImplSpaceRegistryV2&) = delete;
  DefaultOpImplSpaceRegistryV2(const DefaultOpImplSpaceRegistryV2&&) = delete;
  DefaultOpImplSpaceRegistryV2& operator=(const DefaultOpImplSpaceRegistryV2&) & = delete;
  DefaultOpImplSpaceRegistryV2& operator=(const DefaultOpImplSpaceRegistryV2&&) & = delete;

  static DefaultOpImplSpaceRegistryV2& GetInstance();

  const std::shared_ptr<OpImplSpaceRegistryV2> GetSpaceRegistry(
      gert::OppImplVersionTag opp_impl_version = gert::OppImplVersionTag::kOpp) const;

  ge::graphStatus SetSpaceRegistry(
      const std::shared_ptr<OpImplSpaceRegistryV2>& space_registry_v2,
      gert::OppImplVersionTag version_tag = gert::OppImplVersionTag::kOpp);

  void ClearSpaceRegistry();

 private:
  DefaultOpImplSpaceRegistryV2();
  ~DefaultOpImplSpaceRegistryV2() = default;
  DefaultOpImplSpaceRegistryV2ImplPtr impl_;
};
} // namespace gert
#endif // INC_OP_IMPL_SPACE_REGISTRY_V2_H_
