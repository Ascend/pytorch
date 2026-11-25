/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_TILING_PARSE_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_TILING_PARSE_CONTEXT_H_
#include "exe_graph/runtime/extended_kernel_context.h"
#include "graph/types.h"

namespace fe {
class PlatFormInfos;
} // namespace fe

namespace gert {
class TilingParseContext : public ExtendedKernelContext {
 public:
  /**
   * 获取算子编译产生的json字符串
   * @return json字符串
   */
  const ge::char_t* GetCompiledJson() const {
    return GetInputValue<const char*>(0);
  }
  /**
   * 获取`CompiledInfo`实例
   * @tparam T 实例类型，该类型需要与`IMPL_OP`注册时TilingParse的类型一致
   * @return 指向`CompiledInfo`实例的指针
   */
  template <typename T>
  auto GetCompiledInfo() -> T* {
    auto av = GetOutput(0);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetValue<T*>();
  }
  /**
   * 获取 fe::PlatFormInfos 指针
   * @return fe::PlatFormInfos 指针
   */
  fe::PlatFormInfos* GetPlatformInfo() const {
    return GetInputValue<fe::PlatFormInfos*>(1);
  }
};
static_assert(std::is_standard_layout<TilingParseContext>::value, "The class TilingParseContext must be a POD");
} // namespace gert
#endif // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_TILING_PARSE_CONTEXT_H_
