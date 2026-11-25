/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_BASE_OPP_PACKAGE_UTILS_H_
#define INC_BASE_OPP_PACKAGE_UTILS_H_
#include "graph/ascend_string.h"
#include "graph/error_codes.h"

namespace gert {
enum class OppImplVersionTag {
  kOpp,
  kOppKernel,
  // add new version definitions here
  kVersionEnd = 20
};
class OppSoDescImpl;
using OppSoDescImplPtr = std::unique_ptr<OppSoDescImpl>;
class OppSoDesc {
 public:
  explicit OppSoDesc(const std::vector<ge::AscendString>& so_paths, const ge::AscendString& package_name);
  ~OppSoDesc();
  OppSoDesc(const OppSoDesc& other);
  OppSoDesc& operator=(const OppSoDesc& other);
  OppSoDesc(OppSoDesc&& other) noexcept;
  OppSoDesc& operator=(OppSoDesc&& other) noexcept;

  std::vector<ge::AscendString> GetSoPaths() const;
  ge::AscendString GetPackageName() const;

 private:
  OppSoDescImplPtr impl_;
};

class OppPackageUtils {
 public:
  /**
   * 加载所有的安装目录下的so，优先级：自定义算子 > 内置安装目录算子
   */
  static ge::graphStatus LoadAllOppPackage();
};
} // namespace gert
#endif // INC_BASE_OPP_PACKAGE_UTILS_H_
