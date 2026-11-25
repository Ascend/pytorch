/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_REGISTER_REGISTER_ERROR_CODES_H_
#define INC_EXTERNAL_REGISTER_REGISTER_ERROR_CODES_H_

#define SYSID_FWK 3U // Subsystem ID
#define MODID_COMMON 0U // Common module ID

#define DECLARE_ERRORNO(sysid, modid, name, value)                                                         \
  constexpr domi::Status name = ((static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(sysid)))) << 24U) | \
      ((static_cast<uint32_t>(0xFFU & (static_cast<uint32_t>(modid)))) << 16U) |                           \
      (static_cast<uint32_t>((0xFFFFU & (static_cast<uint32_t>(value)))));

#define DECLARE_ERRORNO_COMMON(name, value) DECLARE_ERRORNO(SYSID_FWK, MODID_COMMON, name, value)

namespace domi {
using Status = uint32_t;

// General error code
DECLARE_ERRORNO(0U, 0U, SUCCESS, 0U);
DECLARE_ERRORNO(0xFFU, 0xFFU, FAILED, 0xFFFFFFFFU);
DECLARE_ERRORNO_COMMON(PARAM_INVALID, 1U); // 50331649
DECLARE_ERRORNO(SYSID_FWK, 1U, SCOPE_NOT_CHANGED, 201U);
} // namespace domi

#endif // INC_EXTERNAL_REGISTER_REGISTER_ERROR_CODES_H_
