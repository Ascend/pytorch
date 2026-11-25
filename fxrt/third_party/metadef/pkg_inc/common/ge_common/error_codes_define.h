/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef PKG_INC_COMMON_GE_COMMON_ERROR_CODES_DEFINE_H_
#define PKG_INC_COMMON_GE_COMMON_ERROR_CODES_DEFINE_H_

#include <map>
#include <string>
#include "ge_common/api_error_codes.h"

// Each module defines error codes using the following macros, name cannot be modified to (name)
#define GE_ERRORNO_COMMON(name, value, desc) \
  GE_ERRORNO(                                \
      ge::InnLogRuntime::RT_HOST,            \
      ge::InnErrorCodeType::ERROR_CODE,      \
      ge::InnErrorLevel::COMMON_LEVEL,       \
      ge::InnSystemIdType::SYSID_GE,         \
      ge::InnSubModuleId::COMMON_MODULE,     \
      name,                                  \
      (value),                               \
      (desc))
#define GE_ERRORNO_CLIENT(name, value, desc) \
  GE_ERRORNO(                                \
      ge::InnLogRuntime::RT_HOST,            \
      ge::InnErrorCodeType::ERROR_CODE,      \
      ge::InnErrorLevel::COMMON_LEVEL,       \
      ge::InnSystemIdType::SYSID_GE,         \
      ge::InnSubModuleId::CLIENT_MODULE,     \
      name,                                  \
      (value),                               \
      (desc))
#define GE_ERRORNO_INIT(name, value, desc) \
  GE_ERRORNO(                              \
      ge::InnLogRuntime::RT_HOST,          \
      ge::InnErrorCodeType::ERROR_CODE,    \
      ge::InnErrorLevel::COMMON_LEVEL,     \
      ge::InnSystemIdType::SYSID_GE,       \
      ge::InnSubModuleId::INIT_MODULE,     \
      name,                                \
      (value),                             \
      (desc))
#define GE_ERRORNO_SESSION(name, value, desc) \
  GE_ERRORNO(                                 \
      ge::InnLogRuntime::RT_HOST,             \
      ge::InnErrorCodeType::ERROR_CODE,       \
      ge::InnErrorLevel::COMMON_LEVEL,        \
      ge::InnSystemIdType::SYSID_GE,          \
      ge::InnSubModuleId::SESSION_MODULE,     \
      name,                                   \
      (value),                                \
      (desc))
#define GE_ERRORNO_GRAPH(name, value, desc) \
  GE_ERRORNO(                               \
      ge::InnLogRuntime::RT_HOST,           \
      ge::InnErrorCodeType::ERROR_CODE,     \
      ge::InnErrorLevel::COMMON_LEVEL,      \
      ge::InnSystemIdType::SYSID_GE,        \
      ge::InnSubModuleId::GRAPH_MODULE,     \
      name,                                 \
      (value),                              \
      (desc))
#define GE_ERRORNO_ENGINE(name, value, desc) \
  GE_ERRORNO(                                \
      ge::InnLogRuntime::RT_HOST,            \
      ge::InnErrorCodeType::ERROR_CODE,      \
      ge::InnErrorLevel::COMMON_LEVEL,       \
      ge::InnSystemIdType::SYSID_GE,         \
      ge::InnSubModuleId::ENGINE_MODULE,     \
      name,                                  \
      (value),                               \
      (desc))
#define GE_ERRORNO_OPS(name, value, desc) \
  GE_ERRORNO(                             \
      ge::InnLogRuntime::RT_HOST,         \
      ge::InnErrorCodeType::ERROR_CODE,   \
      ge::InnErrorLevel::COMMON_LEVEL,    \
      ge::InnSystemIdType::SYSID_GE,      \
      ge::InnSubModuleId::OPS_MODULE,     \
      name,                               \
      (value),                            \
      (desc))
#define GE_ERRORNO_PLUGIN(name, value, desc) \
  GE_ERRORNO(                                \
      ge::InnLogRuntime::RT_HOST,            \
      ge::InnErrorCodeType::ERROR_CODE,      \
      ge::InnErrorLevel::COMMON_LEVEL,       \
      ge::InnSystemIdType::SYSID_GE,         \
      ge::InnSubModuleId::PLUGIN_MODULE,     \
      name,                                  \
      (value),                               \
      (desc))
#define GE_ERRORNO_RUNTIME(name, value, desc) \
  GE_ERRORNO(                                 \
      ge::InnLogRuntime::RT_HOST,             \
      ge::InnErrorCodeType::ERROR_CODE,       \
      ge::InnErrorLevel::COMMON_LEVEL,        \
      ge::InnSystemIdType::SYSID_GE,          \
      ge::InnSubModuleId::RUNTIME_MODULE,     \
      name,                                   \
      (value),                                \
      (desc))
#define GE_ERRORNO_EXECUTOR(name, value, desc) \
  GE_ERRORNO(                                  \
      ge::InnLogRuntime::RT_DEVICE,            \
      ge::InnErrorCodeType::ERROR_CODE,        \
      ge::InnErrorLevel::COMMON_LEVEL,         \
      ge::InnSystemIdType::SYSID_GE,           \
      ge::InnSubModuleId::EXECUTOR_MODULE,     \
      name,                                    \
      (value),                                 \
      (desc))
#define GE_ERRORNO_GENERATOR(name, value, desc) \
  GE_ERRORNO(                                   \
      ge::InnLogRuntime::RT_HOST,               \
      ge::InnErrorCodeType::ERROR_CODE,         \
      ge::InnErrorLevel::COMMON_LEVEL,          \
      ge::InnSystemIdType::SYSID_GE,            \
      ge::InnSubModuleId::GENERATOR_MODULE,     \
      name,                                     \
      (value),                                  \
      (desc))

#define LLM_ERRORNO_COMMON(name, value, desc) \
  GE_ERRORNO(                                 \
      ge::InnLogRuntime::RT_HOST,             \
      ge::InnErrorCodeType::ERROR_CODE,       \
      ge::InnErrorLevel::COMMON_LEVEL,        \
      ge::InnSystemIdType::SYSID_GE,          \
      ge::InnSubModuleId::LLM_ENGINE_MODULE,  \
      name,                                   \
      (value),                                \
      (desc))

// Get error code description
#define GE_GET_ERRORNO_STR(value) ge::StatusFactory::Instance()->GetErrDesc(value)

#define RT_ERROR_TO_GE_STATUS(RT_ERROR) static_cast<ge::Status>(RT_ERROR)

namespace ge {
// System ID
enum class InnSystemIdType : std::uint8_t { SYSID_GE = 8 };

enum class InnLogRuntime : std::uint8_t {
  RT_HOST = 0b01,
  RT_DEVICE = 0b10,
};

// Sub model
enum class InnSubModuleId : std::uint8_t {
  COMMON_MODULE = 0,
  CLIENT_MODULE = 1,
  INIT_MODULE = 2,
  SESSION_MODULE = 3,
  GRAPH_MODULE = 4,
  ENGINE_MODULE = 5,
  OPS_MODULE = 6,
  PLUGIN_MODULE = 7,
  RUNTIME_MODULE = 8,
  EXECUTOR_MODULE = 9,
  GENERATOR_MODULE = 10,
  LLM_ENGINE_MODULE = 11,
};

// Error code type
enum class InnErrorCodeType : std::uint8_t {
  ERROR_CODE = 0b01,
  EXCEPTION_CODE = 0b10,
};

// Error level
enum class InnErrorLevel : std::uint8_t {
  COMMON_LEVEL = 0b000,
  SUGGESTION_LEVEL = 0b001,
  MINOR_LEVEL = 0b010,
  MAJOR_LEVEL = 0b011,
  CRITICAL_LEVEL = 0b100,
};

// Common module error code definition
GE_ERRORNO_COMMON(PARAM_INVALID, 1, "Parameter invalid!"); // 1343225857
GE_ERRORNO_COMMON(RT_FAILED, 3, "Failed to call runtime API!"); // 1343225859
GE_ERRORNO_COMMON(INTERNAL_ERROR, 4, "Internal errors"); // 1343225860

// Error code for plugin manager
GE_ERRORNO_COMMON(GE_PLGMGR_FUNC_NOT_EXIST, 32, "Failed to find any function!"); // 1343225888
GE_ERRORNO_COMMON(GE_PLGMGR_INVOKE_FAILED, 33, "Failed to invoke any function!"); // 1343225889

// Graph module error code definition
GE_ERRORNO_GRAPH(NOT_CHANGED, 64, "The node of the graph no changed."); // 1343242304
} // namespace ge

#endif // PKG_INC_COMMON_GE_COMMON_ERROR_CODES_DEFINE_H_
