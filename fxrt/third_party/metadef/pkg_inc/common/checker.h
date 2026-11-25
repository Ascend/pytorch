/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_COMMON_CHECKER_H_
#define METADEF_CXX_INC_COMMON_CHECKER_H_
#include <securec.h>
#include <sstream>
#include <mutex>
#include "graph/error_codes.h"
#include "common/ge_common/debug/ge_log.h"

struct ErrorResult {
  operator bool() const {
    return false;
  }
  operator ge::graphStatus() const {
    return ge::PARAM_INVALID;
  }
  template <typename T>
  operator std::unique_ptr<T>() const {
    return nullptr;
  }
  template <typename T>
  operator std::shared_ptr<T>() const {
    return nullptr;
  }
  template <typename T>
  operator T*() const {
    return nullptr;
  }
  template <typename T>
  operator std::vector<std::shared_ptr<T>>() const {
    return {};
  }
  template <typename T>
  operator std::vector<T>() const {
    return {};
  }
  operator std::string() const {
    return "";
  }
  template <typename T>
  operator T() const {
    return T();
  }
};

inline std::vector<char> CreateErrorMsg(const char* format, ...) {
  va_list args;
  va_start(args, format);
  va_list args_copy;
  va_copy(args_copy, args);
  const size_t len = static_cast<size_t>(vsnprintf(nullptr, 0, format, args_copy));
  va_end(args_copy);
  std::vector<char> msg(len + 1U, '\0');
  const auto ret = vsnprintf_s(msg.data(), len + 1U, len, format, args);
  va_end(args);
  return (ret > 0) ? msg : std::vector<char>{};
}

inline std::vector<char> CreateErrorMsg() {
  return {};
}

#define GE_ASSERT_EQ(x, y)                                                                     \
  do {                                                                                         \
    const auto& xv = (x);                                                                      \
    const auto& yv = (y);                                                                      \
    if (xv != yv) {                                                                            \
      std::stringstream ss;                                                                    \
      ss << "Assert (" << #x << " == " << #y << ") failed, expect " << yv << " actual " << xv; \
      REPORT_INNER_ERR_MSG("E19999", "%s", ss.str().c_str());                                  \
      GELOGE(ge::FAILED, "%s", ss.str().c_str());                                              \
      return ::ErrorResult();                                                                  \
    }                                                                                          \
  } while (false)

#define GE_WARN_ASSERT(exp, ...)                                      \
  do {                                                                \
    if (!(exp)) {                                                     \
      auto msg = CreateErrorMsg(__VA_ARGS__);                         \
      GELOGW("Assert failed: %s", (msg.empty() ? #exp : msg.data())); \
      return ::ErrorResult();                                         \
    }                                                                 \
  } while (false)

#define GE_ASSERT(exp, ...)                                       \
  do {                                                            \
    if (!(exp)) {                                                 \
      auto msg = CreateErrorMsg(__VA_ARGS__);                     \
      if (msg.empty()) {                                          \
        REPORT_INNER_ERR_MSG("E19999", "Assert %s failed", #exp); \
        GELOGE(ge::FAILED, "Assert %s failed", #exp);             \
      } else {                                                    \
        REPORT_INNER_ERR_MSG("E19999", "%s", msg.data());         \
        GELOGE(ge::FAILED, "%s", msg.data());                     \
      }                                                           \
      return ::ErrorResult();                                     \
    }                                                             \
  } while (false)

#define GE_ASSERT_NOTNULL(v, ...) GE_ASSERT(((v) != nullptr), __VA_ARGS__)
#define GE_ASSERT_SUCCESS(v, ...) GE_ASSERT(((v) == ge::SUCCESS), __VA_ARGS__)
#define GE_ASSERT_GRAPH_SUCCESS(v, ...) GE_ASSERT(((v) == ge::GRAPH_SUCCESS), __VA_ARGS__)
#define GE_ASSERT_RT_OK(v, ...) GE_ASSERT(((v) == 0), __VA_ARGS__)
#define GE_ASSERT_EOK(v, ...) GE_ASSERT(((v) == EOK), __VA_ARGS__)
#define GE_ASSERT_TRUE(v, ...) GE_ASSERT((v), __VA_ARGS__)
#define GE_ASSERT_HYPER_SUCCESS(v, ...) GE_ASSERT(((v).IsSuccess()), __VA_ARGS__)

#define GE_WARN_ASSERT_GRAPH_SUCCESS(v, ...) GE_WARN_ASSERT(((v) == ge::GRAPH_SUCCESS), __VA_ARGS__)
#endif // METADEF_CXX_INC_COMMON_CHECKER_H_
