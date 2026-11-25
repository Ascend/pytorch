/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXTERNAL_UTILS_EXTERN_MATH_UTIL_H
#define METADEF_CXX_INC_EXTERNAL_UTILS_EXTERN_MATH_UTIL_H

#include <iostream>
#include <cstdint>
#include <limits>

namespace ge {
template <typename T>
class IntegerChecker {
 public:
  template <typename T1>
  static bool Compat(const T1 v) {
    static_assert(
        ((sizeof(T) <= sizeof(uint64_t)) && (sizeof(T1) <= sizeof(uint64_t))),
        "IntegerChecker can only check integers less than 64 bits");
    if (v >= static_cast<T1>(0)) {
      return static_cast<uint64_t>(v) <= static_cast<uint64_t>(std::numeric_limits<T>::max());
    }
    return static_cast<int64_t>(v) >= static_cast<int64_t>(std::numeric_limits<T>::min());
  }
};

template <typename TLhs, typename TRhs, typename TRet>
bool MulOverflow(TLhs lhs, TRhs rhs, TRet& ret) {
#if __GNUC__ >= 5
  return __builtin_mul_overflow(lhs, rhs, &ret);
#else
  if ((!IntegerChecker<TRet>::Compat(lhs)) || (!IntegerChecker<TRet>::Compat(rhs))) {
    return true;
  }
  if ((lhs == 0) || (rhs == 0)) {
    ret = 0;
    return false;
  }
  TRet reminder = std::numeric_limits<TRet>::max() / static_cast<TRet>(rhs);
  const TRet lhs_ret_type = static_cast<TRet>(lhs);
  if (lhs_ret_type < 0) {
    if (reminder > 0) {
      reminder *= static_cast<TRet>(-1);
    }
    if (lhs_ret_type < reminder) {
      return true;
    }
  } else {
    if (reminder < 0) {
      reminder *= static_cast<TRet>(-1);
    }
    if (lhs_ret_type > reminder) {
      return true;
    }
  }
  ret = static_cast<TRet>(lhs) * static_cast<TRet>(rhs);
  return false;
#endif
}

template <typename TLhs, typename TRhs, typename TRet>
bool AddOverflow(TLhs lhs, TRhs rhs, TRet& ret) {
#if __GNUC__ >= 5
  return __builtin_add_overflow(lhs, rhs, &ret);
#else
  if ((!IntegerChecker<TRet>::Compat(lhs)) || (!IntegerChecker<TRet>::Compat(rhs))) {
    return true;
  }
  if (rhs >= 0) {
    if (static_cast<TRet>(lhs) > std::numeric_limits<TRet>::max() - static_cast<TRet>(rhs)) {
      return true;
    }
  } else {
    if (static_cast<TRet>(lhs) < std::numeric_limits<TRet>::min() - static_cast<TRet>(rhs)) {
      return true;
    }
  }
  ret = static_cast<TRet>(lhs) + static_cast<TRet>(rhs);
  return false;
#endif
}
} // namespace ge

#endif // METADEF_CXX_INC_EXTERNAL_UTILS_EXTERN_MATH_UTIL_H
