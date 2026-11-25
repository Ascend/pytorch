/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTE_GRAPH_TYPE_ID_H
#define EXECUTE_GRAPH_TYPE_ID_H
#include <type_traits>
#include "graph/types.h"

namespace ge {
class GeTensor;
class GeTensorDesc;
class Buffer;
class NamedAttrs;
namespace proto {
class GraphDef;
}

template <typename T>
struct GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeIdHolder {
  static char_t id;
};

template <typename T>
char_t TypeIdHolder<T>::id = static_cast<char_t>(0);

using TypeId = void*;
constexpr TypeId kInvalidTypeId = nullptr;

template <typename T>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId(const T& obj) {
  (void)obj;
  return GetTypeId<T>();
}

template <typename T>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId() {
  using PureT = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  return &(TypeIdHolder<PureT>::id);
}

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<bool>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::string>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<float>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<int64_t>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<GeTensorDesc>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<GeTensor>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<Buffer>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<proto::GraphDef>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<NamedAttrs>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<std::vector<int64_t>>>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<DataType>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<std::vector<float>>>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<std::string>>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<float>>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<bool>>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<int64_t>>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<GeTensorDesc>>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<GeTensor>>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<Buffer>>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<proto::GraphDef>>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<NamedAttrs>>();

template <>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<DataType>>();
} // namespace ge
#endif // EXECUTE_GRAPH_TYPE_ID_H
