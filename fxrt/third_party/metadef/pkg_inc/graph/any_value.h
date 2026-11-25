/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTE_GRAPH_ANY_VALUE_H
#define EXECUTE_GRAPH_ANY_VALUE_H
#include <functional>
#include <memory>
#include <vector>

#include "graph/types.h"
#include "graph/type_id.h"
#include "graph/error_codes.h"
#include "graph/def_types.h"
namespace ge {
class ComputeGraph;
using GeTensorPtr = std::shared_ptr<GeTensor>;
using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;
class AnyValue {
 public:
  // 后续删除，新增代码请勿使用这堆using
  using INT = int64_t;
  using FLOAT = float;
  using BOOL = bool;
  using STR = std::string;
  using TENSOR = GeTensorPtr;
  using TENSOR_DESC = GeTensorDesc;
  using GRAPH = ComputeGraphPtr;
  using BYTES = Buffer;
  using NAMED_ATTRS = ge::NamedAttrs;
  using DATA_TYPE = ge::DataType;

  using LIST_INT = std::vector<INT>;
  using LIST_FLOAT = std::vector<FLOAT>;
  using LIST_BOOL = std::vector<BOOL>;
  using LIST_STR = std::vector<STR>;
  using LIST_TENSOR = std::vector<TENSOR>;
  using LIST_TENSOR_DESC = std::vector<TENSOR_DESC>;
  using LIST_GRAPH = std::vector<GRAPH>;
  using LIST_BYTES = std::vector<BYTES>;
  using LIST_NAMED_ATTRS = std::vector<NAMED_ATTRS>;
  using LIST_DATA_TYPE = std::vector<DATA_TYPE>;
  using LIST_LIST_INT = std::vector<std::vector<int64_t>>;
  using LIST_LIST_FLOAT = std::vector<std::vector<float>>;
  using NamedAttrs = ge::NamedAttrs;
  // public type definitions
  // 这堆ValueType的预定义本质上是反向依赖，AnyValue不应该反向依赖ComputeGraph等数据结构
  // 后续整改掉
  enum ValueType {
    VT_NONE = 0,
    VT_STRING = 1,
    VT_FLOAT = 2,
    VT_BOOL = 3,
    VT_INT = 4,
    VT_TENSOR_DESC = 5,
    VT_TENSOR = 6,
    VT_BYTES = 7,
    VT_GRAPH = 8,
    VT_NAMED_ATTRS = 9,
    VT_LIST_LIST_INT = 10,
    VT_DATA_TYPE = 11,
    VT_LIST_LIST_FLOAT = 12,

    VT_LIST_BASE = 1000,
    VT_LIST_STRING = static_cast<int32_t>(VT_LIST_BASE) + static_cast<int32_t>(VT_STRING),
    VT_LIST_FLOAT = static_cast<int32_t>(VT_LIST_BASE) + static_cast<int32_t>(VT_FLOAT),
    VT_LIST_BOOL = static_cast<int32_t>(VT_LIST_BASE) + static_cast<int32_t>(VT_BOOL),
    VT_LIST_INT = static_cast<int32_t>(VT_LIST_BASE) + static_cast<int32_t>(VT_INT),
    VT_LIST_TENSOR_DESC = static_cast<int32_t>(VT_LIST_BASE) + static_cast<int32_t>(VT_TENSOR_DESC),
    VT_LIST_TENSOR = static_cast<int32_t>(VT_LIST_BASE) + static_cast<int32_t>(VT_TENSOR),
    VT_LIST_BYTES = static_cast<int32_t>(VT_LIST_BASE) + static_cast<int32_t>(VT_BYTES),
    VT_LIST_GRAPH = static_cast<int32_t>(VT_LIST_BASE) + static_cast<int32_t>(VT_GRAPH),
    VT_LIST_NAMED_ATTRS = static_cast<int32_t>(VT_LIST_BASE) + static_cast<int32_t>(VT_NAMED_ATTRS),
    VT_LIST_DATA_TYPE = static_cast<int32_t>(VT_LIST_BASE) + static_cast<int32_t>(VT_DATA_TYPE),
  };

  AnyValue() = default;
  AnyValue(AnyValue&& other) noexcept;
  AnyValue(const AnyValue& other) {
    if (!other.IsEmpty()) {
      other.operate_(OperateType::kOpClone, &other, this);
    }
  }
  AnyValue& operator=(AnyValue&& other) noexcept;
  AnyValue& operator=(const AnyValue& other);
  ~AnyValue() noexcept {
    Clear();
  }

  template <class T>
  static AnyValue CreateFrom(T&& value) {
    AnyValue av;
    av.InnerSet(std::forward<T>(value));
    return av;
  }
  // 如果只有万能引用，那么Set<int>(左值)这种调用方法会出错，因此有了这个函数
  template <typename T>
  static AnyValue CreateFrom(const T& value) {
    AnyValue av;
    av.InnerSet(value);
    return av;
  }

  template <class T>
  graphStatus SetValue(T&& value) {
    Clear();
    InnerSet(std::forward<T>(value));
    return GRAPH_SUCCESS;
  }

  // 如果只有万能引用，那么Set<int>(左值)这种调用方法会出错，因此有了这个函数
  template <typename T>
  graphStatus SetValue(const T& value) {
    Clear();
    InnerSet(value);
    return GRAPH_SUCCESS;
  }

  template <typename T>
  graphStatus SetValue(std::initializer_list<T> values) {
    Clear();
    InnerSet(std::vector<T>(std::move(values)));
    return GRAPH_SUCCESS;
  }

  template <typename T>
  graphStatus GetValue(T& value) const {
    auto* const p = Get<T>();
    if (p == nullptr) {
      return GRAPH_FAILED;
    }
    value = *p;
    return GRAPH_SUCCESS;
  }

  template <class T>
  const T* Get() const {
    if (!SameType<T>()) {
      return nullptr;
    }
    if (IsEmpty()) {
      return nullptr;
    }
    return PtrToPtr<const void, const T>(GetAddr());
  }
  template <class T>
  T* MutableGet();

  template <class T>
  bool SameType() const noexcept {
    if (operate_ == nullptr) {
      return false;
    }
    TypeId tid = kInvalidTypeId;
    operate_(OperateType::kGetTypeId, this, &tid);
    return tid == GetTypeId<T>();
  }

  void Swap(AnyValue& other) noexcept;

  void Clear() noexcept {
    if (operate_ == nullptr) {
      return;
    }
    operate_(OperateType::kOpClear, nullptr, this);
  }

  bool IsEmpty() const noexcept {
    return operate_ == nullptr;
  }

  ValueType GetValueType() const noexcept;
  TypeId GetValueTypeId() const noexcept;
  AnyValue Copy() const;

 private:
  template <typename T>
  void InnerSet(T&& value) {
    using PureT = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    using Inline = std::integral_constant<bool, sizeof(PureT) <= sizeof(holder_)>;
    using Operations = typename std::
        conditional<Inline{}, AnyValue::InlineOperations<PureT>, AnyValue::AllocateOperations<PureT>>::type;

    Operations::Construct(std::forward<T>(value), this);
  }
  const void* GetAddr() const;

  enum class OperateType : uint32_t { kOpClear, kOpGetAddr, kOpClone, kOpMove, kGetTypeId, kOperateTypeEnd };

  template <typename T>
  class InlineOperations {
   public:
    static void Operate(const OperateType ot, const AnyValue* const av, void* const out);
    static void Construct(const T& value, AnyValue* const av);
    static void Construct(T&& value, AnyValue* const av);
  };

  template <typename T>
  class AllocateOperations {
   public:
    static void Operate(const OperateType ot, const AnyValue* const av, void* const out);
    static void Construct(const T& value, AnyValue* const av);
    static void Construct(T&& value, AnyValue* const av);
  };

  using ValueBuf = std::aligned_storage<sizeof(void*)>::type;
  using ValueHolder = union {
    void* pointer;
    std::aligned_storage<sizeof(void*)>::type inline_buf;
  };
  ValueHolder holder_ = {nullptr};

  void (*operate_)(OperateType ot, const AnyValue* av, void* out){nullptr};
};
using GeAttrValue = AnyValue;

template <typename T>
void AnyValue::AllocateOperations<T>::Construct(const T& value, AnyValue* const av) {
  av->holder_.pointer = new (std::nothrow) T(value);
  av->operate_ = AnyValue::AllocateOperations<T>::Operate;
}
template <typename T>
void AnyValue::AllocateOperations<T>::Construct(T&& value, AnyValue* const av) {
  av->holder_.pointer = ::new (std::nothrow) T(std::forward<T>(value));
  av->operate_ = AnyValue::AllocateOperations<T>::Operate;
}
template <typename T>
void AnyValue::AllocateOperations<T>::Operate(
    const AnyValue::OperateType ot,
    const AnyValue* const av,
    void* const out) {
  switch (ot) {
    case OperateType::kOpClear: {
      auto* const av_p = PtrToPtr<void, AnyValue>(out);
      delete PtrToPtr<void, T>(av_p->holder_.pointer);
      av_p->holder_.pointer = nullptr;
      av_p->operate_ = nullptr;
      break;
    }
    case OperateType::kOpGetAddr:
      *PtrToPtr<void, void*>(out) = const_cast<void*>(av->holder_.pointer);
      break;
    case OperateType::kOpClone:
      PtrToPtr<void, AnyValue>(out)->holder_.pointer =
          new (std::nothrow) T(*PtrToPtr<const void, const T>(av->holder_.pointer));
      PtrToPtr<void, AnyValue>(out)->operate_ = av->operate_;
      break;
    case OperateType::kOpMove: {
      auto* const av_p = PtrToPtr<void, AnyValue>(out);
      av_p->holder_.pointer = av->holder_.pointer;
      av_p->operate_ = av->operate_;
      const_cast<AnyValue*>(av)->holder_.pointer = nullptr;
      break;
    }
    case OperateType::kGetTypeId:
      *PtrToPtr<void, TypeId>(out) = GetTypeId<T>();
      break;
    default:
      break;
  }
}
template <typename T>
void AnyValue::InlineOperations<T>::Construct(const T& value, AnyValue* const av) {
  (void)::new (&(av->holder_.inline_buf)) T(value);
  av->operate_ = AnyValue::InlineOperations<T>::Operate;
}
template <typename T>
void AnyValue::InlineOperations<T>::Construct(T&& value, AnyValue* const av) {
  Construct(value, av);
}
template <typename T>
void AnyValue::InlineOperations<T>::Operate(const AnyValue::OperateType ot, const AnyValue* const av, void* const out) {
  switch (ot) {
    case OperateType::kOpClear: {
      auto* const av_p = PtrToPtr<void, AnyValue>(out);
      PtrToPtr<std::aligned_storage<sizeof(void*)>::type, T>(&av_p->holder_.inline_buf)->~T();
      av_p->operate_ = nullptr;
      break;
    }
    case OperateType::kOpGetAddr:
      *PtrToPtr<void, void*>(out) = const_cast<void*>(PtrToPtr<const ValueBuf, const void>(&av->holder_.inline_buf));
      break;
    case OperateType::kOpClone: {
      auto* const av_p = PtrToPtr<void, AnyValue>(out);
      (void)new (&av_p->holder_.inline_buf)
          T(*PtrToPtr<const std::aligned_storage<sizeof(void*)>::type, const T>(&av->holder_.inline_buf));
      av_p->operate_ = av->operate_;
      break;
    }
    case OperateType::kOpMove: {
      auto* const av_p = PtrToPtr<void, AnyValue>(out);
      auto* const moved_t_p = const_cast<T*>(PtrToPtr<const ValueBuf, const T>(&av->holder_.inline_buf));
      (void)new (&av_p->holder_.inline_buf) T(std::move(*moved_t_p));
      av_p->operate_ = av->operate_;
      break;
    }
    case OperateType::kGetTypeId:
      *PtrToPtr<void, TypeId>(out) = GetTypeId<T>();
      break;
    default:
      break;
  }
}

template <class T>
auto AnyValue::MutableGet() -> T* {
  if (!SameType<T>()) {
    return nullptr;
  }
  if (IsEmpty()) {
    return nullptr;
  }
  void* addr = nullptr;
  operate_(OperateType::kOpGetAddr, this, &addr);
  return PtrToPtr<void, T>(addr);
}
} // namespace ge

#endif // EXECUTE_GRAPH_ANY_VALUE_H
