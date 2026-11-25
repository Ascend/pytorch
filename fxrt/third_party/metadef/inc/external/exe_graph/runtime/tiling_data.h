/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_TILING_DATA_H_
#define METADEF_CXX_INC_EXE_GRAPH_TILING_DATA_H_
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <securec.h>
#include <memory>
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "graph/error_codes.h"
#include "utils/extern_math_util.h"

namespace gert {
enum class AttrDataType : int32_t {
  kBool = 0,
  kString,
  kInt32,
  kInt64,
  kUint32,
  kFloat32,
  kFloat16,
  kListBool,
  kListString,
  kListInt32,
  kListInt64,
  kListUint32,
  kListFloat32,
  kListFloat16,
  kListListInt32,
  kListListInt64,
  kBfloat16,
  kInt8,
  kInt16,
  kListInt8,
  kListInt16,
  kListListBool,
  kListListFloat16,
  kListBfloat16,
  kListListBfloat16,
  kListListFloat32,
  kListListInt8,
  kListListInt16,
  kUint8,
  kListUint8,
  kListListUint8,
  kUint16,
  kListUint16,
  kListListUint16,
  kListListUint32,
  kUint64,
  kListUint64,
  kListListUint64,
  kTypeEnd
};
class TilingData {
 public:
  /**
   * 获取本实例可容纳的最大tiling data长度
   * @return 最大tiling data长度
   */
  size_t GetCapacity() const {
    return capacity_;
  }
  /**
   * 获取tiling data长度
   * @return tiling data长度
   */
  size_t GetDataSize() const {
    return data_size_;
  }
  /**
   * 设置tiling data长度
   * @param size tiling data长度
   */
  void SetDataSize(const size_t size) {
    data_size_ = size;
  }
  /**
   * 获取data指针
   * @return data指针
   */
  void* GetData() {
    return data_;
  }
  /**
   * 获取data指针
   * @return data指针
   */
  const void* GetData() const {
    return data_;
  }
  /**
   * 向后添加tiling data，若添加超过可容纳的最大长度，则添加失败
   * @tparam T 添加的tiling data的类型
   * @param data 添加的tiling data实例
   * @return 成功返回ge::GRAPH_SUCCESS
   */
  template <typename T, typename std::enable_if<std::is_standard_layout<T>::value, int>::type = 0>
  ge::graphStatus Append(const T& data) {
    auto data_ptr = Expand(sizeof(data));
    if (data_ptr == nullptr) {
      return ge::GRAPH_FAILED;
    }
    if (memcpy_s(data_ptr, sizeof(data), &data, sizeof(data)) != EOK) {
      data_size_ -= sizeof(data);
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }

  template <typename T, typename std::enable_if<std::is_standard_layout<T>::value, int>::type = 0>
  ge::graphStatus Append(const T* data, size_t append_num) {
    size_t append_size;
    if (ge::MulOverflow(sizeof(T), append_num, append_size)) {
      return ge::GRAPH_MUL_OVERFLOW;
    }
    auto data_ptr = Expand(append_size);
    if (data_ptr == nullptr) {
      return ge::GRAPH_ADD_OVERFLOW;
    }
    // Expand中已保证append_size的合法性，为减少冗余判断逻辑此处使用memcpy直接拷贝
    (void)memcpy(data_ptr, data, append_size);
    return ge::GRAPH_SUCCESS;
  }

  /**
   * 将tiling data拓展size大小
   * @param size 需要拓展的大小，单位为字节
   * @return 返回扩展对应size的内存地址
   */
  void* Expand(size_t size) {
    size_t after_size;
    if (ge::AddOverflow(data_size_, size, after_size)) {
      return nullptr;
    }
    if (after_size > capacity_) {
      return nullptr;
    }
    void* ret = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(data_) + data_size_);
    data_size_ = after_size;
    return ret;
  }

  /**
   * 通过最大容量创建一个TilingData类实例
   * @param cap_size 最大容量，单位为字节
   * @return 实例指针
   */
  static std::unique_ptr<uint8_t[]> CreateCap(const size_t cap_size) {
    size_t total_size;
    if (ge::AddOverflow(sizeof(TilingData), cap_size, total_size)) {
      return nullptr;
    }
    auto td_buf = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]());
    if (td_buf == nullptr) {
      return nullptr;
    }
    const auto td = reinterpret_cast<TilingData*>(td_buf.get());
    td->Init(cap_size, td_buf.get() + sizeof(TilingData));
    return td_buf;
  }
  /**
   * 通过最大容量计算TilingData实例所占用的内存空间
   * @param cap_size 最大容量，单位为字节
   * @param total_size 内存空间，单位为字节
   * @return 成功返回ge::GRAPH_SUCCESS
   */
  static ge::graphStatus CalcTotalSize(const size_t cap_size, size_t& total_size) {
    if (ge::AddOverflow(sizeof(TilingData), cap_size, total_size)) {
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 初始化TilingData
   * @param cap_size 最大容量
   * @param data tiling data的地址
   */
  void Init(const size_t cap_size, void* const data) {
    capacity_ = cap_size;
    data_size_ = 0UL;
    data_ = data;
    (void)memset_s(reserved_, sizeof(reserved_), 0, sizeof(reserved_));
  }

  ge::graphStatus AppendConvertedAttrVal(
      const RuntimeAttrs* attrs,
      const size_t attr_index,
      const AttrDataType src_type,
      const AttrDataType dst_type);
  TilingData(const TilingData&) = delete;
  TilingData(TilingData&&) = delete;
  TilingData operator=(const TilingData&) = delete;
  TilingData operator=(TilingData&&) = delete;

 private:
  TilingData() = default;
  size_t capacity_;
  size_t data_size_;
  void* data_;
  uint8_t reserved_[40]; // Reserved field, 32+8, do not directly use when only 8-byte left
};

/**
 * 向后添加tiling data，若添加超过可容纳的最大长度，则忽略本次操作
 * @tparam T 添加的tiling data的类型
 * @param out TilingData类实例
 * @param data 添加的tiling data的实例
 * @return
 */
template <typename T>
TilingData& operator<<(TilingData& out, const T& data) {
  out.Append(data); // we cannot throw exception, so callers cannot get the error information
  return out;
}
static_assert(std::is_standard_layout<TilingData>::value, "The class TilingData must be a POD");
} // namespace gert

#endif // METADEF_CXX_INC_EXE_GRAPH_TILING_DATA_H_
