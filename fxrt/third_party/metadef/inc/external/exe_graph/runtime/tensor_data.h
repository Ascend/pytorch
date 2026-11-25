/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_TENSOR_DATA_H_
#define METADEF_CXX_INC_EXE_GRAPH_TENSOR_DATA_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include "graph/error_codes.h"

namespace gert {
using TensorAddress = void*;
using ConstTensorAddressPtr = const void*;

enum TensorPlacement : int32_t {
  kOnDeviceHbm, ///< Tensor位于Device上的HBM内存
  kOnHost, ///< Tensor位于Host
  kFollowing, ///< Tensor位于Host，且数据紧跟在结构体后面
  kOnDeviceP2p, ///< Tensor位于Device上的P2p内存
  kTensorPlacementEnd
};

class TensorPlacementUtils {
 public:
  static bool IsOnDevice(TensorPlacement placement) {
    return (placement == kOnDeviceHbm) || (placement == kOnDeviceP2p);
  }
  static bool IsOnHost(TensorPlacement placement) {
    return (placement == kOnHost) || (placement == kFollowing);
  }
  static bool IsOnHostFollowing(TensorPlacement placement) {
    return (placement == kFollowing);
  }
  static bool IsOnHostNotFollowing(TensorPlacement placement) {
    return (placement == kOnHost);
  }
  static bool IsOnDeviceHbm(TensorPlacement placement) {
    return (placement == kOnDeviceHbm);
  }
  static bool IsOnDeviceP2p(TensorPlacement placement) {
    return (placement == kOnDeviceP2p);
  }
};

enum TensorOperateType : int32_t {
  kGetTensorAddress, ///< 获取Tensor的地址
  kFreeTensor, ///< 释放Tensor
  kPlusShareCount, ///< 共享Tensor
  kTensorOperateType
};

/**
 * Tensor的管理函数
 */
using TensorAddrManager = ge::graphStatus (*)(TensorAddress addr, TensorOperateType operate_type, void** out);

class TensorData {
 public:
  /**
   * 构造一个TensorData
   * @param addr tensor的地址
   * @param manager tensor data的管理函数，若manager为空，则认为addr就是tensor的数据地址，且此数据不需要被释放
   */
  // memse函数misra告警屏蔽
  explicit TensorData(TensorAddress addr = nullptr, const TensorAddrManager manager = nullptr)
      : TensorData(addr, manager, 0U, kTensorPlacementEnd) {}
  explicit TensorData(TensorAddress addr, const TensorAddrManager manager, size_t size, TensorPlacement placement)
      : addr_(addr), manager_(manager), size_(size), placement_(placement), reserved_0_(0U) {
    (void)memset(reserved_1_, 0, sizeof(reserved_1_));
  }
  TensorData(const TensorData&) = delete;
  TensorData(TensorData&& other) noexcept
      : addr_(other.addr_), manager_(other.manager_), size_(other.size_), placement_(other.placement_) {
    other.addr_ = nullptr;
    other.manager_ = nullptr;
    other.size_ = 0U;
    other.placement_ = kTensorPlacementEnd;
    reserved_0_ = other.reserved_0_;
    (void)memcpy(reserved_1_, other.reserved_1_, sizeof(reserved_1_));
  }
  TensorData& operator=(const TensorData& other) = delete;
  TensorData& operator=(TensorData&& other) noexcept {
    if (this != &other) {
      reserved_0_ = other.reserved_0_;
      (void)memcpy(reserved_1_, other.reserved_1_, sizeof(reserved_1_));
      const auto old_addr = addr_;
      const auto old_manager = manager_;
      const auto old_size = size_;
      const auto old_placement = placement_;
      addr_ = other.addr_;
      manager_ = other.manager_;
      size_ = other.size_;
      placement_ = other.placement_;
      other.addr_ = old_addr;
      other.manager_ = old_manager;
      other.size_ = old_size;
      other.placement_ = old_placement;
      if (other.Free() == ge::GRAPH_SUCCESS) {
        other.addr_ = nullptr;
        other.manager_ = nullptr;
        other.size_ = 0U;
        other.placement_ = kTensorPlacementEnd;
      }
    }
    return *this;
  }
  ~TensorData() noexcept {
    static_cast<void>(Free());
  }

  /**
   * 获取tensor地址
   * @return tensor地址
   */
  TensorAddress GetAddr() const {
    if (manager_ == nullptr || addr_ == nullptr) {
      return addr_;
    }
    TensorAddress addr;
    if (manager_(addr_, kGetTensorAddress, &addr) != ge::GRAPH_SUCCESS) {
      return nullptr;
    } else {
      return addr;
    }
  }

  /**
   * 获取tensor的内存大小
   * @return tensor所占内存大小
   */
  size_t GetSize() const {
    return size_;
  }
  /**
   * 设置tensor的内存大小
   * @param tensor的内存大小
   */
  void SetSize(const size_t size) {
    size_ = size;
  }

  /**
   * 获取tensor的placement
   * @return tensor的placement
   */
  TensorPlacement GetPlacement() const {
    return placement_;
  }
  /**
   * 设置tensor的placement
   * @param tensor的placement
   */
  void SetPlacement(const TensorPlacement placement) {
    placement_ = placement;
  }
  /**
   * 释放tensor
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus Free() {
    if (manager_ == nullptr) {
      return ge::GRAPH_SUCCESS;
    }
    const auto ret = manager_(addr_, kFreeTensor, nullptr);
    if (ret == ge::GRAPH_SUCCESS) {
      addr_ = nullptr;
      manager_ = nullptr;
    }
    return ret;
  }
  /**
   * 设置tensor地址
   * @param addr tensor地址
   * @param manager tensor的管理函数
   */
  ge::graphStatus SetAddr(const ConstTensorAddressPtr addr, TensorAddrManager manager) {
    if (addr_ == addr && manager_ == manager) {
      return ge::GRAPH_SUCCESS;
    }
    const auto ret = Free();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    addr_ = const_cast<TensorAddress>(addr);
    manager_ = manager;
    return ge::GRAPH_SUCCESS;
  }

  bool IsSharedWith(const TensorData& other) const {
    return (
        addr_ == other.addr_ && manager_ == other.manager_ && size_ == other.size_ && placement_ == other.placement_);
  }

  ge::graphStatus ShareFrom(const TensorData& other) {
    if (IsSharedWith(other)) {
      return ge::GRAPH_SUCCESS;
    }
    const auto ret = Free();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }

    addr_ = other.addr_;
    manager_ = other.manager_;
    size_ = other.size_;
    placement_ = other.placement_;
    if (manager_ == nullptr) {
      return ge::GRAPH_SUCCESS;
    }
    const auto share_ret = manager_(addr_, kPlusShareCount, nullptr);
    if (share_ret != ge::GRAPH_SUCCESS) {
      addr_ = nullptr;
      manager_ = nullptr;
      size_ = 0U;
      placement_ = kTensorPlacementEnd;
    }
    return share_ret;
  }

  /**
   * 释放对TensorAddress的所有权，本接口调用后，本对象不再管理TensorAddress，
   * 而且TensorAddress并没有被释放，因此调用者负责通过manager释放TensorAddress
   * @param manager 出参，用于管理返回的`TensorAddress`的函数，若本对象无所有权，那么manager被设置为nullptr
   * @return 本对象持有的`TensorAddress`指针，若本对象不持有任何指针，则返回nullptr
   */
  TensorAddress Release(TensorAddrManager& manager) {
    auto tmp_addr = addr_;
    manager = manager_;

    addr_ = nullptr;
    manager_ = nullptr;
    size_ = 0U;
    placement_ = kTensorPlacementEnd;

    return tmp_addr;
  }

 private:
  TensorAddress addr_;
  TensorAddrManager manager_;
  size_t size_;
  TensorPlacement placement_;
  uint32_t reserved_0_; // Reserved field, 8-byte aligned for TensorPlacement
  uint8_t reserved_1_[40]; // Reserved field, 32+8, do not directly use when only 8-byte left

  friend class TensorUtils;
};
} // namespace gert

#endif // METADEF_CXX_INC_EXE_GRAPH_TENSOR_DATA_H_
