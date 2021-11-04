// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION. 
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <mutex>
#include "NPUQueue.h"
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/npu/NPUException.h>
#include <c10/npu/NPUMacros.h>
#include <c10/npu/npu_log.h>
#include <c10/util/Exception.h>
#include "c10/npu/npu_log.h"
#include <third_party/acl/inc/acl/acl_op.h>

namespace c10 {
namespace npu {

class C10_NPU_API NPUStream {
 public:
  enum Unchecked { UNCHECKED };

  explicit NPUStream(Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == DeviceType::NPU);
  }

  explicit NPUStream(Unchecked, Stream stream) : stream_(stream) {}

  ~NPUStream(){}

  bool operator==(const NPUStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const NPUStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  /// Implicit conversion to rtStream_t.
  operator aclrtStream() const {
    return stream();
  }

  /// Implicit conversion to pytorch Stream.
  operator Stream() const {
    return unwrap();
  }

  /// Get the NPU device index that this stream is associated with.
  DeviceIndex device_index() const {
    return stream_.device_index();
  }

  /// Get the full Device that this stream is associated with.  The Device
  /// is guaranteed to be a NPU device.
  Device device() const {
    return Device(DeviceType::NPU, device_index());
  }

  StreamId id() const {
    return stream_.id();
  }

  /*
  bool query() const {
      DeviceGuard guard{stream_.device()};
      aclError err = aclrtQueryStream(stream());

      if (err == ACL_ERROR_NONE) {
          return true;
      } else if (err != ACL_ERROR_NOT_READY) {
          C10_NPU_CHECK(err);
      }

      return false;
  } */

  void synchronize() const {
    DeviceGuard guard{stream_.device()};
    C10_NPU_CHECK(aclrtSynchronizeStream(stream()));
  }

  /// Explicit conversion to rtStream_t.
  C10_API aclrtStream stream() const;

  /// Explicit conversion to Stream.
  Stream unwrap() const {
    return stream_;
  }

  uint64_t pack() const noexcept {
    return stream_.pack();
  }

  static NPUStream unpack(uint64_t bits) {
    return NPUStream(Stream::unpack(bits));
  }

 private:
  Stream stream_;
};

CAFFE2_API NPUStream getNPUStreamFromPool(DeviceIndex device = -1);

CAFFE2_API NPUStream getDefaultNPUStream(DeviceIndex device_index = -1);

CAFFE2_API NPUStream getCurrentNPUStream(DeviceIndex device_index = -1);

CAFFE2_API NPUStream getCurrentSecondaryStream(DeviceIndex device_index = -1);

CAFFE2_API aclrtStream getCurrentNPUStreamNoWait(DeviceIndex device_index = -1);

CAFFE2_API void npuSynchronizeDevice();

CAFFE2_API void enCurrentNPUStream(
    void* cur_paras,
    SmallVector<Storage, N>& needClearVec,
    DeviceIndex device_index = -1);

CAFFE2_API void setCurrentNPUStream(NPUStream stream);

C10_API std::ostream& operator<<(std::ostream& stream, const NPUStream& s);

} // namespace npu
} // namespace c10

namespace std {
template <>
struct hash<c10::npu::NPUStream> {
  size_t operator()(c10::npu::NPUStream s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
} // namespace std
