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
#include "third_party/acl/inc/acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUQueue.h"
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include "torch_npu/csrc/core/npu/NPUException.h"
#include <c10/util/SmallVector.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include <c10/util/Exception.h>
#include "torch_npu/csrc/core/npu/npu_log.h"

#include "third_party/acl/inc/acl/acl_op.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace c10_npu {

class C10_NPU_API NPUStream {
public:
    enum Unchecked { UNCHECKED };

    explicit NPUStream(c10::Stream stream) : stream_(stream)
    {
        TORCH_CHECK(stream_.device_type() == at_npu::key::NativeDeviceType, PTA_ERROR(ErrCode::PARAM));
    }

    explicit NPUStream(Unchecked, c10::Stream stream) : stream_(stream) {}

    ~NPUStream() {}

    bool operator==(const NPUStream& other) const noexcept
    {
        return unwrap() == other.unwrap();
    }

    bool operator!=(const NPUStream& other) const noexcept
    {
        return unwrap() != other.unwrap();
    }

    // Implicit conversion to rtStream_t.
    operator aclrtStream() const
    {
        return stream();
    }

    // Implicit conversion to pytorch Stream.
    operator c10::Stream() const
    {
        return unwrap();
    }

    // Get the NPU device index that this stream is associated with.
    c10::DeviceIndex device_index() const
    {
        return stream_.device_index();
    }

    // Get the full Device that this stream is associated with.  The Device
    // is guaranteed to be a NPU device.
    c10::Device device() const
    {
        return c10::Device(at_npu::key::NativeDeviceType, device_index());
    }

    c10::StreamId id() const
    {
        return stream_.id();
    }

    bool query() const
    {
        c10::DeviceGuard guard{stream_.device()};
        acl::aclrtStreamStatus status = acl::ACL_STREAM_STATUS_RESERVED;
        NPU_CHECK_ERROR(acl::AclrtStreamQuery(stream(), &status));
        if (status == acl::ACL_STREAM_STATUS_COMPLETE) {
            return true;
        }
        return false;
    }

    void synchronize() const
    {
        c10::DeviceGuard guard{stream_.device()};
        NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream()));
    }

    // Explicit conversion to rtStream_t.
    aclrtStream stream() const;

    // Explicit conversion to Stream.
    c10::Stream unwrap() const
    {
        return stream_;
    }

    uint64_t pack() const noexcept
    {
        return stream_.pack();
    }

    static NPUStream unpack(uint64_t bits)
    {
        return NPUStream(c10::Stream::unpack(bits));
    }

    void setDataPreprocessStream(bool is_data_preprocess_stream);

    bool isDataPreprocessStream();

    void setSyncLaunchStream(bool is_sync_launch);

    bool isSyncLaunchStream() const;

    // Explicit conversion to rtStream_t, with out empty taskQ.
    aclrtStream stream(const bool need_empty) const;

private:
    c10::Stream stream_;
};

C10_NPU_API NPUStream getNPUStreamFromPool(c10::DeviceIndex device = -1);

C10_NPU_API NPUStream getDefaultNPUStream(c10::DeviceIndex device_index = -1);

C10_NPU_API NPUStream getCurrentNPUStream(c10::DeviceIndex device_index = -1);

NPUStream getCurrentSecondaryStream(c10::DeviceIndex device_index = -1);

aclrtStream getCurrentNPUStreamNoWait(c10::DeviceIndex device_index = -1);

NPUStatus emptyAllNPUStream();

std::string getRepoInfo();

C10_NPU_API bool npuSynchronizeDevice(bool check_error = true);

void enCurrentNPUStream(void* cur_paras, c10::DeviceIndex device_index = -1);

C10_NPU_API void setCurrentNPUStream(NPUStream stream);

C10_NPU_API  bool StreamInitFlag();

std::ostream& operator<<(std::ostream& stream, const NPUStream& s);

} // namespace c10_npu

namespace std {
template <>
struct hash<c10_npu::NPUStream> {
    size_t operator()(c10_npu::NPUStream s) const noexcept
    {
        return std::hash<c10::Stream>{}(s.unwrap());
    }
};
} // namespace std
