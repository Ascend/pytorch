// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "third_party/acl/inc/acl/acl.h"
#include <cstdint>
#include <utility>

namespace c10_npu {
/*
* NPUEvents are movable not copyable wrappers around NPU's events.
* NPUEvents are constructed lazily when first recorded.
*/
struct C10_NPU_API NPUEvent {
    // Constructors
    // Default value for `flags` is specified below
    NPUEvent();
    NPUEvent(unsigned int flags) : flags_(flags) {}
    ~NPUEvent();

    NPUEvent(const NPUEvent&) = delete;
    NPUEvent& operator=(const NPUEvent&) = delete;

    NPUEvent(NPUEvent&& other);
    NPUEvent& operator=(NPUEvent&& other);

    operator aclrtEvent() const { return event(); }

    // aclrtEvent do not support Less than operator until now

    c10::optional<at::Device> device() const
    {
        if (is_created_) {
            return at::Device(at_npu::key::NativeDeviceType, device_index_);
        } else {
            return {};
        }
    }

    bool isCreated() const { return is_created_; }
    c10::DeviceIndex device_index() const { return device_index_; }
    aclrtEvent event() const { return event_; }

    bool query() const;
    void record();
    void recordOnce(const NPUStream& stream);
    void record(const NPUStream& stream);
    void block(const NPUStream& stream);
    float elapsed_time(const NPUEvent& other) const;
    void synchronize() const;

    // npu do not support IpcEventHandle until now

private:
    unsigned int flags_;
    bool is_created_ = false;
    bool was_recorded_ = false;
    c10::DeviceIndex device_index_ = -1;
    aclrtEvent event_ = nullptr;

    void createEvent(c10::DeviceIndex device_index);
    void moveHelper(NPUEvent&& other);
};

} // namespace c10_npu
