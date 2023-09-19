// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include <string>
#include <memory>
#include <mutex>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/framework/graph/tools/NPUTdtDataset.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/framework/interface/AclTdtInterface.h"
namespace c10_npu {
class NpuTdtChannel {
public:
  NpuTdtChannel() = default;
  NpuTdtChannel(int32_t time_out,
                int32_t capacity,
                const std::string& channel_name) :
          time_out_(time_out),
          capacity_(capacity),
          channel_name_(channel_name) {}
  virtual ~NpuTdtChannel();
  virtual bool Init();

  virtual std::shared_ptr<TdtDataSet> Dequeue();

  const std::string& GetChannelName() const {
    return channel_name_;
  }

  int32_t GetTimeOut() const {
    return time_out_;
  }

private:
  std::mutex channel_mutex_;
  bool inited_ = false;
  int32_t time_out_ = -1;
  uint32_t capacity_ = 0;
  int32_t device_id_ = 0;
  acltdtChannelHandle* channel_handle_ = nullptr;
  std::string channel_name_ = "DefaultChannel";
};
} // namespace c10_npu

