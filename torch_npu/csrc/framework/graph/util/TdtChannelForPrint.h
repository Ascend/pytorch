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

#include <string>
#include <memory>
#include <tuple>
#include <mutex>

#include <c10/macros/Export.h>
#include <c10/util/intrusive_ptr.h>

#include "torch_npu/csrc/core/NPUTensorImpl.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/framework/graph/tools/NPUTdtDataset.h"
#include "torch_npu/csrc/framework/interface/AclTdtInterface.h"
#include "torch_npu/csrc/framework/graph/tools/NPUTdtChannel.h"
namespace at_npu {
namespace native {
using TupleToPrint = std::tuple<std::vector<at::Tensor>, std::string>;
class TdtChannelForPrint {
public:
  static TdtChannelForPrint& GetInstance();

  bool Init(int64_t capacity);

  void Finalize() {
    std::lock_guard<std::mutex> lock(channel_mutex_);
    if (channel_ != nullptr) {
      delete channel_;
      channel_ = nullptr;
    }
  }

  const std::string& GetChannelName(int64_t capacity) {
    if (channel_ == nullptr) {
      this->Init(capacity);
    }
    TORCH_CHECK(channel_ != nullptr, "Channel is none during GetChannelName");
    return channel_->GetChannelName();
  }

  TupleToPrint GetTupleToPrint();

  TdtChannelForPrint(const TdtChannelForPrint& other) = delete;
  TdtChannelForPrint& operator=(const TdtChannelForPrint& other) = delete;
  TdtChannelForPrint(TdtChannelForPrint&& other) = delete;
  TdtChannelForPrint& operator=(TdtChannelForPrint&& other) = delete;

private:
  std::mutex channel_mutex_;
  c10_npu::NpuTdtChannel* channel_ = nullptr;

  TdtChannelForPrint() = default;
  std::shared_ptr<c10_npu::TdtDataSet> GetNextDatasetToPrint();
};
} // namespace native
} // namespace at_npu

