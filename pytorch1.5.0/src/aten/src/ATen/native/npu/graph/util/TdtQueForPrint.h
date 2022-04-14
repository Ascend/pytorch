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
#include <c10/core/TensorImpl.h>
#include <c10/macros/Export.h>
#include <c10/npu/NPUException.h>
#include <c10/npu/interface/AclInterface.h>
#include <c10/npu/interface/AclTdtInterface.h>
#include <c10/npu/tools/NPUTdtDataset.h>
#include <c10/npu/tools/NPUTdtChannelQueue.h>
#include <c10/util/intrusive_ptr.h>
namespace at {
namespace native {
namespace npu {
using TupleToPrint = std::tuple<std::vector<Tensor>, std::string>;
class TORCH_NPU_API TdtQueForPrint {
public:
  static TdtQueForPrint& GetInstance();

  bool Init();

  void Finalize() {
    std::lock_guard<std::mutex> lock(que_mutex_);
    if (que_ != nullptr) {
      delete que_;
      que_ = nullptr;
    }
  }

  const std::string& GetChannelName() const {
    TORCH_CHECK(que_ != nullptr, "Que is none during GetChannelName");
    return que_->GetChannelName();
  }

  TupleToPrint GetTupleToPrint();

  TdtQueForPrint(const TdtQueForPrint& other) = delete;
  TdtQueForPrint& operator=(const TdtQueForPrint& other) = delete;
  TdtQueForPrint(TdtQueForPrint&& other) = delete;
  TdtQueForPrint& operator=(TdtQueForPrint&& other) = delete;

private:
  std::mutex que_mutex_;
  c10::npu::NpuTdtChannelQue* que_;

  TdtQueForPrint();
  std::shared_ptr<c10::npu::TdtDataSet> GetNextDatasetToPrint();
};
}
}
}