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

