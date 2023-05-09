#pragma once

#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/utils/NpuDataDumpMgr.h"

namespace c10_npu {
struct SecondaryStreamGuard {
  explicit SecondaryStreamGuard() = delete;
  // During datadump using one stream to run 
  // datadump used tdtchannel, queue does not support multiple streams
  explicit SecondaryStreamGuard(c10::Stream stream)
      : guard_(at_npu::native::NpuDataDumpMgr::GetInstance().IsDatadumpEnable()
                   ? c10_npu::getCurrentNPUStream()
                   : stream){};

  ~SecondaryStreamGuard() {
    c10_npu::NPUEvent npu_event;
    npu_event.record(guard_.current_stream());
    npu_event.block(guard_.original_stream());
  }

private:
  c10_npu::NPUStreamGuard guard_;
};
}  // namespace c10_npu
