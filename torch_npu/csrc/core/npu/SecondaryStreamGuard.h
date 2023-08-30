#pragma once

#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace c10_npu {
struct SecondaryStreamGuard {
  explicit SecondaryStreamGuard() = delete;
  explicit SecondaryStreamGuard(c10::Stream stream) : guard_(stream) {};

  ~SecondaryStreamGuard() {
    c10_npu::NPUEvent npu_event;
    npu_event.record(guard_.current_stream());
    npu_event.block(guard_.original_stream());
  }

private:
  c10_npu::NPUStreamGuard guard_;
};
}  // namespace c10_npu
