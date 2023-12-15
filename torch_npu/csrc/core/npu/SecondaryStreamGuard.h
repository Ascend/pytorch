#pragma once

#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace c10_npu {
struct SecondaryStreamGuard {
  explicit SecondaryStreamGuard() = delete;
  explicit SecondaryStreamGuard(c10::Stream stream) : guard_(stream) {};

  ~SecondaryStreamGuard();

private:
  c10_npu::NPUStreamGuard guard_;
};
}  // namespace c10_npu
