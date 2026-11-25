#pragma once

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace torch_npu {
namespace utils {
class C10_NPU_API OverflowUtil {
 public:
  ~OverflowUtil();

  static OverflowUtil* GetInstance() {
    static OverflowUtil instance;
    return &instance;
  }

  void EnableOverflowNpu();
  bool CheckOverflowNpu();
  void ClearOverflowNpu();

 private:
  OverflowUtil();
  bool hasOverflow = false;
};
} // namespace utils
} // namespace torch_npu
