#pragma once

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace torch_npu {
namespace utils {

class OverflowUtil {
public:
  ~OverflowUtil();

  static OverflowUtil *GetInstance() {
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

}
}
