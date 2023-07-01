#pragma once
#include <c10/core/TensorOptions.h>
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"

namespace torch_npu {
namespace utils {

void npu_lazy_init();

void npu_set_run_yet_variable_to_false();

}
}