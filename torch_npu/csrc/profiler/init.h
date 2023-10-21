#ifndef PROFILER_INIT_INC
#define PROFILER_INIT_INC
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace torch_npu {
namespace profiler {
TORCH_NPU_API PyMethodDef* profiler_functions();
}
}

#endif // PROFILER_INIT_INC