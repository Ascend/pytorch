#ifndef PROFILER_INIT_INC
#define PROFILER_INIT_INC
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace torch_npu {
namespace profiler {
enum class ExperConfigType {
    TRACE_LEVEL = 0,
    METRICS,
    L2_CACHE,
    RECORD_OP_ARGS,
    MSPROF_TX,
    OP_ATTR,
    MSTX_DOMAIN_INCLUDE,
    MSTX_DOMAIN_EXCLUDE
};

TORCH_NPU_API PyMethodDef* profiler_functions();
TORCH_NPU_API void initMstx(PyObject *module);
}
}

#endif // PROFILER_INIT_INC