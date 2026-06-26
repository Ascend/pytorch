#ifndef THNP_SHAPE_HANDLING_INC
#define THNP_SHAPE_HANDLING_INC

#include <torch/csrc/python_headers.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

TORCH_NPU_API void THNPShapeHandling_init(PyObject* module);

#endif // THNP_SHAPE_HANDLING_INC
