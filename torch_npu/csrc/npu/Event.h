#ifndef THNP_EVENT_INC
#define THNP_EVENT_INC

#include <torch/csrc/python_headers.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUEvent.h"

struct THNPEvent {
    PyObject_HEAD
    c10_npu::NPUEvent npu_event;
};

extern PyObject *THNPEventClass;

TORCH_NPU_API void THNPEvent_init(PyObject *module);

inline bool THNPEvent_Check(PyObject* obj)
{
    return THNPEventClass && PyObject_IsInstance(obj, THNPEventClass);
}

#endif // THNP_EVENT_INC
