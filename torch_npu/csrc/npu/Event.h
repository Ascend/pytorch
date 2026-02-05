#ifndef THNP_EVENT_INC
#define THNP_EVENT_INC

#include <torch/csrc/Event.h>
#include <torch/csrc/python_headers.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUEvent.h"

struct THNPEvent : public THPEvent {
    c10_npu::NPUEvent npu_event;
};

extern PyObject *THNPEventClass;

TORCH_NPU_API void THNPEvent_init(PyObject *module);

inline bool THNPEvent_Check(PyObject* obj)
{
    return THNPEventClass && PyObject_IsInstance(obj, THNPEventClass);
}

c10_npu::NPUEvent* THNPUtils_PyObject_to_NPUEvent(PyObject* py_event);

#endif // THNP_EVENT_INC
