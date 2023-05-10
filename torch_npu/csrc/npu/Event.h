#ifndef THNP_EVENT_INC
#define THNP_EVENT_INC

#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include <torch/csrc/python_headers.h>

struct THNPEvent {
  PyObject_HEAD
  c10_npu::NPUEvent npu_event;
};
extern PyObject *THNPEventClass;

void THNPEvent_init(PyObject *module);

inline bool THNPEvent_Check(PyObject* obj) {
  return THNPEventClass && PyObject_IsInstance(obj, THNPEventClass);
}

#endif // THNP_EVENT_INC
