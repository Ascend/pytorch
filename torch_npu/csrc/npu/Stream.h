#ifndef THNP_STREAM_INC
#define THNP_STREAM_INC

#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>

struct THNPStream : THPStream {
  c10_npu::NPUStream npu_stream;
};
extern PyObject *THNPStreamClass;

TORCH_NPU_API void THNPStream_init(PyObject *module);

inline bool THNPStream_Check(PyObject* obj) {
  return THNPStreamClass && PyObject_IsInstance(obj, THNPStreamClass);
}

TORCH_NPU_API std::vector<c10::optional<c10_npu::NPUStream>> THNPUtils_PySequence_to_NPUStreamList(PyObject* obj);

#endif // THNP_STREAM_INC
