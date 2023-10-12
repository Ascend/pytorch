#ifdef USE_RPC_FRAMEWORK

#pragma once

#include <torch/csrc/python_headers.h>

namespace torch_npu {
namespace distributed {
namespace rpc {

PyObject *rpc_npu_init(PyObject *_unused, PyObject *noargs);

}
} // namespace distributed
} // namespace torch_npu

#endif
