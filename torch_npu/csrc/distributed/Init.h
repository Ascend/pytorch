#pragma once

#include <torch/csrc/python_headers.h>

namespace torch_npu {
namespace distributed {

PyMethodDef* python_functions();

} // namespace distributed
} // namespace torch_npu