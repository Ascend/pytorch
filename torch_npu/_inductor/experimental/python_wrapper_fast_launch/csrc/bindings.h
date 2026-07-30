#pragma once

#ifndef BUILD_LIBTORCH

#include <Python.h>

#include "torch_npu/csrc/core/npu/NPUMacros.h"

TORCH_NPU_API void RegisterNPUFastLaunchBindings(PyObject* module);

#endif
