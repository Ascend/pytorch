#pragma once

#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_compat.h>

namespace torch_npu {
namespace profiler {

inline THPCodeObjectPtr PyFrame_GetCode_NPU(PyFrameObject* frame)
{
    return THPCodeObjectPtr(PyFrame_GetCode(frame));
}

inline PyFrameObject* PyEval_GetFrame_NPU()
{
    auto frame = PyEval_GetFrame();
    Py_XINCREF(frame);
    return frame;
}

inline THPObjectPtr PyFrame_GetLocals_NPU(PyFrameObject* frame)
{
    return THPObjectPtr(PyFrame_GetLocals(frame));
}

} // namespace profiler
} // namespace torch_npu
