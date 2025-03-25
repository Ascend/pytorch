#ifndef PROFILER_NPU_INC
#define PROFILER_NPU_INC

#include <string>
#include <unordered_map>
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_compat.h>

#include "torch_npu/csrc/core/npu/NPUException.h"

namespace torch_npu {
namespace profiler {

std::unordered_map<std::string, c10::IValue> saveExtraArgs(const at::RecordFunction &fn);

uint64_t computeFlops(const std::string &op_name, const std::unordered_map<std::string, c10::IValue> &extra_args);

class NPURecordFunction {
public:
    NPURecordFunction(bool enable_ = false) : enable(enable_)
    {
        if (NPURecordFunction::use_npu_simple) {
            at::enableRecordFunction(enable);
        }
    }

    ~NPURecordFunction()
    {
        if (NPURecordFunction::use_npu_simple) {
            at::enableRecordFunction(!enable);
        }
    }
    bool enable = false;
    static bool use_npu_simple;
};

inline THPCodeObjectPtr PyFrame_GetCode_NPU(PyFrameObject *frame)
{
    return THPCodeObjectPtr(PyFrame_GetCode(frame));
}

inline PyFrameObject *PyEval_GetFrame_NPU()
{
    auto frame = PyEval_GetFrame();
    Py_XINCREF(frame);
    return frame;
}

inline THPObjectPtr PyFrame_GetLocals_NPU(PyFrameObject *frame)
{
    return THPObjectPtr(PyFrame_GetLocals(frame));
}
}
}

#endif // PROFILER_NPU_INC