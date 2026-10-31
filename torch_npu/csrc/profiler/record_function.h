#pragma once

#include <ATen/record_function.h>

#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace torch_npu {
namespace profiler {

class TORCH_NPU_API NPURecordFunction {
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

} // namespace profiler
} // namespace torch_npu
