#ifndef PROFILER_NPU_INC
#define PROFILER_NPU_INC

#include <string>
#include <unordered_map>
#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/profiler/record_function.h"

namespace torch_npu {
namespace profiler {

std::unordered_map<std::string, c10::IValue> saveExtraArgs(const at::RecordFunction &fn);

uint64_t computeFlops(const std::string &op_name, const std::unordered_map<std::string, c10::IValue> &extra_args);

}
}

#endif // PROFILER_NPU_INC