#include "torch_npu/csrc/framework/utils/NpuDataDumpMgr.h"

#include <algorithm>
#include <map>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
void NpuDataDumpMgr::DatadumpEnqueue(const at::TensorList &inputs,
                                     const at::TensorList &outputs,
                                     const string &opName) {
  if (!enableFlag_) {
    return;
  }
  int idx = NpuDataDumpMgr::GetDatadumpOpIdx(opName);
  if (idx < 0) {
    return;
  }
  ASCEND_LOGI("Datadump enque: %s", opName.c_str());
  enableFlag_ = false;
  string tensorName = std::to_string(idx) + '_' + opName;
  if (!inputs.empty()) {
    at_npu::native::NPUNativeFunctions::npu_enque_tensor(
        inputs, tensorName + "_input", capacity_);
  }
  if (!outputs.empty()) {
    at_npu::native::NPUNativeFunctions::npu_enque_tensor(
        outputs, tensorName + "_output", capacity_);
  }
  enableFlag_ = true;
}

void NpuDataDumpMgr::EnableDatadump(
    const c10::SmallVector<std::string, N> &opWhiteList, int64_t capacity) {
  ASCEND_LOGI("Datadump enable.");
  opWhiteList_ = opWhiteList;
  enableFlag_ = true;
  capacity_ = capacity;
}
void NpuDataDumpMgr::DisableDatadump() {
  ASCEND_LOGI("Datadump disable.");
  enableFlag_ = false;
}

bool NpuDataDumpMgr::IsDatadumpEnable() const { return enableFlag_; }

int NpuDataDumpMgr::GetDatadumpOpIdx(const std::string &opName) {
  if (opWhiteList_.empty() ||
      (std::find(opWhiteList_.begin(), opWhiteList_.end(), opName) !=
       opWhiteList_.end())) {
    return index_++;
  }
  return -1;
}
}  // namespace native
}  // namespace at_npu
