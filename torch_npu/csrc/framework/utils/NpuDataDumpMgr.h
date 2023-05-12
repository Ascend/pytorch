#pragma once

#include <string>
#include <vector>

#include "third_party/acl/inc/op_proto/data_flow_ops.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/graph/util/TdtChannelForPrint.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {
class NpuDataDumpMgr {
public:
  ~NpuDataDumpMgr() {}
  NpuDataDumpMgr(const NpuDataDumpMgr &) = delete;
  NpuDataDumpMgr &operator=(const NpuDataDumpMgr &) = delete;
  static NpuDataDumpMgr &GetInstance() {
    static NpuDataDumpMgr instance;
    return instance;
  }

  void EnableDatadump(const c10::SmallVector<std::string, N> &opWhiteList, int64_t capacity);
  void DisableDatadump();
  bool IsDatadumpEnable() const;
  void DatadumpEnqueue(const at::TensorList &inputs,
                       const at::TensorList &outputs, const string &opName);

private:
  NpuDataDumpMgr() {}

  int GetDatadumpOpIdx(const std::string &opName);
  bool enableFlag_ = false;
  c10::SmallVector<std::string, N> opWhiteList_;
  int index_ = 0;
  int64_t capacity_ = 0;
};
}  // namespace native
}  // namespace at_npu
