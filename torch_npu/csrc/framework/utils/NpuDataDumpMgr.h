// Copyright (c) 2022 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
