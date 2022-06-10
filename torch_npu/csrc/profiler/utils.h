// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION. 
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

#ifndef PROFILER_NPU_INC
#define PROFILER_NPU_INC

#include <string>
#include <unordered_map>
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <c10/util/Exception.h>

namespace torch_npu {
namespace profiler{

std::unordered_map<std::string, c10::IValue> saveExtraArgs(const at::RecordFunction& fn);

uint64_t computeFlops(const std::string &op_name, 
    const std::unordered_map<std::string, c10::IValue> &extra_args);

class NPURecordFunction {
public:
  NPURecordFunction(bool enable_ = false) : enable(enable_) {
    if (NPURecordFunction::use_npu_simple) {
      at::enableRecordFunction(enable);
    }
  }

  ~NPURecordFunction(){
    if (NPURecordFunction::use_npu_simple) {
      at::enableRecordFunction(!enable);
    }
  }
  bool enable = false;
  static bool use_npu_simple;
};

}
}

#endif // PROFILER_NPU_INC