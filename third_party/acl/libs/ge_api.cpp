// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include "ge/ge_api.h"

namespace ge {
Status Session::RunGraphWithStreamAsync(
    uint32_t graph_id,
    void* stream,
    const std::vector<Tensor>& inputs,
    std::vector<Tensor>& outputs) {

    sessionId_ = -1;
  return ge::SUCCESS;
}

Session::Session(const std::map<AscendString, AscendString>& options) {}

Session::~Session() {}

Status Session::AddGraph(uint32_t graphId, const Graph& graph) {
  return ge::SUCCESS;
}

Status GEInitialize(const std::map<AscendString, AscendString>& options) {
  return ge::SUCCESS;
}

Status GEFinalize() {
  return ge::SUCCESS;
}
} // namespace ge