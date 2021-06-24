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


#include "AutoTune.h"
#include "c10/util/Exception.h"
#include <ATen/native/npu/interface/AoeInterface.h>
#include <ATen/native/npu/interface/EnvVariables.h>
#include <ATen/native/npu/interface/GeHelper.h>

namespace at {
namespace native {
namespace npu {

AutoTune* AutoTune::GetInstance() {
  static AutoTune instance;
  return &instance;
}

AutoTune::AutoTune()
  : isInited(false){
}

AutoTune::~AutoTune() {
  if (isInited) {
    DeInit();
  }
}

void AutoTune::Init() {
  std::map<ge::AscendString, ge::AscendString> globalOptions;
  auto ret = aoe::initialize(globalOptions);
  if (ret) {
    TORCH_CHECK(ret, "aoe::initialize failed. error code:", ret);
    return;
  }
  std::map<ge::AscendString, ge::AscendString> sessionOptions;
  sessionOptions["job_type"] = "2";
  ret = aoe::create_session(sessionOptions, this->sessionId);
  if (ret) {
    TORCH_CHECK(ret, "aoe::create_session failed. error code:", ret);
    return;
  }
}

void AutoTune::DeInit() {
  aoe::destroy_session(this->sessionId);
  aoe::finalize();
}

void AutoTune::Do(const std::string& name, Graph& tuningGraph) {
  if (not env::AutoTuneEnabled()) {
    return;
  }

  // TransData is not support tuning.
  if (name == "TransData") {
    return;
  }

  if (!isInited) {
    Init();
    isInited = true;
  }

  ge::Graph graph;
  tuningGraph.GeGraph(graph);
  auto ret = aoe::set_tuning_graph(this->sessionId, graph);
  if (ret) {
    TORCH_CHECK(ret, "aoe::set_tuning_graph failed. error code:", ret);
    return;
  }
  std::map<ge::AscendString, ge::AscendString> tuingOptions;
  ret = aoe::tuning_graph(this->sessionId, tuingOptions);
  if (ret) {
    TORCH_CHECK(ret, "aoe::tuning_graph failed. error code:", ret);
    return;
  }
}

} // namespace npu
} // namespace native
} // namespace at
