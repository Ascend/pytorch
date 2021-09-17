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


#include "AoeInterface.h"
#include "c10/npu/register/FunctionLoader.h"
#include "c10/util/Exception.h"
#include <iostream>

namespace at {
namespace native {
namespace npu {
namespace aoe {


#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libaoe_tuning, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)              \
  GET_FUNCTION(libaoe_tuning, funcName)

REGISTER_LIBRARY(libaoe_tuning)
LOAD_FUNCTION(AoeInitialize)
LOAD_FUNCTION(AoeFinalize)
LOAD_FUNCTION(AoeCreateSession)
LOAD_FUNCTION(AoeDestroySession)
LOAD_FUNCTION(AoeSetTuningGraph)
LOAD_FUNCTION(AoeTuningGraph)



AoeStatus initialize(const std::map<ge::AscendString, ge::AscendString> &globalOptions) {
  typedef AoeStatus(*aoeInitFunc)(const std::map<ge::AscendString, ge::AscendString>&);
  static aoeInitFunc func = nullptr;
  if (func == nullptr) {
    func = (aoeInitFunc)GET_FUNC(AoeInitialize);
  }
  TORCH_CHECK(func, "Failed to find function ", "AoeInitialize");
  auto ret = func(globalOptions);
  return ret;
}

AoeStatus finalize() {
  typedef AoeStatus(*aoeFinalizeFunc)();
  static aoeFinalizeFunc func = nullptr;
  if (func == nullptr) {
    func = (aoeFinalizeFunc)GET_FUNC(AoeFinalize);
  }
  TORCH_CHECK(func, "Failed to find function ", "AoeFinalize");
  auto ret = func();
  return ret;
}

AoeStatus create_session(const std::map<ge::AscendString, ge::AscendString> &sessionOptions, SessionId &sessionId) {
  typedef AoeStatus(*aoeCreateSession)(const std::map<ge::AscendString, ge::AscendString> &, SessionId &);
  aoeCreateSession func = nullptr;
  if (func == nullptr) {
    func = (aoeCreateSession)GET_FUNC(AoeCreateSession);
  }
  TORCH_CHECK(func, "Failed to find function ", "AoeCreateSession");
  auto ret = func(sessionOptions, sessionId);
  return ret;
}

AoeStatus destroy_session(SessionId sessionId) {
  typedef AoeStatus(*aoeDestroySession)(SessionId);
  aoeDestroySession func = nullptr;
  if (func == nullptr) {
    func = (aoeDestroySession)GET_FUNC(AoeDestroySession);
  }
  TORCH_CHECK(func, "Failed to find function ", "AoeDestroySession");
  auto ret = func(sessionId);
  return ret;
}

AoeStatus set_tuning_graph(SessionId sessionId, ge::Graph &tuningGraph) {
  typedef AoeStatus(*aoeSetTuningGraph)(SessionId, ge::Graph &);
  aoeSetTuningGraph func = nullptr;
  if (func == nullptr) {
    func = (aoeSetTuningGraph)GET_FUNC(AoeSetTuningGraph);
  }
  TORCH_CHECK(func, "Failed to find function ", "AoeSetTuningGraph");
  auto ret = func(sessionId, tuningGraph);
  return ret;
}

AoeStatus tuning_graph(SessionId sessionId, const std::map<ge::AscendString, ge::AscendString> &tuingOptions) {
  typedef AoeStatus(*aoeTuningGraph)(SessionId, const std::map<ge::AscendString, ge::AscendString> &);
  aoeTuningGraph func = nullptr;
  if (func == nullptr) {
    func = (aoeTuningGraph)GET_FUNC(AoeTuningGraph);
  }
  TORCH_CHECK(func, "Failed to find function ", "AoeTuningGraph");
  auto ret = func(sessionId, tuingOptions);
  return ret;
}

} // namespace aoe
} // namespace npu
} // namespace native
} // namespace at
