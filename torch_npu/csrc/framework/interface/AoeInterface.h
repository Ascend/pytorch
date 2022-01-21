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

#ifndef __PLUGIN_NATIVE_NPU_INTERFACE_AOEINTERFACE__
#define __PLUGIN_NATIVE_NPU_INTERFACE_AOEINTERFACE__

#include "third_party/acl/inc/graph/ascend_string.h"
#include "third_party/acl/inc/graph/graph.h"

namespace at_npu {
namespace native {
namespace aoe {

/**
  SessionId is provide by aoe, it used to store session id.
  */
using SessionId = uint64_t;
/**
  AoeStatues is provide by aoe, it used to store the return value.
  */
using AoeStatus = int32_t;

/**
  This API is used to init aoe, it need be called once at process.
  */
AoeStatus initialize(const std::map<ge::AscendString, ge::AscendString> &globalOptions);
/**
  This API is used to finalize aoe, it need be called once at process.
  */
AoeStatus finalize();
/**
  This API is used to create session, this operation should be called after init.
  */
AoeStatus create_session(const std::map<ge::AscendString, ge::AscendString> &sessionOptions, SessionId &sessionId);
/**
  This API is used to destroy session
  */
AoeStatus destroy_session(SessionId sessionId);
/**
  This API is used to associate session and graph
  */
AoeStatus set_tuning_graph(SessionId sessionId, ge::Graph &tuningGraph);
/**
  This API is used to tuning graphs at session
  */
AoeStatus tuning_graph(SessionId sessionId, const std::map<ge::AscendString, ge::AscendString> &tuingOptions);

} // namespace aoe
} // namespace native
} // namespace at_npu

#endif // __NATIVE_NPU_INTERFACE_AOEINTERFACE__