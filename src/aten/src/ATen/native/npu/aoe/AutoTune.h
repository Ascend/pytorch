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

#ifndef __NATIVE_NPU_AOE_AUTOTUNE__
#define __NATIVE_NPU_AOE_AUTOTUNE__

#include <atomic>
#include <ATen/native/npu/interface/AoeInterface.h>
#include <ATen/native/npu/interface/Graph.h>
#include<c10/core/thread_pool.h>
namespace at {
namespace native {
namespace npu {
/** 
  class Autotune provide API for tuning.
  */
class AutotuneManager {

public:
  static AutotuneManager* GetInstance();
  void PushGraph(const std::string& name, Graph& tuningGraph);

private:
  std::atomic<bool> isInited;
  std::shared_ptr<TaskThreadPool> thread_pool_;
  std::vector<aoe::SessionId> sessionIDs;
  std::vector<ge::Graph> ge_graphs;
  std::map<ge::AscendString, ge::AscendString> sessionOptions;

  AutotuneManager();
  ~AutotuneManager();

  void Init(); 
  void DeInit();

  void DoGraphsTune();
  void DoGraphTune(ge::Graph& ge_graph, aoe::SessionId sessionId);
  void TuningGraphs();
  void WaitThreadsFinished();

  void CreatSessions();
  void DestroySessions();

};

} // namespace npu
} // namespace native
} // namespace at

#endif // __NATIVE_NPU_AOE_AUTOTUNE__