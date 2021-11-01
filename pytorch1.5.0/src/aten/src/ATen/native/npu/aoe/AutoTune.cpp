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
// It is better to set MAX_TUNE_THREADS = 8~64, support set to be 1~64
constexpr std::size_t MAX_TUNE_THREADS = 8;
// tune mode: 0 -> model mode, 2 -> op mode, 4 -> grad mode
constexpr std::size_t AOE_TUNE_MODE = 2;

AutotuneManager* AutotuneManager::GetInstance() {
  static AutotuneManager instance;
  return &instance;
}

AutotuneManager::AutotuneManager()
    : isInited(false)
    , thread_pool_(std::make_shared<TaskThreadPool>(MAX_TUNE_THREADS)) {
  sessionOptions["job_type"] = std::to_string(AOE_TUNE_MODE).c_str();
}

AutotuneManager::~AutotuneManager() {
  DeInit();
}

void AutotuneManager::Init() {
  if (isInited) {
    return;
  }
  std::map<ge::AscendString, ge::AscendString> globalOptions;
  // graph tune parallel num, only support to be 1~64, default=8
  globalOptions["tuning_parallel_num"] = std::to_string(MAX_TUNE_THREADS).c_str();
  auto ret = aoe::initialize(globalOptions);
  if (ret) {
    TORCH_CHECK(ret, "aoe::initialize failed. error code:", ret);
    return;
  }
  isInited = true;
}

void AutotuneManager::DeInit() {
  if (isInited) {
    this->ge_graphs.clear();
    aoe::finalize();
    isInited = false;
  }
}

void AutotuneManager::CreatSessions(){
  for (int i=0; i<MAX_TUNE_THREADS; i++) {
    aoe::SessionId sessionId;
    auto ret = aoe::create_session(sessionOptions, sessionId);
    if (ret) {
      TORCH_CHECK(ret, "aoe::create_session failed. error code:", ret);
      return;
    }
    this->sessionIDs.emplace_back(sessionId);
  }
}

void AutotuneManager::DestroySessions(){
  TORCH_CHECK(this->sessionIDs.size() == MAX_TUNE_THREADS, "The AOE sessionID nums should be same to MAX_TUNE_THREADS!");
  for (auto it=this->sessionIDs.begin(); it<this->sessionIDs.end(); it++) {
    aoe::destroy_session(*it);
  }
  this->sessionIDs.clear();
}

void AutotuneManager::PushGraph(const std::string& name, Graph& tuningGraph) {
  if (not env::AutoTuneEnabled()) {
    return;
  }
  // TransData is not support tuning.
  if (name == "TransData") {
    return;
  }
  if (!isInited) {
    Init();
  }
  ge::Graph ge_graph;
  tuningGraph.GeGraph(ge_graph);

  ge_graphs.emplace_back(std::move(ge_graph));
  if (this->ge_graphs.size() < MAX_TUNE_THREADS) {
    return;
  }
  this->TuningGraphs();
}

void AutotuneManager::DoGraphsTune() {
  auto alive_thread_nums = thread_pool_->numAvailable();
  TORCH_CHECK(alive_thread_nums>=this->ge_graphs.size(), "The ge_graph size is greater than thread_pool_ size!");
  int tune_ops = -1;
  if (alive_thread_nums > 0) {
    for (auto it=this->ge_graphs.begin(); it<this->ge_graphs.begin() + alive_thread_nums && it<this->ge_graphs.end(); it++) {
      tune_ops += 1;
      thread_pool_->run(std::bind(
          &AutotuneManager::DoGraphTune,
          this,
          *it,
          this->sessionIDs[tune_ops]));
          // need to sleep some seconds in master thread to ensure all threads are up!
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}

void AutotuneManager::DoGraphTune(ge::Graph& ge_graph, aoe::SessionId sessionId) {
  auto ret = aoe::set_tuning_graph(sessionId, ge_graph);
  if (ret) {
    TORCH_CHECK(ret, "aoe::set_tuning_graph failed. error code:", ret);
    return;
  }
  std::map<ge::AscendString, ge::AscendString> tuingOptions;
  ret = aoe::tuning_graph(sessionId, tuingOptions);
  if (ret) {
    TORCH_CHECK(ret, "aoe::tuning_graph failed. error code:", ret);
    return;
  }
}

void AutotuneManager::TuningGraphs() {
  if (this->ge_graphs.size() == MAX_TUNE_THREADS && 
    thread_pool_->numAvailable() == MAX_TUNE_THREADS) {
      this->CreatSessions();
      this->DoGraphsTune();
      this->WaitThreadsFinished();
      this->ge_graphs.clear();
      this->DestroySessions();
    } else {
      TORCH_CHECK(false, "TuningGraphs failed, the size of ge_graphs"
      " and thread_pool's numAvailable should be same to MAX_TUNE_THREADS");
    }
}

void AutotuneManager::WaitThreadsFinished() {
  if (thread_pool_ != nullptr || thread_pool_->numAvailable() < MAX_TUNE_THREADS) {
    thread_pool_->waitWorkComplete();
  }
}

} // namespace npu
} // namespace native
} // namespace at