// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
#ifndef TORCH_NPU_TOOLKIT_PROFILER_THREAD_INC
#define TORCH_NPU_TOOLKIT_PROFILER_THREAD_INC

#include <signal.h>
#include <sys/prctl.h>
#include <pthread.h>
#include <string>
namespace torch_npu {
namespace toolkit {
namespace profiler {
class Thread {
public:
  Thread()
    : isAlive_(false),
      pid_(0),
      threadName_("NPUProfiler") {};

  ~Thread() {
    if (isAlive_) {
      pthread_kill(pid_, 0);
    }
  }

  void SetThreadName(const std::string &name) {
    if (!name.empty()) {
      threadName_ = name;
    }
  }

  std::string GetThreadName() {
    return threadName_;
  }

  int Start() {
    int ret = pthread_create(&pid_, nullptr, Execute, (void*)this);
    isAlive_ = (ret == 0) ? true : false;
    return ret;
  }

  int Stop() {
    return Join();
  }

  int Join() {
    int ret = pthread_join(pid_, nullptr);
    isAlive_ = (ret == 0) ? false : true;
    return ret;
  }

private:
  static void* Execute(void *args) {
    Thread *thr = (Thread *)args;
    prctl(PR_SET_NAME, (unsigned long)thr->GetThreadName().data());
    thr->Run();
    return nullptr;
  }
  virtual void Run() = 0;

private:
  bool isAlive_;
  pthread_t pid_;
  std::string threadName_;
};
} // profiler
} // toolkit
} // torch_npu
#endif
