#pragma once

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
      : is_alive_(false),
        pid_(0),
        thread_name_("NPUProfiler") {};

  ~Thread() {
    if (is_alive_) {
        (void)pthread_cancel(pid_);
        (void)pthread_join(pid_, nullptr);
    }
  }

  void SetThreadName(const std::string &name) {
    if (!name.empty()) {
      thread_name_ = name;
    }
  }

  std::string GetThreadName() {
    return thread_name_;
  }

  int Start() {
    int ret = pthread_create(&pid_, nullptr, Execute, (void*)this);
    is_alive_ = (ret == 0) ? true : false;
    return ret;
  }

  int Stop() {
    return Join();
  }

  int Join() {
    int ret = pthread_join(pid_, nullptr);
    is_alive_ = (ret == 0) ? false : true;
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
  bool is_alive_;
  pthread_t pid_;
  std::string thread_name_;
};
} // profiler
} // toolkit
} // torch_npu
