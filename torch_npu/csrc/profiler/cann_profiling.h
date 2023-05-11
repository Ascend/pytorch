#ifndef __CANN_PROFILING__
#define __CANN_PROFILING__

#include <string>

#include "torch_npu/csrc/framework/interface/AclInterface.h"

typedef enum {
  PROFILING_FINALIZE,
  PROFILING_INIT,
  PROFILING_START,
  PROFILING_STOP
} PROFILING_STATUS;

namespace torch_npu {
namespace profiler {

class NpuProfiling
{
public:
  static NpuProfiling& Instance();
  void Init(const std::string &profilerResultPath);
  void Start(uint64_t npu_event, uint64_t aicore_metrics);
  void Stop();
  void Finalize();
private:
  aclprofConfig* profCfg = nullptr;
  PROFILING_STATUS status = PROFILING_FINALIZE;
  NpuProfiling() = default;
  ~NpuProfiling() = default;
};

class NpuProfilingDispatch
{
public:
  static NpuProfilingDispatch& Instance();
  void start();
  void stop();
private:
  aclprofStepInfo* profStepInfo = nullptr;
  NpuProfilingDispatch() = default;
  ~NpuProfilingDispatch() = default;
  void init();
  void destroy();
};

}
}

#endif // __CANN_PROFILING__