#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/profiler/cann_profiling.h"

namespace at_npu {
namespace native {
namespace env {

REGISTER_OPTION_HOOK(deliverswitch, [](const std::string &val) {
  if (val == "enable") {
    torch_npu::profiler::NpuProfilingDispatch::Instance().start();
  } else {
    torch_npu::profiler::NpuProfilingDispatch::Instance().stop();
  }
})

REGISTER_OPTION_HOOK(profilerResultPath, [](const std::string &val) {
  torch_npu::profiler::NpuProfiling::Instance().Init(val);
})

REGISTER_OPTION_HOOK(profiling, [](const std::string &val) {
  if (val.compare("stop") == 0) {
    torch_npu::profiler::NpuProfiling::NpuProfiling::Instance().Stop();
  } else if (val.compare("finalize") == 0) {
    torch_npu::profiler::NpuProfiling::NpuProfiling::Instance().Finalize();
  } else {
    TORCH_CHECK(false, "profiling input: (", val, " ) error!")
  }
})

}
}
}
