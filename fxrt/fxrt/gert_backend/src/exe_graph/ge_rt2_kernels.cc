#include "exe_graph/ge_rt2_kernels.h"

#include <mutex>

#include "opp/opp_registry.h"

namespace {

std::once_flag g_ge_rt2_init_once;

void DoInitGeRt2Kernels() {
  (void)EnsureOppRegistryReady();
}

}  // namespace

void EnsureGeRt2KernelsReady() {
  std::call_once(g_ge_rt2_init_once, DoInitGeRt2Kernels);
}
