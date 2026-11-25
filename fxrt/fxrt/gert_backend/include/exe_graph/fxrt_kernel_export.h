#pragma once

#include <cstdint>

#include "exe_graph/runtime/kernel_context.h"

#define FXRT_EXPORT_KERNEL_RUN(name, implementation)                                                           \
  extern "C" __attribute__((visibility("default"))) uint32_t fxrt_##name##_Run(gert::KernelContext* context) { \
    return static_cast<uint32_t>(implementation(context));                                                     \
  }

#define FXRT_EXPORT_KERNEL_OUTPUTS(name, implementation)                                                          \
  extern "C"                                                                                                      \
      __attribute__((visibility("default"))) uint32_t fxrt_##name##_CreateOutputs(gert::KernelContext* context) { \
    return static_cast<uint32_t>(implementation(context));                                                        \
  }
