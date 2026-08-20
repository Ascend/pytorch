#pragma once

#include <string>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"

namespace at_npu {
namespace native {

enum class CpuFallbackKind {
  Dispatcher,
  ExplicitKernel,
};

inline void CheckCpuFallbackAllowed(
    const std::string& op_name,
    CpuFallbackKind kind,
    const std::string& reason = "") {
  if (c10_npu::option::OptionsManager::IsCpuFallbackEnable()) {
    return;
  }

  const char* fallback_kind = kind == CpuFallbackKind::Dispatcher ? "dispatcher backend fallback" : "explicit CPU kernel";
  TORCH_CHECK(
      false,
      "The operator '",
      op_name,
      "' would execute its main computation on CPU through ",
      fallback_kind,
      reason.empty() ? "" : " because ",
      reason,
      ", but CPU fallback is disabled by TORCH_NPU_FALLBACK_CPU_DISABLE=1.",
      OPS_ERROR(ErrCode::NOT_SUPPORT));
}

} // namespace native
} // namespace at_npu
