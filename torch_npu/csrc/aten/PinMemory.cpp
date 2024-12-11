#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <c10/core/Storage.h>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/CachingHostAllocator.h"

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/DispatchKeyExtractor.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#else
#include <ATen/ops/is_pinned_ops.h>
#include <ATen/ops/_pin_memory_ops.h>

${ops_headers}
#endif

namespace at_npu {
namespace native {

bool is_pinned(const at::Tensor& self, c10::optional<at::Device> device) {
  // Only CPU tensors can be pinned
  if (!self.is_cpu()) {
    return false;
  }
  c10::DispatchKeySet _dk = c10::DispatchKeySet(c10::computeDispatchKey(c10::nullopt, self.layout(), device.value_or(c10::DeviceType::PrivateUse1)));
  return at::_ops::is_pinned::redispatch(_dk, self, device);
}

at::Tensor _pin_memory(const at::Tensor& self, c10::optional<at::Device> device) {
  TORCH_CHECK(self.device().is_cpu(), "cannot pin '", self.toString(), "' only dense CPU tensors can be pinned", PTA_ERROR(ErrCode::TYPE));
  c10::DispatchKeySet _dk = c10::DispatchKeySet(c10::computeDispatchKey(c10::nullopt, self.layout(), device.value_or(c10::DeviceType::PrivateUse1)));
  return at::_ops::_pin_memory::redispatch(_dk, self, device);
}

class IgnoreWarningHandler : public c10::WarningHandler {
public:

  void process(const c10::Warning& warning) {
    ;
  }
};

c10::WarningHandler* getIgnoreHandler() {
  static IgnoreWarningHandler handler_ = IgnoreWarningHandler();
  return &handler_;
};

// use to ignore the warning info when overriding operator for CPU-implement
#define WITH_IGNORE_WARNING_OVERRIDE_OPERATOR(enable)                         \
  int enter_warning() {                                                       \
    if (enable) {                                                             \
      c10::WarningUtils::set_warning_handler(getIgnoreHandler());             \
    }                                                                         \
    return 1;                                                                 \
  }                                                                           \
  static int _temp_enter_warning = enter_warning();                           \
  TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {                                \
    m.impl(TORCH_SELECTIVE_NAME("aten::is_pinned"), TORCH_FN(is_pinned));     \
    m.impl(TORCH_SELECTIVE_NAME("aten::_pin_memory"), TORCH_FN(_pin_memory)); \
  }                                                                           \
  int exit_warning() {                                                        \
    if (enable) {                                                             \
      c10::WarningUtils::set_warning_handler(nullptr);                        \
    }                                                                         \
    return 1;                                                                 \
  }                                                                           \
  static int _temp_exit_warning = exit_warning();

WITH_IGNORE_WARNING_OVERRIDE_OPERATOR(true)

}
}
