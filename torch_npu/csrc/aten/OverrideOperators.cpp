#include "torch_npu/csrc/aten/OverrideOperators.h"
#include "torch_npu/csrc/core/npu/NPURunMode.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "op_plugin/OpInterface.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/profiler/utils.h"
#endif

namespace at_npu {
namespace native {

at::Tensor wrapper___embedding_bag_dense_backward(
    const at::Tensor & grad,
    const at::Tensor & indices,
    const at::Tensor & offset2bag,
    const at::Tensor & bag_size,
    const at::Tensor & maximum_indices,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const c10::optional<at::Tensor> & per_sample_weights,
    int64_t padding_idx) {
    // No device check
  // DeviceGuard omitted
#ifndef BUILD_LIBTORCH
torch_npu::profiler::NPURecordFunction guard;
#endif

  return at_npu::native::NPUNativeFunctions::_embedding_bag_dense_backward(
      grad, indices, offset2bag, bag_size, maximum_indices, num_weights,
      scale_grad_by_freq, mode, per_sample_weights, padding_idx);
}

at::Tensor wrapper__nan_to_num(
    const at::Tensor& self,
    c10::optional<double> nan,
    c10::optional<double> posinf,
    c10::optional<double> neginf) {
  // No device check
  // DeviceGuard omitted
#ifndef BUILD_LIBTORCH
  torch_npu::profiler::NPURecordFunction guard;
#endif

  return at_npu::native::NPUNativeFunctions::nan_to_num(self, nan, posinf, neginf);
}

at::Tensor& wrapper_out_nan_to_num_out(
    const at::Tensor& self,
    c10::optional<double> nan,
    c10::optional<double> posinf,
    c10::optional<double> neginf,
    at::Tensor& out) {
  // No device check
  // DeviceGuard omitted
#ifndef BUILD_LIBTORCH
  torch_npu::profiler::NPURecordFunction guard;
#endif

  return at_npu::native::NPUNativeFunctions::nan_to_num_out(self, nan, posinf, neginf, out);
}

at::Tensor& wrapper__nan_to_num_(
    at::Tensor& self,
    c10::optional<double> nan,
    c10::optional<double> posinf,
    c10::optional<double> neginf) {
  // No device check
  // DeviceGuard omitted
#ifndef BUILD_LIBTORCH
  torch_npu::profiler::NPURecordFunction guard;
#endif

  return at_npu::native::NPUNativeFunctions::nan_to_num_(self, nan, posinf, neginf);
}

at::Tensor wrapper__argmin(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  // No device check
  // DeviceGuard omitted
#ifndef BUILD_LIBTORCH
torch_npu::profiler::NPURecordFunction guard;
#endif
  return op_plugin::argmin(self, dim, keepdim);
}
at::Tensor wrapper__argmax(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  // No device check
  // DeviceGuard omitted
#ifndef BUILD_LIBTORCH
torch_npu::profiler::NPURecordFunction guard;
#endif
  return op_plugin::argmax(self, dim, keepdim);
}


class IgnoreWarningHandler: public c10::WarningHandler {
public:
  void process(
      const c10::SourceLocation& source_location,
      const std::string& msg,
      const bool /*verbatim*/)  {
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
    if(enable) {                                                              \
      c10::Warning::set_warning_handler(getIgnoreHandler());             \
    }                                                                         \
    return 1;                                                                 \
  }                                                                           \
  static int _temp_enter_warning = enter_warning();                           \
  TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {                                \
    m.impl(TORCH_SELECTIVE_NAME("aten::is_pinned"), TORCH_FN(is_pinned));     \
    m.impl(TORCH_SELECTIVE_NAME("aten::_pin_memory"), TORCH_FN(_pin_memory)); \
    m.impl(TORCH_SELECTIVE_NAME("aten::_to_copy"), TORCH_FN(_to_copy));       \
  }                                                                           \
  TORCH_LIBRARY_IMPL(aten, CPU, m) {                                          \
    m.impl("empty.memory_format", TORCH_FN(empty_memory_format));             \
    m.impl("empty_strided", TORCH_FN(empty_strided));                         \
    m.impl("true_divide.Tensor", TORCH_FN(true_divide_Tensor));               \
    m.impl("true_divide.out", TORCH_FN(true_divide_out_Tensor));              \
    m.impl("true_divide_.Tensor", TORCH_FN(true_divide__Tensor));             \
  }                                                                           \
  TORCH_LIBRARY_IMPL(aten, CatchAll, m) {                                     \
    m.impl("_has_compatible_shallow_copy_type",                               \
        TORCH_FN(_has_compatible_shallow_copy_type));                         \
  }                                                                           \
  TORCH_LIBRARY_IMPL(aten, XLA, m) {                                          \
    m.impl("argmin", TORCH_FN(wrapper__argmin));                              \
    m.impl("argmax", TORCH_FN(wrapper__argmax));                              \
    m.impl("nan_to_num", TORCH_FN(wrapper__nan_to_num));                      \
    m.impl("nan_to_num_", TORCH_FN(wrapper__nan_to_num_));                    \
    m.impl("nan_to_num.out", TORCH_FN(wrapper_out_nan_to_num_out));           \
    m.impl("_embedding_bag_dense_backward",                                   \
        TORCH_FN(wrapper___embedding_bag_dense_backward));                    \
  }                                                                           \
  int exit_warning() {                                                        \
    if(enable) {                                                              \
      c10::Warning::set_warning_handler(nullptr);                        \
    }                                                                         \
    return 1;                                                                 \
  }                                                                           \
  static int _temp_exit_warning = exit_warning();

WITH_IGNORE_WARNING_OVERRIDE_OPERATOR(true)
}
}
