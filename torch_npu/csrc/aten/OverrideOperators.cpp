#include "torch_npu/csrc/aten/OverrideOperators.h"
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

at::Tensor wrapper__argmin(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  // No device check
  // DeviceGuard omitted
#ifndef BUILD_LIBTORCH
torch_npu::profiler::NPURecordFunction guard;
#endif
  return at_npu::native::NPUNativeFunctions::argmin(self, dim, keepdim);
}
at::Tensor wrapper__argmax(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  // No device check
  // DeviceGuard omitted
#ifndef BUILD_LIBTORCH
torch_npu::profiler::NPURecordFunction guard;
#endif
  return at_npu::native::NPUNativeFunctions::argmax(self, dim, keepdim);
}


class IgnoreWarningHandler: public c10::WarningHandler {
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
    if(enable) {                                                              \
      c10::WarningUtils::set_warning_handler(getIgnoreHandler());             \
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
    m.impl("_embedding_bag_dense_backward",                                   \
        TORCH_FN(wrapper___embedding_bag_dense_backward));                    \
  }                                                                           \
  int exit_warning() {                                                        \
    if(enable) {                                                              \
      c10::WarningUtils::set_warning_handler(nullptr);                        \
    }                                                                         \
    return 1;                                                                 \
  }                                                                           \
  static int _temp_exit_warning = exit_warning();

WITH_IGNORE_WARNING_OVERRIDE_OPERATOR(true)
}
}
