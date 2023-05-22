#include "torch_npu/csrc/aten/OverrideOperators.h"


namespace at_npu {
namespace native {
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
      c10::Warning::set_warning_handler(getIgnoreHandler());                  \
    }                                                                         \
    return 1;                                                                 \
  }                                                                           \
  static int _temp_enter_warning = enter_warning();                           \
  TORCH_LIBRARY_IMPL(aten, CPU, m) {                                          \
    m.impl("pin_memory", TORCH_FN(pin_memory));                               \
    m.impl("true_divide.Tensor", TORCH_FN(true_divide_Tensor));               \
    m.impl("true_divide.out", TORCH_FN(true_divide_out_Tensor));              \
    m.impl("true_divide_.Tensor", TORCH_FN(true_divide__Tensor));             \
  }                                                                           \
  TORCH_LIBRARY_IMPL(aten, Math, m) {                                         \
    m.impl("_has_compatible_shallow_copy_type",                               \
        TORCH_FN(_has_compatible_shallow_copy_type));                         \
  }                                                                           \
  int exit_warning() {                                                        \
    if(enable) {                                                              \
      c10::Warning::set_warning_handler(nullptr);                             \
    }                                                                         \
    return 1;                                                                 \
  }                                                                           \
  static int _temp_exit_warning = exit_warning();

WITH_IGNORE_WARNING_OVERRIDE_OPERATOR(true)

}
}
