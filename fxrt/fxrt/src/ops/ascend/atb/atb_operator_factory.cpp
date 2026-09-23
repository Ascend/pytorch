#include "ops/ascend/atb/atb_operator_factory.h"
#include "ops/ascend/atb/atb_linear.h"

extern "C" {

FXRT_EXPORT void* CreateAtbLinear() {
  return new fxrt::ops::AtbLinear();
}
}
