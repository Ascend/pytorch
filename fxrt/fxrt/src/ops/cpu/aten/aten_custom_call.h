#ifndef __OPS_CPU_ATEN_ATEN_CUSTOM_CALL_H__
#define __OPS_CPU_ATEN_ATEN_CUSTOM_CALL_H__

#include "ops/op_base/op_custom_call.h"

namespace fxrt {
namespace ops {
class AtenCustomCall : public OpCustomCall {
 public:
  AtenCustomCall() = default;
  ~AtenCustomCall() = default;
};
} // namespace ops
} // namespace fxrt
#endif // __OPS_CPU_ATEN_ATEN_CUSTOM_CALL_H__
