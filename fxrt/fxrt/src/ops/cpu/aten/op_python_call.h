#ifndef __OPS_CPU_ATEN_OP_PYTHON_CALL_H__
#define __OPS_CPU_ATEN_OP_PYTHON_CALL_H__

#include <ops/op_base/op_python_call.h>

namespace fxrt {
namespace ops {
class CPUOpPythonCall : public OpPythonCall {
 public:
  CPUOpPythonCall() = default;
  ~CPUOpPythonCall() override = default;
};
} // namespace ops
} // namespace fxrt

#endif // __OPS_CPU_ATEN_OP_PYTHON_CALL_H__
