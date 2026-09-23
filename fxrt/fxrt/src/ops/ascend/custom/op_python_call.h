#ifndef __OPS_ASCEND_CUSTOM_OP_PYTHON_CALL_H__
#define __OPS_ASCEND_CUSTOM_OP_PYTHON_CALL_H__

#include <ops/op_base/op_python_call.h>

namespace fxrt {
namespace ops {
class AscendOpPythonCall : public OpPythonCall {
 public:
  AscendOpPythonCall() = default;
  ~AscendOpPythonCall() override = default;
};
} // namespace ops
} // namespace fxrt

#endif // __OPS_ASCEND_CUSTOM_OP_PYTHON_CALL_H__
