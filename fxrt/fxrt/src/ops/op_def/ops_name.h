#ifndef __OPS_OPS_NAME_H__
#define __OPS_OPS_NAME_H__

#include "common/visible.h"

namespace fxrt {
namespace ops {
#define OP(O) Op_##O,
enum Op {
#include "ops/op_def/ops.list"
  Op_End
};
#undef OP

Op MatchOp(const char* op);
FXRT_EXPORT const char* ToStr(Op op);
} // namespace ops
} // namespace fxrt

#endif // __OPS_OPS_NAME_H__
