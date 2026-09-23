#ifndef __OPS_CPU_ATEN_ATEN_ALIAS_H__
#define __OPS_CPU_ATEN_ATEN_ALIAS_H__

#include "ops/op_base/op_alias.h"

namespace fxrt {
namespace ops {
class AtenAlias : public OpAlias {
 public:
  AtenAlias() {}
  ~AtenAlias() override = default;
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_CPU_ATEN_ATEN_ALIAS_H__
