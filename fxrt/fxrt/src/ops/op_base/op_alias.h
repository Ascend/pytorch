#ifndef __OPS_OP_BASE_OP_ALIAS_H__
#define __OPS_OP_BASE_OP_ALIAS_H__

#include <vector>

#include "ops/operator.h"

namespace fxrt {
namespace ops {
class OpAlias : public Operator {
 public:
  OpAlias() = default;
  ~OpAlias() override = default;

  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override {
    return SUCCESS;
  }

  std::vector<std::pair<uint32_t, uint32_t>> GetOutputInputRefPairs() const override {
    return {std::pair<uint32_t, uint32_t>(0, 0)};
  }

  bool NeedLaunch() override {
    return false;
  }
};
} // namespace ops
} // namespace fxrt
#endif // __OPS_OP_BASE_OP_ALIAS_H__
