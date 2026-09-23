#ifndef __OPS_ASCEND_ATB_ATB_LINEAR_H__
#define __OPS_ASCEND_ATB_ATB_LINEAR_H__

#include "ops/ascend/atb/atb_base.h"

namespace fxrt {
namespace ops {

class FXRT_EXPORT AtbLinear : public AtbBase {
 public:
  AtbLinear() : AtbBase("linear") {}
  ~AtbLinear() override = default;

  OpsErrorCode CalcWorkspace(
      const std::vector<const ir::Value*>& inputs,
      const ir::Value* output,
      size_t* workspace_size) override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& inputs,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;

 private:
  bool IsBiasNone(const std::vector<const ir::Value*>& inputs) const {
    return inputs.size() <= 2 || inputs[2] == nullptr || inputs[2]->IsNone();
  }
};

} // namespace ops
} // namespace fxrt

#endif // __OPS_ASCEND_ATB_ATB_LINEAR_H__
