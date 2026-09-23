#ifndef __OPS_ASCEND_ATB_ATB_ADD_H__
#define __OPS_ASCEND_ATB_ATB_ADD_H__

#include <vector>
#include "ops/ascend/atb/atb_base.h"

namespace fxrt {
namespace ops {

class AtbAdd : public AtbBase {
 public:
  AtbAdd();
  ~AtbAdd() override = default;

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
};

} // namespace ops
} // namespace fxrt

#endif
