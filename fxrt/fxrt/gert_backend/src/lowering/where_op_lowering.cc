#include "lowering/dual_stage/dual_stage_op_lowering.h"

#include "lowering/op_lowering_registry.h"
#include "mrt_ge/mrt_node_utils.h"
#include "ops/op_def/ops_name.h"
#include "register/op_catalog.h"

namespace fx_converter {
namespace {
class WhereOpLowering final : public DualStageOpLowering {
 public:
  const char *Name() const override { return "WhereInt16Aclnn"; }

 protected:
  const OpCatalogEntry *Entry(const fxrt::ir::NodePtr &node) const override {
    // SelectV2's precompiled binaries omit int16 on the pinned CANN release.
    // Other supported dtypes continue to use the existing .o catalog row.
    if (node == nullptr || node->op != fxrt::ops::Op_where || DtypeOfNode(node) != ge::DT_INT16) {
      return nullptr;
    }
    static const OpCatalogEntry entry = [] {
      OpCatalogEntry row;
      row.logical_name = "where";
      row.opp_type = "SelectV2";
      row.arity = 3U;
      row.strided_inputs = true;
      return row;
    }();
    return &entry;
  }
};
}  // namespace
}  // namespace fx_converter

FX_REGISTER_OP_LOWERING(90, std::make_unique<::fx_converter::WhereOpLowering>());
