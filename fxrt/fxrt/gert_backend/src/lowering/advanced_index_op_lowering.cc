#include "lowering/dual_stage/dual_stage_op_lowering.h"

#include "lowering/op_lowering_registry.h"
#include "ops/op_def/ops_name.h"
#include "register/op_catalog.h"

namespace fx_converter {
namespace {
class AdvancedIndexOpLowering final : public DualStageOpLowering {
 public:
  const char *Name() const override { return "AdvancedIndexAclnn"; }

 protected:
  const OpCatalogEntry *Entry(const fxrt::ir::NodePtr &node) const override {
    if (node == nullptr || node->op != fxrt::ops::Op_index_tensor || node->inputs.size() < 3U) {
      return nullptr;
    }
    // The Python hook admits only a contiguous leading group of integer indices.
    // Existing single-axis Gather .o lowering keeps its catalog row unchanged.
    static const OpCatalogEntry entry = [] {
      OpCatalogEntry row;
      row.logical_name = "index_tensor";
      row.opp_type = "Index";
      row.arity = 0U;
      row.strided_inputs = true;
      return row;
    }();
    return &entry;
  }
};
}  // namespace
}  // namespace fx_converter

FX_REGISTER_OP_LOWERING(90, std::make_unique<::fx_converter::AdvancedIndexOpLowering>());
