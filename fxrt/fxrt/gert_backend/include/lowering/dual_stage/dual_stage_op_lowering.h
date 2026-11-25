#pragma once

#include "lowering/op_lowering.h"

namespace fx_converter {

struct CatalogAttrExtras;
struct DualStageOpFwk;
struct OpCatalogEntry;

// True when the OPP type has both execute-prepare and execute-launch installed.
bool IsOppDualStageReady(const char* opp_type);

// Catalog rows with more than one output. Single-output ops stay on LowerNaryDualStage.
bool LowerMultiOutputDualStage(
    const fxrt::ir::NodePtr& node,
    const std::string& tag,
    LoweringContext& ctx,
    const OpCatalogEntry& entry,
    const DualStageOpFwk& fwk,
    std::vector<TensorRef>* ins,
    const CatalogAttrExtras& extras,
    std::string* err,
    LoweringResult* out);

// Shared OPP dual-stage lowering (GE LoweringAclnnNode analogue). Catalog rows
// select normal ops; specialized ABI/dtype dispatch overrides only Entry().
class DualStageOpLowering : public OpLowering {
 public:
  const char* Name() const override;
  bool Match(const fxrt::ir::NodePtr& node, const LoweringContext* ctx) const override;
  InputLayoutKind InputLayout(const fxrt::ir::NodePtr& node, size_t index) const override;
  bool Lower(
      const fxrt::ir::NodePtr& node,
      const std::string& tag,
      LoweringContext& ctx,
      LoweringResult* out,
      std::string* err) override;

 protected:
  // Specialized native ABIs reuse the standard prepare/workspace/launch wiring.
  virtual const OpCatalogEntry* Entry(const fxrt::ir::NodePtr& node) const;

 private:
  // Cache DualStageOpFwk per OPP type within one Convert (one BuildFwk per type).
  bool EnsureFwk(LoweringContext& ctx, const char* opp_type, const DualStageOpFwk** fwk_out, std::string* err);
};

} // namespace fx_converter
