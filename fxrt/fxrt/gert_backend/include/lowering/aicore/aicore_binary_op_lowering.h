#pragma once

#include "lowering/op_lowering.h"

namespace fx_converter {

struct OpCatalogEntry;

// The layout planner selects an OPP implementation after proving its geometry.
// The entry must outlive Convert: CNI assembly retains its schema pointer.
bool LowerAiCoreBinaryWithCatalog(
    const fxrt::ir::NodePtr& node,
    const OpCatalogEntry& entry,
    const std::string& tag,
    LoweringContext& ctx,
    LoweringResult* out,
    std::string* err);

// AICore binary-reuse lowering (path C). Convert gates on CANN prebuilt .o via
// simplifiedKey, then wires GE-isomorphic RT nodes:
// AllocLaunchArg → TilingParse/Tiling → SinkNodeBinWithHandle → LaunchKernelWithHandle.
// Not DualStage / aclnn.
class AiCoreBinaryOpLowering final : public OpLowering {
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
};

// Match before DualStage (kDefaultDualStagePriority = 1000).
constexpr int kAiCoreBinaryPriority = 100;

} // namespace fx_converter
