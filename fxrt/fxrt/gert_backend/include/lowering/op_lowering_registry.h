#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lowering/op_lowering.h"

namespace fx_converter {

// Priority: lower value is matched first. Use < kDefaultDualStagePriority for specific lowerings.
constexpr int kDefaultDualStagePriority = 1000;

class OpLoweringRegistry {
 public:
  static OpLoweringRegistry& Instance();

  // Inserts by ascending priority (stable among equal priorities by registration order).
  void Register(std::unique_ptr<OpLowering> lowering, int priority = kDefaultDualStagePriority);
  // ctx is forwarded to Match; pass nullptr (or omit) for the pre-dispatch support check.
  OpLowering* Find(const fxrt::ir::NodePtr& node, const LoweringContext* ctx = nullptr) const;
  // Union of every registered lowering's SupportedLogicalOps(), deduplicated.
  void AppendSupportedLogicalOps(std::vector<std::string>* names) const;

 private:
  OpLoweringRegistry();
  struct Entry {
    int priority = 0;
    std::unique_ptr<OpLowering> lowering;
  };
  std::vector<Entry> entries_;
};

// Static-init helper used by FX_REGISTER_OP_LOWERING.
struct OpLoweringRegistrar {
  OpLoweringRegistrar(int priority, std::unique_ptr<OpLowering> lowering) {
    OpLoweringRegistry::Instance().Register(std::move(lowering), priority);
  }
};

} // namespace fx_converter

// Register an OpLowering at shared-library load time. Place in the .cc that owns the lowering.
// Example: FX_REGISTER_OP_LOWERING(100, std::make_unique<XxxLowering>());
#define FX_CONCAT_INNER(a, b) a##b
#define FX_CONCAT(a, b) FX_CONCAT_INNER(a, b)
#define FX_REGISTER_OP_LOWERING(priority, factory_expr)                                                             \
  namespace {                                                                                                       \
  static ::fx_converter::OpLoweringRegistrar FX_CONCAT(_fx_op_lowering_reg_, __LINE__)((priority), (factory_expr)); \
  }
