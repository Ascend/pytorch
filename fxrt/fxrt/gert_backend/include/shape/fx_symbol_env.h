#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "ir/symbolic/symbolic.h"

namespace fx_converter {

// Shared symbol environment for one ExecuteGraph / one Execute.
// Free vars are assigned once from graph-input shapes, then every InferShape
// evaluates fxrt::ir::SymbolicExpr against this env.
struct FxSymbolEnv {
  struct Seed {
    std::string name;
    int32_t feed_index = 0; // Model input index
    int32_t dim_index = 0;
  };

  // Canonical fxrt::ir::SymbolicVar instances. SymbolicVar holds its own value, so interning
  // by name is what makes one SetValue reach every rule tree.
  std::unordered_map<std::string, fxrt::ir::SymbolicExprPtr> ir_vars;
  // Runtime values for the current Execute.
  std::unordered_map<std::string, int64_t> values;
  // How to fill values from user feed tensors at Execute entry.
  std::vector<Seed> seeds;
  bool bound = false;

  void ClearValues() {
    values.clear();
    bound = false;
  }

  fxrt::ir::SymbolicExprPtr& EnsureIrVar(const std::string& name) {
    auto it = ir_vars.find(name);
    if (it != ir_vars.end()) {
      return it->second;
    }
    auto inserted = ir_vars.emplace(name, fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicVar>(name));
    return inserted.first->second;
  }

  void SetValue(const std::string& name, int64_t v) {
    values[name] = v;
    auto& ir_var = EnsureIrVar(name);
    static_cast<fxrt::ir::SymbolicVar*>(ir_var.get())->SetValue(v);
  }
};

} // namespace fx_converter
