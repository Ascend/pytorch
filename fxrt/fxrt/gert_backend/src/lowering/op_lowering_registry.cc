#include "lowering/op_lowering_registry.h"

#include <algorithm>

namespace fx_converter {

OpLoweringRegistry &OpLoweringRegistry::Instance() {
  static OpLoweringRegistry inst;
  return inst;
}

OpLoweringRegistry::OpLoweringRegistry() = default;

void OpLoweringRegistry::Register(std::unique_ptr<OpLowering> lowering, int priority) {
  if (lowering == nullptr) {
    return;
  }
  Entry entry{priority, std::move(lowering)};
  auto it = entries_.begin();
  while (it != entries_.end() && it->priority <= priority) {
    ++it;
  }
  entries_.insert(it, std::move(entry));
}

OpLowering *OpLoweringRegistry::Find(const fxrt::ir::NodePtr &node, const LoweringContext *ctx) const {
  for (const auto &e : entries_) {
    if (e.lowering != nullptr && e.lowering->Match(node, ctx)) {
      return e.lowering.get();
    }
  }
  return nullptr;
}

void OpLoweringRegistry::AppendSupportedLogicalOps(std::vector<std::string> *names) const {
  if (names == nullptr) {
    return;
  }
  for (const auto &entry : entries_) {
    if (entry.lowering == nullptr) {
      continue;
    }
    for (const auto &name : entry.lowering->SupportedLogicalOps()) {
      if (std::find(names->begin(), names->end(), name) == names->end()) {
        names->push_back(name);
      }
    }
  }
}

}  // namespace fx_converter
