#include "ops/custom_op_register.h"
#include <mutex>
#include <algorithm>
#include <utility>
#include <vector>
#include <string>
#include <memory>

namespace fxrt {
namespace ops {

CustomOpRegistry& CustomOpRegistry::GetInstance() {
  static CustomOpRegistry instance;
  return instance;
}

bool CustomOpRegistry::RegisterCustomOp(const std::string& op_name, CustomOpFactory&& factory) {
  if (custom_ops_.find(op_name) != custom_ops_.end()) {
    return false;
  }

  custom_ops_.emplace(op_name, std::move(factory));
  return true;
}

std::unique_ptr<Operator> CustomOpRegistry::CreateCustomOp(const std::string& op_name) {
  auto it = custom_ops_.find(op_name);
  if (it == custom_ops_.end()) {
    return nullptr;
  }

  return it->second();
}

bool CustomOpRegistry::IsCustomOpRegistered(const std::string& op_name) const {
  return custom_ops_.find(op_name) != custom_ops_.end();
}

std::vector<std::string> CustomOpRegistry::GetRegisteredOpNames() const {
  std::vector<std::string> names;
  names.reserve(custom_ops_.size());

  std::transform(
      custom_ops_.begin(), custom_ops_.end(), std::back_inserter(names), [](const auto& pair) { return pair.first; });

  return names;
}

bool CustomOpRegistry::UnregisterCustomOp(const std::string& op_name) {
  auto it = custom_ops_.find(op_name);
  if (it != custom_ops_.end()) {
    custom_ops_.erase(it);
    return true;
  }

  return false;
}

std::unique_ptr<Operator> CreateCustomOperator(const std::string& name) {
  return CustomOpRegistry::GetInstance().CreateCustomOp(name);
}

} // namespace ops
} // namespace fxrt
