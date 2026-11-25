#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "acl/acl.h"
#include "common/logger.h"
#include "exe_graph/execute_graph_builder.h"
#include "exe_graph/graph_builder_utils.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "lowering/common/view_utils.h"
#include "lowering/common/wiring.h"
#include "lowering/op_lowering.h"
#include "lowering/op_lowering_registry.h"
#include "mrt_ge/mrt_node_utils.h"
#include "ops/op_def/ops_name.h"

namespace fx_converter {
namespace {

bool IsScalarBool(const fxrt::ir::NodePtr &node) {
  if (node == nullptr || node->output == nullptr || !node->output->IsTensor() || DtypeOfNode(node) != ge::DT_BOOL) {
    return false;
  }
  const auto tensor = node->output->ToTensor();
  if (tensor == nullptr) {
    return false;
  }
  const auto &shape = tensor->Shape();
  return shape.empty();
}

bool DependsOn(const fxrt::ir::NodePtr &node, const fxrt::ir::NodePtr &source) {
  std::vector<fxrt::ir::NodePtr> pending{node};
  std::unordered_set<const fxrt::ir::Node *> visited;
  while (!pending.empty()) {
    const auto current = pending.back();
    pending.pop_back();
    if (current == source) return true;
    if (current == nullptr || !visited.insert(current.get()).second) continue;
    pending.insert(pending.end(), current->inputs.begin(), current->inputs.end());
  }
  return false;
}

// These factories are immutable graph constants, not persistent writable state.
// Only the private zeros->fill initialization emitted by full() may be folded.
bool ValidateConstantFactory(const fxrt::ir::NodePtr &node, const LoweringContext &context, std::string *error) {
  if (context.compute_nodes == nullptr) return false;
  for (const auto &consumer : *context.compute_nodes) {
    if (consumer == nullptr || std::string(fxrt::ops::ToStr(consumer->op)).rfind("inplace_", 0) != 0 ||
        !DependsOn(consumer, node)) continue;
    bool private_fill = consumer->op == fxrt::ops::Op_inplace_fill_scalar && !consumer->inputs.empty() &&
                        consumer->inputs.front() == node && context.return_values != nullptr &&
                        std::find(context.return_values->begin(), context.return_values->end(), node) ==
                            context.return_values->end();
    for (const auto &other : *context.compute_nodes) {
      if (other == nullptr || other == consumer) continue;
      for (const auto &input : other->inputs) {
        if (input == node) private_fill = false;
      }
    }
    if (!private_fill) {
      if (error != nullptr) {
        *error = "static bool factory constant folding does not support observable mutation";
      }
      return false;
    }
  }
  return true;
}

bool ReadFillValue(const fxrt::ir::NodePtr &node, bool *value, std::string *error) {
  for (const auto &input : node->inputs) {
    if (input == nullptr || input->output == nullptr) {
      continue;
    }
    if (input->output->IsBool()) {
      *value = input->output->ToBool();
      return true;
    }
    if (input->output->IsInt()) {
      const int64_t fill_value = input->output->ToInt();
      if (fill_value == 0 || fill_value == 1) {
        *value = fill_value != 0;
        return true;
      }
    }
  }
  if (error != nullptr) {
    *error = "static bool fill lowering requires a compile-time bool value";
  }
  return false;
}

bool MaterializeScalarBool(const std::string &tag, bool value, LoweringContext &context, LoweringResult *result,
                           std::string *error) {
  uint8_t data = value ? 1U : 0U;
  void *device_data = nullptr;
  constexpr size_t kAllocationSize = sizeof(uint64_t);
  if (aclrtMalloc(&device_data, kAllocationSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || device_data == nullptr) {
    if (error != nullptr) {
      *error = "static bool factory lowering: device allocation failed";
    }
    return false;
  }
  context.built->aicore_device_bufs.emplace_back(device_data, [](void *ptr) { (void)aclrtFree(ptr); });
  if (aclrtMemcpy(device_data, kAllocationSize, &data, sizeof(data), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    if (error != nullptr) {
      *error = "static bool factory lowering: host-to-device copy failed";
    }
    return false;
  }

  gert::StorageShape storage_shape;
  gert::StorageFormat storage_format(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType());
  gert::Tensor tensor(storage_shape, storage_format, gert::kOnDeviceHbm, ge::DT_BOOL, device_data);
  auto tensor_blob = std::make_shared<std::vector<uint8_t>>(sizeof(tensor));
  std::memcpy(tensor_blob->data(), &tensor, sizeof(tensor));
  context.built->binary_bindata_blobs.push_back(tensor_blob);
  auto *constant = AddConstNode(context.main, "const_scalar_tensor_bool_" + tag, tensor_blob->data(),
                                tensor_blob->size(), false, *context.sym_gen, FxrtConstType::kGertTensorPtr);
  if (constant == nullptr) {
    if (error != nullptr) {
      *error = "static bool factory lowering: tensor constant creation failed";
    }
    return false;
  }
  context.kernel_nodes->push_back(constant);
  result->outputs.resize(1);
  TensorRef &output = result->outputs[0];
  output.producer = constant;
  output.symbol = constant->GetExtendInfo()->GetOutputSymbol(0);
  output.dtype = ge::DT_BOOL;
  output.shape.clear();
  output.storage_shape.clear();
  output.logical_shape.clear();
  uint64_t shape_symbol = 0U;
  uint64_t address_symbol = 0U;
  auto *split = GetOrCreateSplitDataTensor(context, output, tag, 0U, &shape_symbol, &address_symbol);
  if (split == nullptr) {
    if (error != nullptr) {
      *error = "static bool factory lowering: tensor unpack failed";
    }
    return false;
  }
  output.unpacked.shape = TensorEndpoint{split, 0, shape_symbol};
  output.unpacked.addr = TensorEndpoint{split, 1, address_symbol};
  result->skip_cni = true;
  return true;
}

class StaticBoolFactoryOpLowering final : public OpLowering {
 public:
  const char *Name() const override { return "static_bool_factory_const"; }

  std::vector<std::string> SupportedLogicalOps() const override { return {"zeros", "inplace_fill_scalar"}; }

  bool Match(const fxrt::ir::NodePtr &node, const LoweringContext *) const override {
    if (node == nullptr) {
      return false;
    }
    if (node->op == fxrt::ops::Op_zeros) {
      // Legacy zeros has only a size tuple. The converter factory mapping now
      // supplies an explicit value and lowers that form through native Fill.
      return node->inputs.size() == 1U && IsScalarBool(node);
    }
    return node->op == fxrt::ops::Op_inplace_fill_scalar && IsScalarBool(node) && !node->inputs.empty() &&
           node->inputs[0] != nullptr && node->inputs[0]->op == fxrt::ops::Op_zeros && IsScalarBool(node->inputs[0]);
  }

  bool Lower(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &context, LoweringResult *result,
             std::string *error) override {
    if (result == nullptr || !ValidateLoweringContext(context, false, "static bool factory lowering", error)) {
      return false;
    }
    bool value = false;
    if (node->op == fxrt::ops::Op_zeros && !ValidateConstantFactory(node, context, error)) {
      return false;
    }
    if (node->op == fxrt::ops::Op_inplace_fill_scalar &&
        (!ValidateConstantFactory(node->inputs.front(), context, error) || !ReadFillValue(node, &value, error))) {
      return false;
    }
    return MaterializeScalarBool(tag, value, context, result, error);
  }
};

}  // namespace
}  // namespace fx_converter

FX_REGISTER_OP_LOWERING(100, std::make_unique<::fx_converter::StaticBoolFactoryOpLowering>());
