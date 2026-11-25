/**
 * aten.convolution -> aclnnConvolution, replacing the generic torch fallback.
 *
 * The node reaches Convert as Op_custom_call carrying "aten.convolution" plus the
 * ordered aten argument list, so this lowering matches on that name and takes over
 * only for the parameter set the catalog row can express (unit stride/dilation,
 * no padding, transposed=false, groups=1). Anything else declines and keeps the
 * exact torch fallback, so behaviour never silently changes.
 *
 * Bit-exactness: this calls aclnnConvolution, the same kernel torch_npu runs for
 * aten.convolution, so mode3 stays identical to mode1. Rewriting a k=1 conv as a
 * matmul is NOT equivalent on Ascend (measured max abs diff 5.4e-04).
 */

#include <cstdint>
#include <string>
#include <vector>

#include "lowering/dual_stage/dual_stage_op_lowering.h"
#include "lowering/op_lowering_registry.h"
#include "ops/op_def/ops_name.h"
#include "register/op_catalog.h"

namespace fx_converter {
namespace {

// Positional aten.convolution arguments after the three tensors.
constexpr size_t kArgStride = 3U;
constexpr size_t kArgPadding = 4U;
constexpr size_t kArgDilation = 5U;
constexpr size_t kArgTransposed = 6U;
constexpr size_t kArgOutputPadding = 7U;
constexpr size_t kArgGroups = 8U;

bool ReadConstantTuple(const fxrt::ir::ValuePtr &value, std::vector<int64_t> *out) {
  if (value == nullptr || !value->IsTuple()) return false;
  const auto &tuple = value->ToTuple();
  if (tuple == nullptr) return false;
  out->clear();
  for (const auto &element : *tuple) {
    if (element == nullptr || !element->IsInt()) return false;
    out->push_back(element->ToInt());
  }
  return true;
}

bool ReadConstantInt(const fxrt::ir::ValuePtr &value, int64_t *out) {
  // aten exports transposed as a bool and groups as an int; accept either spelling.
  if (value == nullptr) return false;
  if (value->IsInt()) {
    *out = value->ToInt();
    return true;
  }
  if (value->IsBool()) {
    *out = value->ToBool() ? 1 : 0;
    return true;
  }
  return false;
}

bool AllOnes(const std::vector<int64_t> &values) {
  for (const int64_t v : values) {
    if (v != 1) return false;
  }
  return true;
}

bool AllZeros(const std::vector<int64_t> &values) {
  for (const int64_t v : values) {
    if (v != 0) return false;
  }
  return true;
}

class ConvolutionAclnnLowering final : public DualStageOpLowering {
 public:
  const char *Name() const override { return "ConvolutionAclnn"; }

  bool Match(const fxrt::ir::NodePtr &node, const LoweringContext *ctx) const override {
    (void)ctx;
    if (node == nullptr || !IsAtenConvolution(node)) return false;
    // Only the parameter set the shared attr row encodes; everything else keeps the
    // torch fallback rather than risking a silently different convolution.
    std::vector<int64_t> stride;
    std::vector<int64_t> padding;
    std::vector<int64_t> dilation;
    std::vector<int64_t> output_padding;
    int64_t transposed = 0;
    int64_t groups = 1;
    if (!ReadConstantTuple(ArgValue(node, kArgStride), &stride) ||
        !ReadConstantTuple(ArgValue(node, kArgPadding), &padding) ||
        !ReadConstantTuple(ArgValue(node, kArgDilation), &dilation) ||
        !ReadConstantTuple(ArgValue(node, kArgOutputPadding), &output_padding) ||
        !ReadConstantInt(ArgValue(node, kArgTransposed), &transposed) ||
        !ReadConstantInt(ArgValue(node, kArgGroups), &groups)) {
      return false;
    }
    if (transposed != 0 || groups != 1) return false;
    return AllOnes(stride) && AllZeros(padding) && AllOnes(dilation) && !stride.empty() &&
           stride.size() == padding.size() && stride.size() == dilation.size();
  }

 protected:
  const OpCatalogEntry *Entry(const fxrt::ir::NodePtr &node) const override {
    if (node == nullptr || !IsAtenConvolution(node)) return nullptr;
    // Opp type Conv2D: the mount point whose prepare/launch
    // InstallFxConvolutionDualStage replaces with aclnnConvolution.
    //
    // The aten positional arguments are fixed by Match: transposed=false and
    // groups=1 are already accepted there, so the Bool is declared skipped (an
    // undeclared bool fails dual-stage collect) and the rest ride the attr
    // defaults, which mirror those fixed values.
    static const OpBehavior behavior = [] {
      OpBehavior b;
      b.input_roles = kRoles;
      b.input_role_count = 1U;
      return b;
    }();
    static const GeAttr attrs[] = {
        {"strides", GeAttrType::kListInt, 1},
        {"pads", GeAttrType::kListInt, 0},
        {"dilations", GeAttrType::kListInt, 1},
        {"transposed", GeAttrType::kBool, 0},
        {"output_padding", GeAttrType::kListInt, 0},
        {"groups", GeAttrType::kInt, 1},
    };
    static const OpCatalogEntry entry = [] {
      OpCatalogEntry row;
      row.logical_name = "convolution";
      row.opp_type = "Conv2D";
      row.arity = 0U;
      row.behavior = &behavior;
      row.attrs = attrs;
      row.attr_count = sizeof(attrs) / sizeof(attrs[0]);
      return row;
    }();
    return &entry;
  }

 private:
  // The only non-tensor input for the accepted subset is the transposed Bool.
  static inline const InputRoleRule kRoles[1] = {{IrInputKind::kBool, InputRole::kSkip, nullptr, -1}};

  static bool IsAtenConvolution(const fxrt::ir::NodePtr &node) {
    if (node->op != fxrt::ops::Op_custom_call || node->inputs.empty()) return false;
    const auto &name = node->inputs[0];
    if (name == nullptr || name->output == nullptr || !name->output->IsString()) return false;
    const std::string qualified = name->output->ToString();
    return qualified.rfind("aten.convolution", 0U) == 0U;
  }

  // inputs[0] is the synthesized name node; ordered aten args follow it.
  static const fxrt::ir::ValuePtr &ArgValue(const fxrt::ir::NodePtr &node, size_t arg_index) {
    static const fxrt::ir::ValuePtr kNone;
    const size_t slot = 1U + arg_index;
    if (slot >= node->inputs.size()) return kNone;
    const auto &input = node->inputs[slot];
    if (input == nullptr) return kNone;
    return input->output;
  }
};

}  // namespace
}  // namespace fx_converter

// 900: ahead of the generic dual-stage (1000) and the torch fallback (10000).
FX_REGISTER_OP_LOWERING(900, std::make_unique<::fx_converter::ConvolutionAclnnLowering>());
