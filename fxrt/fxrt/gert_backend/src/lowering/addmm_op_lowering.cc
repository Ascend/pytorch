/**
 * aten.addmm -> aclnnAddmm, replacing the generic torch fallback.
 *
 * beta and alpha follow the three tensor arguments in ATen schema order. Read
 * them by position, skipping the custom_call name, and publish named CNI attrs.
 * Kind-based scalar ordinals also count that name string and differ for integer
 * spellings, so they cannot identify these arguments reliably.
 *
 * Mount point is MatMulV2, which OPP registers with a native
 * OPTIONAL_INPUT(bias).
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

// Positional aten.addmm arguments after the three tensors: beta, alpha.
constexpr size_t kArgBeta = 3U;
constexpr size_t kArgAlpha = 4U;

bool ReadConstantDouble(const fxrt::ir::ValuePtr &value, double *out) {
  if (value == nullptr || out == nullptr) return false;
  if (value->IsDouble()) {
    *out = value->ToDouble();
    return true;
  }
  if (value->IsInt()) {
    *out = static_cast<double>(value->ToInt());
    return true;
  }
  return false;
}

class AddmmAclnnLowering final : public DualStageOpLowering {
 public:
  const char *Name() const override { return "AddmmAclnn"; }

  bool Match(const fxrt::ir::NodePtr &node, const LoweringContext *ctx) const override {
    (void)ctx;
    if (node == nullptr || !IsAtenAddmm(node)) return false;
    // Both scalars must be compile-time constants: they ride the CNI attr channel.
    double beta = 0.0;
    double alpha = 0.0;
    return ReadConstantDouble(ArgValue(node, kArgBeta), &beta) &&
           ReadConstantDouble(ArgValue(node, kArgAlpha), &alpha) && node->inputs.size() >= 1U + kArgAlpha + 1U;
  }

  bool Lower(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &ctx,
             LoweringResult *out, std::string *err) override {
    double beta = 0.0;
    double alpha = 0.0;
    if (!ReadConstantDouble(ArgValue(node, kArgBeta), &beta) ||
        !ReadConstantDouble(ArgValue(node, kArgAlpha), &alpha)) {
      if (err != nullptr) *err = "addmm: expected constant beta and alpha";
      return false;
    }
    if (!DualStageOpLowering::Lower(node, tag, ctx, out, err)) return false;
    // ATen schema positions exclude the synthesized custom_call name. Preserve
    // both integral and floating spellings without relying on kind ordinals.
    out->cni.attrs.SetDouble("beta", beta);
    out->cni.attrs.SetDouble("alpha", alpha);
    return true;
  }

 protected:
  const OpCatalogEntry *Entry(const fxrt::ir::NodePtr &node) const override {
    if (node == nullptr || !IsAtenAddmm(node)) return nullptr;
    // RuntimeAttrs slots 0/1 must remain beta/alpha. Preserve transposed layouts
    // through TensorV2 instead of inserting transpose bools ahead of these attrs.
    static const GeAttr attrs[] = {
        {"beta", GeAttrType::kFloat, 1},
        {"alpha", GeAttrType::kFloat, 1},
    };
    static const OpCatalogEntry entry = [] {
      OpCatalogEntry row;
      row.logical_name = "addmm";
      // MatMulV2 is a real OPP type with a native OPTIONAL_INPUT(bias). Do not
      // mint an entry named "Addmm": OPP registers no such op proto, so the entry
      // has no shape inference and OM2 never binds the launch output.
      row.opp_type = "MatMulV2";
      row.arity = 3U;
      // AddmmPrepare consumes TensorV2 layouts. Preserve transposed weights
      // without emitting transpose attrs into the beta/alpha attr group.
      row.strided_inputs = true;
      // OPP input order; prepare reads tensor 0=mat1, 1=mat2, 2=bias.
      row.opp_input_names = "mat1,mat2,bias";
      row.attrs = attrs;
      row.attr_count = sizeof(attrs) / sizeof(attrs[0]);
      return row;
    }();
    return &entry;
  }

 private:
  static bool IsAtenAddmm(const fxrt::ir::NodePtr &node) {
    if (node->op != fxrt::ops::Op_custom_call || node->inputs.empty()) return false;
    const auto &name = node->inputs[0];
    if (name == nullptr || name->output == nullptr || !name->output->IsString()) return false;
    const std::string qualified = name->output->ToString();
    return qualified.rfind("aten.addmm", 0U) == 0U;
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
FX_REGISTER_OP_LOWERING(900, std::make_unique<::fx_converter::AddmmAclnnLowering>());
