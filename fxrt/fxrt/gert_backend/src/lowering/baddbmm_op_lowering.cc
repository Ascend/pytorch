/**
 * aten.baddbmm -> aclnnBaddbmm, replacing the generic torch fallback.
 *
 * Bit-exactness: baddbmm is NOT bmm + add on Ascend. The fused kernel folds beta
 * into the cube accumulator, so a separate broadcast add rounds differently
 * (measured max abs diff 2.7e-05). Only aclnnBaddbmm -- the kernel torch_npu runs
 * for aten.baddbmm -- keeps mode3 identical to mode1.
 *
 * Out= is batch1@batch2, NOT any input: `self` broadcasts. aclnn validates the
 * output against the matmul result, so passing `self`'s shape would be rejected.
 *
 * Mount point: OPP registers no "Baddbmm" OpImpl entry. "BatchMatMulV2" is a
 * distinct entry from the "BatchMatMul" the plain bmm row already patches, so this
 * reuses it without disturbing that wrapper.
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

// Positional aten.baddbmm arguments after the three tensors: beta, alpha.
constexpr size_t kArgBeta = 3U;
constexpr size_t kArgAlpha = 4U;

bool ReadConstantDouble(const fxrt::ir::ValuePtr &value, double *out) {
  // beta / alpha reach the IR as floats; accept an integral spelling too.
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

class BaddbmmAclnnLowering final : public DualStageOpLowering {
 public:
  const char *Name() const override { return "BaddbmmAclnn"; }

  bool Match(const fxrt::ir::NodePtr &node, const LoweringContext *ctx) const override {
    (void)ctx;
    if (node == nullptr || !IsAtenBaddbmm(node)) return false;
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
      if (err != nullptr) *err = "baddbmm: expected constant beta and alpha";
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
    if (node == nullptr || !IsAtenBaddbmm(node)) return nullptr;
    // beta/alpha are the only attrs, so they land at RuntimeAttrs index 0 and 1.
    static const GeAttr attrs[] = {
        {"beta", GeAttrType::kFloat, 1},
        {"alpha", GeAttrType::kFloat, 1},
    };
    static const OpCatalogEntry entry = [] {
      OpCatalogEntry row;
      row.logical_name = "baddbmm";
      row.opp_type = "BatchMatMulV2";
      row.arity = 3U;
      // self may be an expanded view and batch2 an as_strided view; both need the
      // runtime TensorV2 stride, not a densified copy.
      row.strided_inputs = true;
      row.attrs = attrs;
      row.attr_count = sizeof(attrs) / sizeof(attrs[0]);
      return row;
    }();
    return &entry;
  }

 private:
  static bool IsAtenBaddbmm(const fxrt::ir::NodePtr &node) {
    if (node->op != fxrt::ops::Op_custom_call || node->inputs.empty()) return false;
    const auto &name = node->inputs[0];
    if (name == nullptr || name->output == nullptr || !name->output->IsString()) return false;
    const std::string qualified = name->output->ToString();
    return qualified.rfind("aten.baddbmm", 0U) == 0U;
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
FX_REGISTER_OP_LOWERING(900, std::make_unique<::fx_converter::BaddbmmAclnnLowering>());
