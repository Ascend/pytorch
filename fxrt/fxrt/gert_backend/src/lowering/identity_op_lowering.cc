/**
 * clone / contiguous consume the storage selected by EdgeLayoutPlanner.
 * The planner enforces clone isolation around effects and explicit contiguous
 * requests. Read-only internal clones can alias; model output copy-out preserves
 * separation from caller storage. This lowerer never decides whether to copy.
 *
 * A Cast whose destination dtype equals the dtype its producer actually wrote is
 * the same kind of identity. CANN registers only real dtype conversions (there is
 * no Cast/.../9,2/9,2 binary), so sending the catalog row to the binary selector
 * fails the whole Convert before any kernel runs. Claim it here and alias the
 * source; the layout planner still materializes a dense input when the source
 * layout cannot be consumed directly, and model outputs are copied out because
 * the source allocation is never adopted.
 */

#include "lowering/op_lowering.h"
#include "lowering/common/view_utils.h"
#include "lowering/op_lowering_registry.h"
#include "mrt_ge/mrt_node_utils.h"
#include "ops/op_def/ops_name.h"

namespace fx_converter {
namespace {

// Compare the dtype the producer actually wrote (its TensorRef) with the Cast's
// declared output dtype. The FX-declared input dtype is not authoritative: a
// demoted row (custom_call) can hand back the consumer-declared dtype even when
// the front end declared a narrower one, which is exactly the int64 topk indices
// that used to reach the selector as an int64 -> int64 Cast.
bool IsIdentityCast(const fxrt::ir::NodePtr &node, const LoweringContext *ctx) {
  if (node == nullptr || node->op != fxrt::ops::Op_cast) {
    return false;
  }
  if (ctx == nullptr) {
    // Pre-dispatch support check has no tensors yet; Cast has an owner either way.
    // A real conversion falls through to the AICore row during dispatch.
    return true;
  }
  TensorRef in;
  if (!ResolveInputTensor(node, *ctx, &in, nullptr)) {
    return false;
  }
  return in.dtype == DtypeOfNode(node);
}

}  // namespace

// Before Path C (100): claim the no-op Cast so the binary selector never runs.
constexpr int kIdentityCastPriority = 95;

class IdentityCastOpLowering : public OpLowering {
 public:
  const char *Name() const override { return "IdentityCast"; }

  bool Match(const fxrt::ir::NodePtr &node, const LoweringContext *ctx) const override {
    return IsIdentityCast(node, ctx);
  }

  bool Lower(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &ctx, LoweringResult *out,
             std::string *err) override {
    (void)tag;
    if (out == nullptr || !ValidateLoweringContext(ctx, false, "identity-cast lowering", err)) {
      return false;
    }
    TensorRef in;
    if (!ResolveInputTensor(node, ctx, &in, err)) {
      return false;
    }
    if (in.dtype != DtypeOfNode(node)) {
      if (err != nullptr) {
        *err = "identity-cast lowering: input dtype does not match the cast output";
      }
      return false;
    }
    // convert_element_type returns a tensor distinct from its input; never adopt
    // the producer allocation as this node's externally visible output.
    in.can_adopt_for_model_output = false;
    out->outputs.push_back(in);
    out->launch = nullptr;
    out->prepare = nullptr;
    out->skip_cni = true;
    return true;
  }
};

// After Path C (100), before DualStage (1000).
constexpr int kIdentityAliasPriority = 200;

class IdentityAliasOpLowering : public OpLowering {
 public:
  const char *Name() const override { return "IdentityAlias"; }

  InputLayoutKind InputLayout(const fxrt::ir::NodePtr &node, size_t) const override {
    return node->op == fxrt::ops::Op_clone ? InputLayoutKind::kMetadata : InputLayoutKind::kDense;
  }

  bool Match(const fxrt::ir::NodePtr &node, const LoweringContext *) const override {
    if (node == nullptr) {
      return false;
    }
    return node->op == fxrt::ops::Op_clone || node->op == fxrt::ops::Op_contiguous;
  }

  std::vector<std::string> SupportedLogicalOps() const override { return {"clone", "contiguous"}; }

  bool Lower(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &ctx, LoweringResult *out,
             std::string *err) override {
    (void)tag;
    if (out == nullptr || !ValidateLoweringContext(ctx, false, "identity-alias lowering", err)) {
      return false;
    }
    TensorRef in;
    if (!ResolveInputTensor(node, ctx, &in, err)) {
      return false;
    }
    if (node->op == fxrt::ops::Op_clone) {
      // Never adopt the source allocation as the clone's externally visible output.
      in.can_adopt_for_model_output = false;
    }
    out->outputs.push_back(in);
    out->launch = nullptr;
    out->prepare = nullptr;
    out->skip_cni = true;
    return true;
  }
};

}  // namespace fx_converter

FX_REGISTER_OP_LOWERING(::fx_converter::kIdentityCastPriority,
                        std::make_unique<::fx_converter::IdentityCastOpLowering>());

FX_REGISTER_OP_LOWERING(::fx_converter::kIdentityAliasPriority,
                        std::make_unique<::fx_converter::IdentityAliasOpLowering>());
