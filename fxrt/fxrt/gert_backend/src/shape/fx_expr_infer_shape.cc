#include "shape/fx_expr_infer_shape.h"

#include <cstdint>
#include <sstream>
#include <string_view>
#include <vector>

#include "common/logger.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/storage_shape.h"
#include "shape/fx_rule_eval.h"
#include "shape/fx_shape_rule.h"

namespace {

// Signature must match gert::OpImplRegisterV2::InferShapeKernelFunc (= UINT32 (*)(...)).
UINT32 FxExprInferFromRule(gert::InferShapeContext *ctx) {
  if (ctx == nullptr) {
    return ge::GRAPH_FAILED;
  }
  // Layout: [in0..inN-1, rule0..ruleM-1, infer_func] with M == output slot count. Each
  // output slot gets its own rule: multi-output slots may carry distinct shapes
  // (LayerNormV4 mean/rstd drop the trailing norm dims that y keeps), so mirroring one
  // rule across slots is wrong in general.
  auto *kctx = reinterpret_cast<gert::KernelContext *>(ctx);
  const size_t input_num = kctx->GetInputNum();
  const size_t out_num = kctx->GetOutputNum();
  if (out_num == 0U || input_num < out_num + 1U) {
    RT_GLOG(ERROR) << "FxExprInferFromRule: expected M rules + func for " << out_num << " outputs, got "
                   << input_num << " inputs";
    return ge::GRAPH_FAILED;
  }
  const size_t rule_base = input_num - 1U - out_num;
  // GetOutputShape writes origin only. CalcTensorSizeFromStorage / TensorToOut read storage;
  // ND has no format expand, so storage must match origin even if TransformAllOutputsShape
  // is skipped (CNI outs < InferShape outs).
  for (size_t oi = 0; oi < out_num; ++oi) {
    const FxShapeRule *rule = kctx->GetInputValue<const FxShapeRule *>(rule_base + oi);
    if (!fx_converter::RuleIsEvaluable(rule)) {
      RT_GLOG(ERROR) << "FxExprInferFromRule: unusable rule at input " << (rule_base + oi);
      return ge::GRAPH_FAILED;
    }
    std::vector<int64_t> dims;
    if (!fx_converter::EvalRuleDims(*rule, dims)) {
      return ge::GRAPH_FAILED;
    }
    gert::Shape *y = ctx->GetOutputShape(oi);
    if (y == nullptr) {
      return ge::GRAPH_FAILED;
    }
    y->SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      y->SetDim(i, dims[i]);
    }
    auto *ss = kctx->GetOutputPointer<gert::StorageShape>(oi);
    if (ss == nullptr) {
      return ge::GRAPH_FAILED;
    }
    ss->MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      ss->MutableStorageShape().SetDim(i, dims[i]);
    }
    const char *node_name = ctx->GetNodeName();
    if (node_name != nullptr && std::string_view(node_name).find("op158") != std::string_view::npos) {
      std::ostringstream shape;
      shape << "[";
      for (size_t i = 0; i < dims.size(); ++i) {
        shape << (i == 0 ? "" : ",") << dims[i];
      }
      shape << "]";
      RT_GLOG(INFO) << "[ES_SHAPE_DIAG] InferShape node=" << node_name << " output=" << oi
                    << " rule=" << rule->source << " origin=" << shape.str() << " storage=" << shape.str();
    }
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

gert::OpImplRegisterV2::InferShapeKernelFunc GetFxExprInferShapeFunc() { return &FxExprInferFromRule; }
