#include "shape/mrt_to_ge_expr.h"

#include <algorithm>
#include <cctype>
#include <memory>

#include "common/logger.h"
#include "exe_graph/execute_graph_builder.h"
#include "ir/value/value.h"
#include "mrt_ge/mrt_node_utils.h"
#include "shape/fx_sym_expr_parser.h"

namespace fx_converter {
namespace {

bool IsSimpleIdent(const std::string &s) {
  if (s.empty() || (!std::isalpha(static_cast<unsigned char>(s[0])) && s[0] != '_')) {
    return false;
  }
  for (char c : s) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
      return false;
    }
  }
  return true;
}

fxrt::ir::TensorPtr TensorOf(const fxrt::ir::NodePtr &node) {
  if (node == nullptr || node->output == nullptr || !node->output->IsTensor()) {
    return nullptr;
  }
  return node->output->ToTensor();
}

}  // namespace

void InitSymbolEnvFromGraphInputs(BuiltExecuteGraph *built, const fxrt::ir::Graph &graph) {
  built->symbol_env = std::make_shared<FxSymbolEnv>();
  CollectGraphInputSeeds(graph, *built->symbol_env);
  if (!built->symbol_env->seeds.empty()) {
    RT_GLOG(INFO) << "SymbolEnv: " << built->symbol_env->seeds.size()
                  << " free-var seed(s) from graph inputs (shared Evaluate)";
  } else {
    // Fallback when BindSymbolEnv cannot attach (e.g. no graph feeds): mark bound so
    // InferShape rules with only constant Expressions (SymShapeForExport numeric dims) work.
    built->symbol_env->bound = true;
  }
}

void CollectGraphInputSeeds(const fxrt::ir::Graph &graph, FxSymbolEnv &env) {
  // feed_index must match ModelV2 Data order (filtered model feeds), not raw graph.inputs.
  int32_t feed_idx = 0;
  for (size_t i = 0; i < graph.inputs.size(); ++i) {
    if (!IsModelFeedTensor(graph, graph.inputs[i])) {
      continue;
    }
    // Keep feed indexing aligned with AssignPlaceholderNames / WirePlaceholderFeeds.
    const auto tensor = TensorOf(graph.inputs[i]);
    if (!tensor->HasSymbolicShape()) {
      ++feed_idx;
      continue;
    }
    const auto &syms = tensor->GetSymbolicShape();
    for (size_t d = 0; d < syms.size(); ++d) {
      const auto &e = syms[d];
      if (e == nullptr || e->GetKind() != fxrt::ir::SymbolicExpr::Kind::Variable) {
        continue;
      }
      const auto *var = static_cast<const fxrt::ir::SymbolicVar *>(e.get());
      const std::string &name = var->GetName();
      if (!IsSimpleIdent(name)) {
        continue;
      }
      bool exists = std::any_of(env.seeds.begin(), env.seeds.end(), [&](const auto &s) { return s.name == name; });
      if (!exists) {
        env.seeds.push_back(FxSymbolEnv::Seed{name, feed_idx, static_cast<int32_t>(d)});
      }
    }
    ++feed_idx;
  }
}

std::shared_ptr<FxShapeRule> BuildOutputSymbolicShapeRule(const fxrt::ir::NodePtr &out_node, FxSymbolEnv *env,
                                                          const char *source_tag) {
  return BuildOutputSymbolicShapeRuleFromTensor(TensorOf(out_node), env, source_tag);
}

std::shared_ptr<FxShapeRule> BuildOutputSymbolicShapeRuleFromTensor(const fxrt::ir::TensorPtr &out_t,
                                                                    FxSymbolEnv *env, const char *source_tag) {
  // Allow empty seeds when out dims are concrete SymInt/constants (static graphs).
  if (env == nullptr) {
    return nullptr;
  }
  if (out_t == nullptr) {
    return nullptr;
  }
  auto rule = std::make_shared<FxShapeRule>();
  rule->env = env;
  rule->source = (source_tag != nullptr && source_tag[0] != '\0') ? std::string("shared_env:") + source_tag
                                                                  : std::string("shared_env");
  if (!out_t->HasSymbolicShape()) {
    if (out_t->HasDynamicShape()) {
      // Dynamic dims with no symbols to solve against: keep the loud no-rule error.
      return nullptr;
    }
    // A fully concrete or 0-d output (e.g. ArgMaxV2 keepdim on a 1-d input yields [1])
    // has no symbol for Execute to solve, but the InferShape wiring still expects one
    // rule entry per output dim. Emit constant expressions from the static shape; a
    // 0-d output yields an empty rule, which is a valid no-op at Execute.
    rule->is_rank0 = out_t->Shape().empty();
    for (const int64_t dim : out_t->Shape()) {
      rule->out_dims_ir.push_back(fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicConst>(dim));
    }
    return rule;
  }
  for (const auto &dim_expr : out_t->GetSymbolicShape()) {
    // The ABI bridge hands every dim over as one ToString() string wrapped in a single
    // SymbolicVar. ParseFxSymExpr recovers the tree (floor_div / ceil_div / min included).
    // A dim that does not parse cannot be evaluated or emitted into OM2, so Convert fails.
    const auto ir_expr = ParseFxSymExpr(dim_expr, *env);
    if (ir_expr == nullptr) {
      RT_GLOG(ERROR) << "Symbol shape: failed to parse dim, source=" << rule->source;
      return nullptr;
    }
    rule->out_dims_ir.push_back(ir_expr);
  }
  return rule;
}

}  // namespace fx_converter
