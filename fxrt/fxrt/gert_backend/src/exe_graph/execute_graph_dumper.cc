#include "exe_graph/execute_graph_dumper.h"

#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fx_converter {
namespace {

std::string ExprToStr(const fxrt::ir::SymbolicExprPtr &expr) {
  return (expr != nullptr) ? expr->ToString() : std::string("<null>");
}

void DumpShapeRule(std::ostringstream &oss, size_t idx, const FxShapeRule &rule) {
  oss << "  rule[" << idx << "] source=" << rule.source << " env=" << (rule.env != nullptr ? "shared" : "null")
      << "\n";
  oss << "    out_dims: [";
  for (size_t i = 0; i < rule.out_dims_ir.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << ExprToStr(rule.out_dims_ir[i]);
  }
  oss << "]\n";
  if (rule.env != nullptr) {
    oss << "    env: shared SymbolEnv seeds=" << rule.env->seeds.size()
        << " bound=" << (rule.env->bound ? "true" : "false") << "\n";
  } else {
    oss << "    env: <null>\n";
  }
}

void DumpSubGraph(std::ostringstream &oss, const char *label, const fx_converter::EsGraphRef &g) {
  if (g == nullptr) {
    oss << "  " << label << ": <null>\n";
    return;
  }
  oss << "  " << label << ": name=" << g->GetName() << " nodes=" << GetAllNodes(g.get()).size() << "\n";
}

void DumpKernelTypes(std::ostringstream &oss, const char *label, const std::vector<fx_converter::Node *> &nodes) {
  std::map<std::string, size_t> type_hist;
  for (fx_converter::Node *node : nodes) {
    if (node != nullptr) {
      ++type_hist[std::string(node->GetType())];
    }
  }
  oss << "=== ExecuteGraph " << label << " RT Kernel types ===\n";
  for (const auto &kv : type_hist) {
    oss << "  " << kv.first << " : " << kv.second << "\n";
  }
  static const char *kTilingKeys[] = {
      "FxAiCoreTiling",
      "InductorAutofuseTiling",
  };
  oss << "=== ExecuteGraph " << label << " tiling summary ===\n";
  for (const char *key : kTilingKeys) {
    const auto it = type_hist.find(key);
    oss << "  " << key << " : " << (it == type_hist.end() ? 0 : it->second) << "\n";
  }
}

void DumpMainNodeList(std::ostringstream &oss, const std::vector<fx_converter::Node *> &nodes) {
  oss << "=== ExecuteGraph main nodes (idx | name : type) ===\n";
  for (size_t i = 0; i < nodes.size(); ++i) {
    fx_converter::Node *node = nodes[i];
    if (node == nullptr) {
      continue;
    }
    oss << "  [" << i << "] " << node->GetName() << " : " << node->GetType() << "\n";
  }
}

void DumpMainEdges(std::ostringstream &oss, ge::Graph *graph, const std::vector<fx_converter::Node *> &nodes) {
  oss << "=== ExecuteGraph main data edges (src[out] -> dst[in]) ===\n";
  for (fx_converter::Node *node : nodes) {
    if (node == nullptr) {
      continue;
    }
    for (const auto &edge : GetInDataEdges(graph, node)) {
      if (edge.src == nullptr) {
        continue;
      }
      oss << "  " << edge.src->GetName() << "[" << edge.src_port << "] -> " << node->GetName() << "["
          << edge.dst_port << "]\n";
    }
  }
  oss << "=== ExecuteGraph main control edges (src -> dst) ===\n";
  for (fx_converter::Node *node : nodes) {
    if (node == nullptr) {
      continue;
    }
    for (fx_converter::Node *src : GetInControlNodes(graph, node)) {
      if (src != nullptr) {
        oss << "  " << src->GetName() << " -> " << node->GetName() << "\n";
      }
    }
  }
}

void DumpInferShapeSection(std::ostringstream &oss, const BuiltExecuteGraph &built,
                           const std::vector<fx_converter::Node *> &nodes) {
  oss << "=== InferShape form on graph ===\n";
  oss << "  Patterns:\n";
  oss << "    InferShape(tensors..., Const<FxShapeRule*>, Const<infer_func*>) + BindSymbolEnv\n";
  oss << "    class2: InferShape(x, shape_tensor, Const<infer_func*>) — GetInputTensor(value)\n";
  oss << "    class3: InferShape(x, Const<infer_func*>) emits max; Calc/Alloc by max; Launch refreshes\n";
  if (built.symbol_env != nullptr && !built.symbol_env->seeds.empty()) {
    oss << "  Symbol mode: SHARED env (BindSymbolEnv RT Kernel + InferShape Evaluate)\n";
    oss << "  SymbolEnv seeds:\n";
    for (const auto &s : built.symbol_env->seeds) {
      oss << "    " << s.name << " <- feed[" << s.feed_index << "].dim[" << s.dim_index << "]\n";
    }
  } else {
    oss << "  Symbol mode: <missing seeds — Convert should have failed>\n";
  }
  oss << "  Note: free vars assigned by BindSymbolEnv from Data tensor shapes; Infer evaluates fxrt SymbolicExpr.\n";
  for (size_t ni = 0; ni < nodes.size(); ++ni) {
    fx_converter::Node *node = nodes[ni];
    if (node == nullptr || std::string(node->GetType()) != "InferShape") {
      continue;
    }
    oss << "  [" << ni << "] " << node->GetName() << " : InferShape\n";
    for (size_t i = 0; i < node->InputCount(); ++i) {
      fx_converter::Node *in = node->InputProducer(static_cast<int32_t>(i));
      if (in == nullptr) {
        oss << "    in[" << i << "] <- <null>\n";
      } else {
        oss << "    in[" << i << "] <- " << in->GetName() << " (" << in->GetType() << ")\n";
      }
    }
  }
}

void DumpKeyControlNodes(std::ostringstream &oss, const std::vector<fx_converter::Node *> &nodes) {
  oss << "=== Key control / locator nodes ===\n";
  for (size_t ni = 0; ni < nodes.size(); ++ni) {
    fx_converter::Node *node = nodes[ni];
    if (node == nullptr) {
      continue;
    }
    const std::string ty = node->GetType();
    if (ty == "BindSymbolEnv" || ty == "ExecuteOpPrepare" || ty == "ExecuteOpLaunch" || ty == "NetOutput") {
      oss << "  [" << ni << "] " << node->GetName() << " : " << ty << "\n";
    }
  }
}

void DumpShapeRulesSection(std::ostringstream &oss, const BuiltExecuteGraph &built) {
  oss << "=== FxShapeRule ===\n";
  oss << "  count=" << built.shape_rules.size() << "\n";
  for (size_t i = 0; i < built.shape_rules.size(); ++i) {
    if (built.shape_rules[i] != nullptr) {
      DumpShapeRule(oss, i, *built.shape_rules[i]);
    } else {
      oss << "  rule[" << i << "] <null>\n";
    }
  }
}

}  // namespace

std::string DumpExecuteGraph(const BuiltExecuteGraph &built) {
  std::ostringstream oss;
  if (!built.error.empty()) {
    oss << "=== FxConverter ExecuteGraph dump FAILED ===\n";
    oss << built.error << "\n";
    return oss.str();
  }
  oss << "=== FxConverter ExecuteGraph dump ===\n";
  oss << "=== Subgraphs ===\n";
  DumpSubGraph(oss, "root", built.root);
  DumpSubGraph(oss, "init", built.init);
  DumpSubGraph(oss, "main", built.main);
  DumpSubGraph(oss, "deinit", built.deinit);
  if (built.main == nullptr) {
    oss << "=== ExecuteGraph dump (main=null) ===\n";
    return oss.str();
  }
  if (built.init != nullptr) {
    DumpKernelTypes(oss, "init", GetAllNodes(built.init.get()));
  }
  const std::vector<fx_converter::Node *> nodes = GetAllNodes(built.main.get());
  DumpKernelTypes(oss, "main", nodes);
  DumpMainNodeList(oss, nodes);
  DumpMainEdges(oss, built.main.get(), nodes);
  DumpInferShapeSection(oss, built, nodes);
  DumpKeyControlNodes(oss, nodes);
  DumpShapeRulesSection(oss, built);
  return oss.str();
}

}  // namespace fx_converter
