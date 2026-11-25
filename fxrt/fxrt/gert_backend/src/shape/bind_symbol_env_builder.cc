#include "shape/bind_symbol_env_builder.h"

#include <algorithm>
#include <iterator>

#include "common/logger.h"
#include "mrt_ge/mrt_node_utils.h"

namespace fx_converter {

fx_converter::Node *AttachBindSymbolEnv(ge::Graph *main, FxSymbolEnv *env,
                                  const std::vector<fx_converter::Node *> &graph_input_datas,
                                  const std::vector<fxrt::ir::NodePtr> &graph_inputs, uint64_t &sym_gen,
                                  std::vector<fx_converter::Node *> *kernel_nodes) {
  // Empty seeds are OK: static graphs (concrete as_strided / empty_strided sizes) still
  // need BindSymbolEnv so env->bound is set before FxExprInferFromRule (constant Expression dims).
  if (env == nullptr || graph_input_datas.empty()) {
    return nullptr;
  }
  if (graph_input_datas.size() != graph_inputs.size()) {
    RT_GLOG(ERROR) << "AttachBindSymbolEnv: graph_input_datas.size()=" << graph_input_datas.size()
                   << " != graph_inputs.size()=" << graph_inputs.size();
    return nullptr;
  }
  const ge::TensorDesc empty_td;
  FxSymbolEnv *env_raw = env;
  fx_converter::Node *c_env = AddConstNode(main, "const_symbol_env", &env_raw, sizeof(env_raw), false, sym_gen,
                                           FxrtConstType::kVoidPtr);
  auto bind_desc = std::make_shared<fx_converter::FxNodeSpec>("bind_symbol_env", "BindSymbolEnv");
  bind_desc->AddInputDesc(empty_td);
  for (size_t i = 0; i < graph_input_datas.size(); ++i) {
    const auto shape = ShapeOfNode(graph_inputs[i]);
    const ge::DataType dt = DtypeOfNode(graph_inputs[i]);
    bind_desc->AddInputDesc(MakeNdTensorDesc(shape, dt));
  }
  fx_converter::Node *bind_symbols = fx_converter::AddNode(main, bind_desc);
  const uint64_t s_env = c_env->GetExtendInfo()->GetOutputSymbol(0);
  std::vector<uint64_t> bind_in_syms = {s_env};
  std::transform(graph_input_datas.begin(), graph_input_datas.end(), std::back_inserter(bind_in_syms),
                 [](fx_converter::Node *d) { return d->GetExtendInfo()->GetOutputSymbol(0); });
  WireSymbols(bind_symbols, bind_in_syms, {});
  fx_converter::AddDataEdge(main, c_env, 0, bind_symbols, 0);
  for (size_t i = 0; i < graph_input_datas.size(); ++i) {
    fx_converter::AddDataEdge(main, graph_input_datas[i], 0, bind_symbols, static_cast<int32_t>(i + 1));
  }
  if (kernel_nodes != nullptr) {
    kernel_nodes->push_back(c_env);
    kernel_nodes->push_back(bind_symbols);
  }
  return bind_symbols;
}

}  // namespace fx_converter
