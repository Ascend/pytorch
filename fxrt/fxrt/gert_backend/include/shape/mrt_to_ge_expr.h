#pragma once

#include <memory>

#include "shape/fx_shape_rule.h"
#include "shape/fx_symbol_env.h"
#include "ir/graph.h"
#include "ir/tensor/tensor.h"

struct BuiltExecuteGraph;

namespace fx_converter {

// Create SymbolEnv, collect free-var seeds from graph.inputs, assign to built->symbol_env.
void InitSymbolEnvFromGraphInputs(BuiltExecuteGraph* built, const fxrt::ir::Graph& graph);

// Collect free-var seeds from graph.inputs symbolic shapes into env.
void CollectGraphInputSeeds(const fxrt::ir::Graph& graph, FxSymbolEnv& env);

// Build InferShape rule from out-node symbolic shape + shared SymbolEnv.
// Returns nullptr if the node has no usable shape, env is missing, or a dim fails to parse
// (caller must fail Convert).
std::shared_ptr<FxShapeRule> BuildOutputSymbolicShapeRule(
    const fxrt::ir::NodePtr& out_node,
    FxSymbolEnv* env,
    const char* source_tag);

// Tensor-level core of BuildOutputSymbolicShapeRule: tuple-valued ops (TopKV2) build the
// rule from one output tensor (slot 0); the InferShape kernel mirrors the dims onto every
// output slot (values/indices share dims).
std::shared_ptr<FxShapeRule> BuildOutputSymbolicShapeRuleFromTensor(
    const fxrt::ir::TensorPtr& out_tensor,
    FxSymbolEnv* env,
    const char* source_tag);

} // namespace fx_converter
