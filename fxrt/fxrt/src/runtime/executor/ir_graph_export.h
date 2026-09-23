/**
 * @file ir_graph_export.h
 * @brief IR graph export utilities for ABI compatibility.
 *
 * TEMPORARY SOLUTION: This file provides graph serialization functionality
 * to work around ABI incompatibility issues between different modules.
 *
 * This code should be removed once ABI consistency is properly resolved.
 * Tracking: Added in commit 496e33e (fx converter) and 8c130fa (expand)
 */

#ifndef __IR_GRAPH_EXPORT_H__
#define __IR_GRAPH_EXPORT_H__

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace fxrt {
namespace ir {
class Graph;
} // namespace ir

namespace runtime {

/**
 * @brief Export node in IR graph for ABI bridging.
 *
 * This structure represents a single node (input/parameter/op/return)
 * in the exported graph format.
 */
struct IrGraphNodeExport {
  std::string name; // Node name (input_0, param_0, op0, etc.)
  std::string kind; // Node kind: input | parameter | op | return
  std::string op; // Operation name (for kind="op")
  std::vector<std::string> inputs; // Input node names
  std::vector<int64_t> shape; // Output tensor shape
  std::vector<std::string> sym_shape; // Symbolic shape expressions
  std::string dtype{"float32"}; // Data type

  // Optional axis / concat_dim (e.g. aten.cat). INT64_MIN means unset.
  int64_t axis{std::numeric_limits<int64_t>::min()};

  // Optional float/bool scalar operand (e.g. eq_scalar / ge_scalar).
  bool has_scalar{false};
  double scalar{0.0};

  // Optional int list (e.g. expand size / BroadcastTo shape / TopKV2 k); empty means unset.
  // Mirrors GE Const INT64 shape tensors (CreateShapeConstantNode).
  std::vector<int64_t> int_list;

  // Optional bool attrs as 0/1 (TopKV2 GE order: sorted, largest). Empty means unset.
  std::vector<int64_t> bool_list;

  // Multi-output ops (TopKV2): per-output shapes/dtypes. Empty → use shape/dtype only.
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<std::string> output_dtypes;

  // Recursive ordered arguments and attributes preserve compile-time metadata
  // that cannot be represented by the normalized tensor-only `inputs` field.
  struct ValueExport {
    enum class Kind : uint8_t { kNone, kRef, kInt, kDouble, kBool, kString, kSymbol, kTuple };
    Kind kind{Kind::kNone};
    std::string ref;
    int64_t int_value{0};
    double double_value{0.0};
    bool bool_value{false};
    std::string string_value;
    std::vector<ValueExport> elements;
  };
  std::vector<ValueExport> arguments;
  std::unordered_map<std::string, ValueExport> attrs;
};

/**
 * @brief Exported IR graph structure.
 *
 * This structure contains a flattened list of exported nodes,
 * representing the entire computational graph in a serializable format.
 */
struct IrGraphExport {
  std::string name; // Graph name
  std::vector<IrGraphNodeExport> nodes; // All nodes in topological order
};

// Serialize fxrt::ir::Graph into the ABI bridge DTO (used by GraphExecutor::ExportIrGraph).
IrGraphExport ExportIrGraph(const ::fxrt::ir::Graph* graph, const std::string& name);

} // namespace runtime
} // namespace fxrt

#endif // __IR_GRAPH_EXPORT_H__
