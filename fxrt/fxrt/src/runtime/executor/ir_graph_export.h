/**
 * @file ir_graph_export.h
 * @brief Versioned IR export contract for out-of-tree graph consumers.
 *
 * fxrt flattens its ir::Graph into the DTO below, which GraphExecutor::ExportIrGraph
 * hands to the Python layer as a plain dict (see python/pybind/fxrt/pybind_ir.cpp).
 * An out-of-tree consumer rebuilds its own graph from that dict. Nothing but POD and
 * strings crosses the boundary, by design: the only consumer today is the GE-based FX
 * converter, whose host libraries are built _GLIBCXX_USE_CXX11_ABI=0 while fxrt, torch
 * and torch_npu are ABI=1. A single ir::Graph object cannot be shared across that line.
 *
 * This is therefore NOT a temporary workaround to be deleted once "ABIs are unified" --
 * GE host libraries are ABI=0 by construction, so the boundary is permanent until GE
 * itself moves. Earlier revisions of this comment said otherwise; do not act on that.
 *
 * Changing the shape of what crosses the boundary is a cross-repository contract change:
 *   1. the field list lives in ir_graph_export_fields.inc and is shared with the consumer;
 *   2. bump kIrExportSchemaVersion so a mismatched consumer fails loudly instead of
 *      silently mis-parsing (a dropped field reads back as "unset", not as an error);
 *   3. update the golden expectations in test/fxrt/.
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
 * @brief Version of the exported dict schema.
 *
 * A consumer must reject a dict whose "schema_version" differs from the version it was
 * built against, and must also reject a dict that carries no "schema_version" at all --
 * that is a producer from before the schema was versioned, which is known to omit the
 * layout and symbol-channel keys. Bump this on any change to ir_graph_export_fields.inc.
 */
constexpr int kIrExportSchemaVersion = 1;

/**
 * @brief Key names of the exported per-node dict.
 *
 * Writing a key through these constants rather than a bare string literal turns a
 * misspelling into a compile error. See ir_graph_export_fields.inc for the contract.
 */
namespace ir_export_keys {
#define IR_EXPORT_FIELD(key, always_emitted) constexpr const char* key = #key;
#include "runtime/executor/ir_graph_export_fields.inc"
#undef IR_EXPORT_FIELD
} // namespace ir_export_keys

/**
 * @brief One declared key: its name and whether every node dict carries it.
 */
struct IrExportFieldInfo {
  const char* key;
  bool always_emitted;
};

/**
 * @brief Every key the schema declares, in declaration order.
 *
 * The dict writer is checked against this table, so a key that is written but never
 * declared -- or declared always_emitted but left out -- is caught instead of reaching a
 * consumer as a silently missing field.
 */
inline const std::vector<IrExportFieldInfo>& IrExportFields() {
  static const std::vector<IrExportFieldInfo> fields = {
#define IR_EXPORT_FIELD(key, always_emitted) IrExportFieldInfo{#key, (always_emitted) != 0},
#include "runtime/executor/ir_graph_export_fields.inc"
#undef IR_EXPORT_FIELD
  };
  return fields;
}

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
  std::vector<int64_t> stride; // Tensor element strides (empty = dense)
  std::vector<int64_t> storage_shape; // Physical storage shape
  int64_t storage_offset{0}; // Element offset into storage
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

  // Symbol channel for int_list, index-aligned: a non-empty entry means that element is a
  // symbolic dim and int_list only holds a placeholder for it. Empty means the list is all
  // ints. Both channels matter -- a stride slot left holding its placeholder would read
  // back as a real stride.
  std::vector<std::string> sym_list;

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
