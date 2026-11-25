#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <unordered_map>

// Structured IR snapshot DTO (Python export). Materialized as fxrt::ir::Graph in ABI=0.
enum class IrNodeKind {
  kInput = 0,
  kParameter = 1,
  kOp = 2,
  kReturn = 3,
};

struct IrValueSpec {
  enum class Kind { kNone, kRef, kInt, kDouble, kBool, kString, kSymbol, kTuple };
  Kind kind = Kind::kNone;
  std::string ref;
  int64_t int_value = 0;
  double double_value = 0.0;
  bool bool_value = false;
  std::string string_value;
  std::vector<IrValueSpec> elements;
};

struct IrNodeSpec {
  std::string name;
  IrNodeKind kind = IrNodeKind::kOp;
  std::string op;
  // Dispatcher overload name for custom/torch calls (e.g. "default", "Tensor", "out").
  // Kept separate from op so schema selection never depends on enumeration order.
  std::string overload;
  std::vector<std::string> inputs;
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  std::vector<int64_t> storage_shape;
  int64_t storage_offset = 0;
  // acl/ge format code. 2 is FORMAT_ND, the value used when the key is absent.
  int32_t format = 2;
  // Per tuple-output format. Empty means every slot uses `format`.
  std::vector<int32_t> output_formats;
  std::vector<std::string> sym_shape;
  // Tensor dtype string (e.g. float32 / float16 / bfloat16). Empty -> float32.
  std::string dtype;
  // Optional axis / concat_dim. INT64_MIN means unset.
  int64_t axis = std::numeric_limits<int64_t>::min();
  // Symbol channel for a single symbolic scalar (a Dynamo-lifted SymInt placeholder):
  // non-empty means one non-tensor input is symbolic and axis only holds a placeholder.
  // The axis is the main case; the channel carries any single symbolic scalar.
  std::string sym_axis;
  // Optional float/bool scalar operand (eq_scalar / ge_scalar).
  bool has_scalar = false;
  double scalar = 0.0;
  // Ordered channel for every literal Int operand of the node (axis alone keeps only the
  // last one). Non-empty for nodes like div_mod_scalar [x, scalar, mode]; when set, it
  // replaces the single axis-node rebuild in the import.
  std::vector<int64_t> extra_ints;
  // Symbol channel for extra_ints, index-aligned: a non-empty entry means that slot is a
  // symbolic scalar (arange limit=s27) and extra_ints only holds a placeholder there.
  std::vector<std::string> extra_syms;
  // Optional shape-list (Expand/BroadcastTo size / TopKV2 k); empty means unset. Rebuilt as Tuple Value.
  std::vector<int64_t> int_list;
  // Symbol channel for int_list, index-aligned: a non-empty entry means that element is a
  // symbolic dim and int_list only holds a placeholder for it. Empty means the list is all ints.
  std::vector<std::string> sym_list;
  // Optional bool attrs as 0/1 (TopKV2: sorted, largest). Empty means unset.
  std::vector<int64_t> bool_list;
  // Multi-output (TopKV2): when non-empty, rebuild output as Tuple of tensors.
  std::vector<std::vector<int64_t>> output_shapes;
  // Per-output symbolic shape expressions. Empty entries are allowed for static outputs.
  std::vector<std::vector<std::string>> output_sym_shapes;
  std::vector<std::string> output_dtypes;
  // Recursive raw arguments and backend metadata used by compiled kernels.
  std::vector<IrValueSpec> arguments;
  std::unordered_map<std::string, IrValueSpec> attrs;
};

struct IrGraphSpec {
  std::string name;
  std::vector<IrNodeSpec> nodes;
};
