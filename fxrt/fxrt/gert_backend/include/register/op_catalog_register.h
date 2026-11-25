#pragma once

#include <cstdint>
#include <list>
#include <string>
#include <vector>

#include "register/op_catalog.h"

namespace fx_converter {

// One GE attribute described from Python. src is a GeAttrType name (bool, int, float, str, list_int).
// derived is a GeAttrDerived name ("", "input_count", "output_dtype"): the lowering fills the value.
struct CatalogAttrSpec {
  std::string name;
  std::string src;
  int64_t literal = 0;
  std::string text;
  bool required = false;
  std::string derived;
};

// Overlay or add a catalog row without rebuilding. logical_name must already be an ops.list name.
// recipe is tensor_io or as_strided. attr src must already exist.
struct CatalogOpSpec {
  std::string logical_name;
  std::string opp_type;
  size_t arity = 2;
  std::string lowering = "aicore";
  std::string cni_io = "indexed0";
  std::string recipe = "tensor_io";
  int32_t shape_input_index = -1;
  bool shape_list_broadcast_target = true;
  std::string opp_input_names;
  std::string opp_output_names;
  // CSV of forced GE dtypes per output; empty cells keep the inferred dtype (",int32").
  std::string opp_output_dtypes;
  bool shape_list_as_int32 = false;
  bool capture_axis = false;
  // GE attr the captured axis is published to (e.g. "dim" / "concat_dim"); empty = axis only.
  // When set, the name must also appear in attrs so Apply actually writes it.
  std::string axis_attr;
  bool capture_topk_bools = false;
  bool capture_fill_value = false;
  bool capture_skip_int = false;
  bool capture_skip_bool = false;
  bool infer_transpose_from_layout = false;
  bool prefer_high_performance = true;
  bool bool_as_int8 = false;
  bool strided_inputs = false;
  std::string select_suffix;
  std::string infer_depends;
  std::string optional_inputs;
  std::vector<CatalogAttrSpec> attrs;
};

// Process-wide catalog: builtin rows loaded at shared-library init plus Python overlays.
// Both Register overloads funnel into one tail (RegisterBuilt); a plain behavior is stored
// as nullptr for either source. Rows live until process exit by design; handed-out
// OpCatalogEntry pointers stay valid.
class CatalogRegistry {
 public:
  // Opaque storage row, defined in the .cc. Publicly named so file-local helpers can use it.
  struct OwnedCatalogOp;

  static CatalogRegistry& Instance();

  // Builtin path (FX_CATALOG_OP): trusted compile-time row; replaces the same-named row.
  bool Register(const CatalogRowBuilder& row, const BehaviorBuilder& behavior = BehaviorBuilder());
  // Python overlay path: parses and validates the spec strings; err describes the first failure.
  // Row attr cap (8) / behavior role cap (8) reject with err instead of silently truncating.
  bool Register(const CatalogOpSpec& spec, std::string* err);

  const OpCatalogEntry* Find(const std::string& logical_name) const;

  // Append each row's logical name. Overlay rows replace the same-named builtin slot,
  // so one pass covers both sources.
  void AppendLogicalNames(std::vector<std::string>* names) const;

  CatalogRegistry(const CatalogRegistry&) = delete;
  CatalogRegistry& operator=(const CatalogRegistry&) = delete;

 private:
  enum class RegisterSource : uint8_t { kBuiltin, kPythonOverlay };

  CatalogRegistry();
  ~CatalogRegistry();

  OwnedCatalogOp* FindMutable(const std::string& logical_name);
  // Shared tail of both Register overloads: validate, fold plain behavior, find-or-create
  // the slot, deep-copy strings/attrs/roles into it, rebind pointers, log by source.
  bool RegisterBuilt(
      const OpCatalogEntry& built,
      const OpBehavior& behavior,
      bool overflowed,
      RegisterSource source,
      std::string* err);

  std::list<OwnedCatalogOp> rows_;
};

// Loads one builtin row when the shared library loads. Parenthesize each argument so
// commas inside the builder chain stay part of that argument.
struct CatalogRegistrar {
  explicit CatalogRegistrar(const CatalogRowBuilder& row, const BehaviorBuilder& behavior = BehaviorBuilder()) {
    CatalogRegistry::Instance().Register(row, behavior);
  }
};

} // namespace fx_converter

#define FX_CATALOG_CONCAT_INNER(a, b) a##b
#define FX_CATALOG_CONCAT(a, b) FX_CATALOG_CONCAT_INNER(a, b)
#define FX_CATALOG_OP(...)                                                                                  \
  namespace {                                                                                               \
  static const ::fx_converter::CatalogRegistrar FX_CATALOG_CONCAT(_fx_catalog_reg_, __LINE__)(__VA_ARGS__); \
  }
