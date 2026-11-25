#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "register/op_catalog.h"

namespace fx_converter {

// Path C recipe switches. Not GE attributes; do not add per-op attr fields here.
struct CatalogRecipe {
  bool use_as_strided = false;
};

// One collected attr, keyed by the GE name (same string as GeAttr::name).
// Kind/payload follow fxrt::ir::Value: a typed scalar, not a positional slot.
enum class CatalogAttrKind : uint8_t { kNone = 0, kBool, kInt, kDouble, kStr };

struct CatalogNamedAttr {
  const char* name = nullptr;
  CatalogAttrKind kind = CatalogAttrKind::kNone;
  bool b = false;
  int64_t i = 0;
  float f = 0.f;
  // Borrows the IR string storage; valid while the source graph lives (convert scope).
  const char* s = nullptr;
};

// Per-op named bag. Apply looks up GeAttr::name; a third bool is another name, not ir_bool2.
struct CatalogAttrExtras {
  static constexpr uint8_t kCap = 8;
  CatalogNamedAttr slots[kCap]{};
  uint8_t count = 0;
  CatalogRecipe recipe;

  void SetBool(const char* name, bool value);
  void SetInt(const char* name, int64_t value);
  void SetDouble(const char* name, double value);
  void SetStr(const char* name, const char* value);
  bool BoolOr(const char* name, bool fallback) const;
  int64_t IntOr(const char* name, int64_t fallback) const;
  float FloatOr(const char* name, float fallback) const;
  const char* StrOr(const char* name, const char* fallback) const;
};

// Resolved catalog attr for gert.source_op JSON (RuntimeAttrs slot order).
struct ResolvedCatalogAttr {
  std::string name;
  GeAttrType type = GeAttrType::kBool;
  bool b = false;
  int64_t i = 0;
  float f = 0.f;
  std::string s;
};

// Publish derived attrs, enforce REQUIRED_ATTR, and emit extras-or-default values
// in catalog order. Used by the public gert.source_op path (no OpDesc / AttrUtils).
bool ResolveCatalogAttrs(
    const OpCatalogEntry* entry,
    int64_t axis,
    const CatalogAttrExtras* extras,
    size_t input_count,
    std::vector<ResolvedCatalogAttr>* out,
    const char* op_type_for_log = nullptr);

} // namespace fx_converter
