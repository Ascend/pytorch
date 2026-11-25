#include "lowering/catalog_attrs.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "common/logger.h"

namespace fx_converter {
namespace {

const char *OpTypeOrUnknown(const char *op_type_for_log) {
  return (op_type_for_log != nullptr && op_type_for_log[0] != '\0') ? op_type_for_log : "<unknown>";
}

bool NameEq(const char *left, const char *right) {
  return left != nullptr && right != nullptr && std::strcmp(left, right) == 0;
}

bool ExtrasHas(const CatalogAttrExtras *extras, const char *name) {
  if (extras == nullptr) {
    return false;
  }
  const CatalogNamedAttr *end = extras->slots + extras->count;
  return std::any_of(extras->slots, end, [name](const CatalogNamedAttr &slot) { return NameEq(slot.name, name); });
}

void PublishDerivedAttrs(const OpCatalogEntry *entry, CatalogAttrExtras *extras, int64_t axis, size_t input_count) {
  if (entry == nullptr || extras == nullptr) {
    return;
  }
  const OpBehavior *behavior = entry->behavior;
  if (axis != std::numeric_limits<int64_t>::min() && behavior != nullptr && behavior->input_roles != nullptr) {
    for (uint8_t i = 0; i < behavior->input_role_count; ++i) {
      const InputRoleRule &rule = behavior->input_roles[i];
      if (rule.role == InputRole::kAxisAttr && rule.attr_name != nullptr && !ExtrasHas(extras, rule.attr_name)) {
        extras->SetInt(rule.attr_name, axis);
      }
    }
  }
  if (input_count == 0) {
    return;
  }
  for (uint8_t i = 0; i < entry->attr_count; ++i) {
    const GeAttr &attr = entry->attrs[i];
    if (attr.derived == GeAttrDerived::kInputCount && attr.name != nullptr && !ExtrasHas(extras, attr.name)) {
      extras->SetInt(attr.name, static_cast<int64_t>(input_count));
    }
  }
}

float DefaultFloat(const GeAttr &attr) {
  if (attr.text != nullptr && attr.text[0] != '\0') {
    return std::strtof(attr.text, nullptr);
  }
  return static_cast<float>(attr.literal);
}

const CatalogNamedAttr *FindNamedAttr(const CatalogAttrExtras &extras, const char *name) {
  const CatalogNamedAttr *end = extras.slots + extras.count;
  const auto found = std::find_if(extras.slots, end,
                                  [name](const CatalogNamedAttr &slot) { return NameEq(slot.name, name); });
  return found == end ? nullptr : found;
}

CatalogNamedAttr *EnsureNamedAttr(CatalogAttrExtras *extras, const char *name) {
  if (extras == nullptr || name == nullptr || name[0] == '\0') {
    return nullptr;
  }
  if (const CatalogNamedAttr *found = FindNamedAttr(*extras, name)) {
    return const_cast<CatalogNamedAttr *>(found);
  }
  if (extras->count >= CatalogAttrExtras::kCap) {
    return nullptr;
  }
  CatalogNamedAttr &slot = extras->slots[extras->count++];
  slot.name = name;
  return &slot;
}

ResolvedCatalogAttr ResolveOne(const GeAttr &attr, const CatalogAttrExtras *extras) {
  ResolvedCatalogAttr out;
  out.name = attr.name != nullptr ? attr.name : "";
  out.type = attr.type;
  if (attr.type == GeAttrType::kStr) {
    const char *text = (extras != nullptr) ? extras->StrOr(attr.name, attr.text) : attr.text;
    out.s = text != nullptr ? text : "";
    return out;
  }
  out.b = (extras != nullptr) ? extras->BoolOr(attr.name, attr.literal != 0) : attr.literal != 0;
  out.i = (extras != nullptr) ? extras->IntOr(attr.name, attr.literal) : attr.literal;
  out.f = (extras != nullptr) ? extras->FloatOr(attr.name, DefaultFloat(attr)) : DefaultFloat(attr);
  return out;
}

}  // namespace

void CatalogAttrExtras::SetBool(const char *name, bool value) {
  CatalogNamedAttr *slot = EnsureNamedAttr(this, name);
  if (slot == nullptr) {
    return;
  }
  slot->kind = CatalogAttrKind::kBool;
  slot->b = value;
}

void CatalogAttrExtras::SetInt(const char *name, int64_t value) {
  CatalogNamedAttr *slot = EnsureNamedAttr(this, name);
  if (slot == nullptr) {
    return;
  }
  slot->kind = CatalogAttrKind::kInt;
  slot->i = value;
  slot->f = static_cast<float>(value);
}

void CatalogAttrExtras::SetDouble(const char *name, double value) {
  CatalogNamedAttr *slot = EnsureNamedAttr(this, name);
  if (slot == nullptr) {
    return;
  }
  slot->kind = CatalogAttrKind::kDouble;
  slot->f = static_cast<float>(value);
  slot->i = 0;
}

void CatalogAttrExtras::SetStr(const char *name, const char *value) {
  CatalogNamedAttr *slot = EnsureNamedAttr(this, name);
  if (slot == nullptr) {
    return;
  }
  slot->kind = CatalogAttrKind::kStr;
  slot->s = value;
}

bool CatalogAttrExtras::BoolOr(const char *name, bool fallback) const {
  const CatalogNamedAttr *slot = FindNamedAttr(*this, name);
  if (slot == nullptr) {
    return fallback;
  }
  if (slot->kind == CatalogAttrKind::kBool) {
    return slot->b;
  }
  if (slot->kind == CatalogAttrKind::kInt) {
    return slot->i != 0;
  }
  return fallback;
}

int64_t CatalogAttrExtras::IntOr(const char *name, int64_t fallback) const {
  const CatalogNamedAttr *slot = FindNamedAttr(*this, name);
  return (slot != nullptr && slot->kind == CatalogAttrKind::kInt) ? slot->i : fallback;
}

float CatalogAttrExtras::FloatOr(const char *name, float fallback) const {
  const CatalogNamedAttr *slot = FindNamedAttr(*this, name);
  if (slot == nullptr) {
    return fallback;
  }
  if (slot->kind == CatalogAttrKind::kDouble || slot->kind == CatalogAttrKind::kInt) {
    return slot->f;
  }
  if (slot->kind == CatalogAttrKind::kBool) {
    return slot->b ? 1.f : 0.f;
  }
  return fallback;
}

const char *CatalogAttrExtras::StrOr(const char *name, const char *fallback) const {
  const CatalogNamedAttr *slot = FindNamedAttr(*this, name);
  return (slot != nullptr && slot->kind == CatalogAttrKind::kStr && slot->s != nullptr) ? slot->s : fallback;
}

bool ResolveCatalogAttrs(const OpCatalogEntry *entry, int64_t axis, const CatalogAttrExtras *extras,
                         size_t input_count, std::vector<ResolvedCatalogAttr> *out, const char *op_type_for_log) {
  if (out == nullptr) {
    return false;
  }
  out->clear();
  if (entry == nullptr) {
    return true;
  }
  CatalogAttrExtras local;
  if (extras != nullptr) {
    local = *extras;
  }
  PublishDerivedAttrs(entry, &local, axis, input_count);
  const GeAttr *req = entry->attrs;
  for (uint8_t i = 0; i < entry->attr_count && req != nullptr; ++i) {
    if (req[i].required && !ExtrasHas(&local, req[i].name)) {
      RT_GLOG(ERROR) << "CNI requires attr '" << (req[i].name != nullptr ? req[i].name : "")
                     << "'; refusing default for type=" << OpTypeOrUnknown(op_type_for_log);
      return false;
    }
  }
  out->reserve(entry->attr_count);
  const GeAttr *attrs = entry->attrs;
  for (uint8_t i = 0; i < entry->attr_count && attrs != nullptr; ++i) {
    out->push_back(ResolveOne(attrs[i], &local));
  }
  return true;
}

}  // namespace fx_converter
