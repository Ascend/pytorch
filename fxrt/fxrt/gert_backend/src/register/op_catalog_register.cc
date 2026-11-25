#include "register/op_catalog_register.h"

#include <algorithm>
#include <list>
#include <utility>

#include "common/logger.h"

namespace fx_converter {

struct CatalogRegistry::OwnedCatalogOp {
  std::string logical_name;
  std::string opp_type;
  std::string opp_input_names;
  std::string opp_output_names;
  std::string opp_output_dtypes;
  std::string select_suffix;
  std::string infer_depends;
  std::string optional_inputs;
  std::vector<std::string> attr_names;
  std::vector<std::string> attr_texts;
  std::vector<GeAttr> attrs;
  std::vector<std::string> role_attr_names;
  std::vector<InputRoleRule> input_roles;
  std::vector<OutputRef> output_refs;
  OpBehavior behavior;
  OpCatalogEntry entry;
};

namespace {

using OwnedCatalogOp = CatalogRegistry::OwnedCatalogOp;

const char *NonEmpty(const std::string &text) { return text.empty() ? nullptr : text.c_str(); }

template <typename T>
bool MatchNamed(const std::string &text, const T *rows, size_t count, const char *what, typename T::Value *out,
                std::string *err) {
  for (size_t i = 0; i < count; ++i) {
    if (text == rows[i].name) {
      *out = rows[i].value;
      return true;
    }
  }
  if (err != nullptr) {
    *err = "unknown " + std::string(what) + " '" + text + "' (expect ";
    for (size_t i = 0; i < count; ++i) {
      *err += (i == 0) ? "" : "|";
      *err += rows[i].name;
    }
    *err += ")";
  }
  return false;
}

struct RecipeRow {
  using Value = InputRecipe;
  const char *name;
  InputRecipe value;
};

struct LoweringRow {
  using Value = LoweringKind;
  const char *name;
  LoweringKind value;
};

struct IoRow {
  using Value = CniIoStyle;
  const char *name;
  CniIoStyle value;
};

struct AttrTypeRow {
  using Value = GeAttrType;
  const char *name;
  GeAttrType value;
};

struct AttrDerivedRow {
  using Value = GeAttrDerived;
  const char *name;
  GeAttrDerived value;
};

bool ParseRecipe(const std::string &text, InputRecipe *out, std::string *err) {
  static const RecipeRow kRows[] = {
      {"tensor_io", InputRecipe::kTensorIo},
      {"as_strided", InputRecipe::kAsStrided},
  };
  return MatchNamed(text, kRows, sizeof(kRows) / sizeof(kRows[0]), "recipe", out, err);
}

bool ParseLowering(const std::string &text, LoweringKind *out, std::string *err) {
  static const LoweringRow kRows[] = {
      {"dual_stage", LoweringKind::kDualStageAclnn},
      {"aicore", LoweringKind::kAiCoreBinary},
  };
  return MatchNamed(text, kRows, sizeof(kRows) / sizeof(kRows[0]), "lowering", out, err);
}

bool ParseIo(const std::string &text, CniIoStyle *out, std::string *err) {
  static const IoRow kRows[] = {
      {"indexed0", CniIoStyle::kIndexedFrom0},
      {"indexed1", CniIoStyle::kIndexedFrom0},  // names belong in opp_input_names
      {"packed_x", CniIoStyle::kDynamicPackedX},
  };
  return MatchNamed(text, kRows, sizeof(kRows) / sizeof(kRows[0]), "cni_io", out, err);
}

bool ParseAttrType(const std::string &text, GeAttrType *out, std::string *err) {
  static const AttrTypeRow kRows[] = {
      {"bool", GeAttrType::kBool},
      {"int", GeAttrType::kInt},
      {"float", GeAttrType::kFloat},
      {"str", GeAttrType::kStr},
      {"list_int", GeAttrType::kListInt},
  };
  return MatchNamed(text, kRows, sizeof(kRows) / sizeof(kRows[0]), "attr type", out, err);
}

bool ParseAttrDerived(const std::string &text, GeAttrDerived *out, std::string *err) {
  static const AttrDerivedRow kRows[] = {
      {"", GeAttrDerived::kNone},
      {"input_count", GeAttrDerived::kInputCount},
      {"output_dtype", GeAttrDerived::kOutputDtype},
  };
  return MatchNamed(text, kRows, sizeof(kRows) / sizeof(kRows[0]), "attr derived", out, err);
}

bool ParseAttrTypes(const CatalogOpSpec &spec, std::vector<GeAttrType> *types, std::vector<GeAttrDerived> *deriveds,
                    std::string *err) {
  types->clear();
  deriveds->clear();
  types->reserve(spec.attrs.size());
  deriveds->reserve(spec.attrs.size());
  for (const auto &attr : spec.attrs) {
    if (attr.name.empty()) {
      if (err != nullptr) {
        *err = "catalog attr is missing a name";
      }
      return false;
    }
    GeAttrType type = GeAttrType::kBool;
    if (!ParseAttrType(attr.src, &type, err)) {
      return false;
    }
    GeAttrDerived derived = GeAttrDerived::kNone;
    if (!ParseAttrDerived(attr.derived, &derived, err)) {
      return false;
    }
    types->push_back(type);
    deriveds->push_back(derived);
  }
  return true;
}

// A named axis_attr must also be declared in attrs; otherwise the captured axis is
// published into extras but never written by Apply (silent drop).
bool ValidateAxisAttr(const CatalogOpSpec &spec, std::string *err) {
  if (!spec.capture_axis || spec.axis_attr.empty()) {
    return true;
  }
  const auto found = std::find_if(spec.attrs.begin(), spec.attrs.end(), [&spec](const CatalogAttrSpec &attr) {
    return attr.name == spec.axis_attr;
  });
  if (found != spec.attrs.end()) {
    return true;
  }
  if (err != nullptr) {
    *err = "axis_attr '" + spec.axis_attr + "' is not declared in attrs";
  }
  return false;
}

// Only the dtype vocabulary the lowering knows how to force is accepted here.
bool ValidateOutputDtypes(const CatalogOpSpec &spec, std::string *err) {
  if (spec.opp_output_dtypes.empty()) {
    return true;
  }
  const size_t count = CatalogCsvCount(spec.opp_output_dtypes.c_str());
  for (size_t i = 0; i < count; ++i) {
    const std::string cell = CatalogCsvField(spec.opp_output_dtypes.c_str(), i);
    if (cell.empty() || cell == "int32") {
      continue;
    }
    if (err != nullptr) {
      *err = "unknown opp_output_dtypes cell '" + cell + "' (expect int32 or empty)";
    }
    return false;
  }
  return true;
}

void ApplyBehaviorFlags(BehaviorBuilder *builder, const CatalogOpSpec &spec) {
  if (spec.capture_axis) {
    builder->CaptureAxis(spec.axis_attr.empty() ? nullptr : spec.axis_attr.c_str());
  }
  if (spec.capture_topk_bools) {
    builder->CaptureTopkBools();
  }
  if (spec.capture_fill_value) {
    builder->CaptureFill();
  }
  if (spec.capture_skip_int) {
    builder->SkipInt();
  }
  if (spec.capture_skip_bool) {
    builder->SkipBool();
  }
  if (spec.infer_transpose_from_layout) {
    builder->InferTransposeFromLayout();
  }
  builder->PreferHighPerformance(spec.prefer_high_performance);
  if (spec.bool_as_int8) {
    builder->BoolAsInt8();
  }
}

// Fills a BehaviorBuilder from a Python spec. The builder points at spec storage, which
// outlives the registration call; the shared tail deep-copies everything into the slot.
BehaviorBuilder BuildSpecBehavior(const CatalogOpSpec &spec, InputRecipe recipe) {
  BehaviorBuilder builder;
  if (recipe == InputRecipe::kAsStrided) {
    builder.AsStrided();
  }
  ApplyBehaviorFlags(&builder, spec);
  if (!spec.select_suffix.empty()) {
    builder.SelectSuffix(spec.select_suffix.c_str());
  }
  return builder;
}

// Fills a CatalogRowBuilder from a Python spec. Same lifetime contract as BuildSpecBehavior.
// Attr() appends use the row's owned-storage cap (8).
CatalogRowBuilder BuildSpecRow(const CatalogOpSpec &spec, LoweringKind lowering, CniIoStyle io,
                               const std::vector<GeAttrType> &types, const std::vector<GeAttrDerived> &deriveds) {
  CatalogRowBuilder row(spec.logical_name.c_str());
  row.Opp(spec.opp_type.c_str()).Lowering(lowering);
  if (spec.strided_inputs) row.StridedInputs();
  if (spec.arity != 2U) {
    row.Arity(spec.arity);
  }
  if (io == CniIoStyle::kDynamicPackedX) {
    row.PackedX();
  }
  if (spec.shape_input_index >= 0) {
    row.ShapeInput(spec.shape_input_index);
  }
  if (!spec.shape_list_broadcast_target) {
    row.ShapeListVector();
  }
  if (!spec.opp_input_names.empty()) {
    row.Inputs(spec.opp_input_names.c_str());
  }
  if (!spec.opp_output_names.empty()) {
    row.Outputs(spec.opp_output_names.c_str());
  }
  if (!spec.opp_output_dtypes.empty()) {
    row.OutputDtypes(spec.opp_output_dtypes.c_str());
  }
  if (spec.shape_list_as_int32) {
    row.ShapeListInt32();
  }
  if (!spec.infer_depends.empty()) {
    row.InferDepends(spec.infer_depends.c_str());
  }
  if (!spec.optional_inputs.empty()) {
    row.OptionalInputs(spec.optional_inputs.c_str());
  }
  for (size_t i = 0; i < spec.attrs.size(); ++i) {
    const int64_t literal = spec.attrs[i].literal;
    const char *text = spec.attrs[i].text.empty() ? nullptr : spec.attrs[i].text.c_str();
    if (deriveds[i] != GeAttrDerived::kNone) {
      row.DerivedAttr(spec.attrs[i].name.c_str(), types[i], deriveds[i], spec.attrs[i].required, literal, text);
    } else if (spec.attrs[i].required) {
      row.RequiredAttr(spec.attrs[i].name.c_str(), types[i], literal, text);
    } else {
      row.Attr(spec.attrs[i].name.c_str(), types[i], literal, text);
    }
  }
  return row;
}

const char *OrEmpty(const char *text) { return text != nullptr ? text : ""; }

bool IsPlainBehavior(const OpBehavior &behavior) {
  return behavior.recipe == InputRecipe::kTensorIo && behavior.input_role_count == 0 &&
         !behavior.infer_transpose_from_layout && behavior.prefer_high_performance &&
         !behavior.bool_as_int8 && !behavior.select_single_output && behavior.select_suffix == nullptr;
}

void StoreStrings(OwnedCatalogOp *slot, const OpCatalogEntry &built, const OpBehavior &behavior,
                  bool has_behavior) {
  slot->logical_name = OrEmpty(built.logical_name);
  slot->opp_type = OrEmpty(built.opp_type);
  slot->opp_input_names = OrEmpty(built.opp_input_names);
  slot->opp_output_names = OrEmpty(built.opp_output_names);
  slot->opp_output_dtypes = OrEmpty(built.opp_output_dtypes);
  slot->select_suffix = has_behavior ? OrEmpty(behavior.select_suffix) : "";
  slot->infer_depends = OrEmpty(built.infer_depends);
  slot->optional_inputs = OrEmpty(built.optional_inputs);
}

void StoreAttrs(OwnedCatalogOp *slot, const OpCatalogEntry &built) {
  slot->attr_names.clear();
  slot->attr_texts.clear();
  slot->attrs.clear();
  if (built.attr_count == 0 || built.attrs == nullptr) {
    return;
  }
  slot->attr_names.reserve(built.attr_count);
  slot->attr_texts.reserve(built.attr_count);
  slot->attrs.reserve(built.attr_count);
  for (uint8_t i = 0; i < built.attr_count; ++i) {
    const GeAttr &src = built.attrs[i];
    slot->attr_names.emplace_back(OrEmpty(src.name));
    slot->attr_texts.emplace_back(OrEmpty(src.text));
  }
  for (uint8_t i = 0; i < built.attr_count; ++i) {
    GeAttr attr = built.attrs[i];
    attr.name = slot->attr_names[i].c_str();
    attr.text = slot->attr_texts[i].empty() ? nullptr : slot->attr_texts[i].c_str();
    slot->attrs.push_back(attr);
  }
}

void StoreOutputRefs(OwnedCatalogOp *slot, const OpCatalogEntry &built) {
  slot->output_refs.clear();
  if (built.output_ref_count == 0 || built.output_refs == nullptr) {
    return;
  }
  slot->output_refs.assign(built.output_refs, built.output_refs + built.output_ref_count);
}

void StoreInputRoles(OwnedCatalogOp *slot, const OpBehavior &behavior) {
  slot->role_attr_names.clear();
  slot->input_roles.clear();
  if (behavior.input_role_count == 0 || behavior.input_roles == nullptr) {
    return;
  }
  slot->role_attr_names.reserve(behavior.input_role_count);
  slot->input_roles.reserve(behavior.input_role_count);
  for (uint8_t i = 0; i < behavior.input_role_count; ++i) {
    slot->role_attr_names.emplace_back(OrEmpty(behavior.input_roles[i].attr_name));
    slot->input_roles.push_back(behavior.input_roles[i]);
  }
  for (size_t i = 0; i < slot->input_roles.size(); ++i) {
    slot->input_roles[i].attr_name =
        slot->role_attr_names[i].empty() ? nullptr : slot->role_attr_names[i].c_str();
  }
}

// Rebinds the stored behavior's pointer fields at the slot's own storage.
void RebindSlotBehavior(OwnedCatalogOp *slot) {
  slot->behavior.select_suffix = NonEmpty(slot->select_suffix);
  slot->behavior.input_roles = slot->input_roles.empty() ? nullptr : slot->input_roles.data();
  slot->behavior.input_role_count = static_cast<uint8_t>(slot->input_roles.size());
}

void PointEntryAtSlot(OwnedCatalogOp *slot, const OpCatalogEntry &built, bool has_behavior) {
  slot->entry = built;
  slot->entry.logical_name = slot->logical_name.c_str();
  slot->entry.opp_type = slot->opp_type.c_str();
  slot->entry.opp_input_names = NonEmpty(slot->opp_input_names);
  slot->entry.opp_output_names = NonEmpty(slot->opp_output_names);
  slot->entry.opp_output_dtypes = NonEmpty(slot->opp_output_dtypes);
  slot->entry.infer_depends = NonEmpty(slot->infer_depends);
  slot->entry.optional_inputs = NonEmpty(slot->optional_inputs);
  slot->entry.attrs = slot->attrs.empty() ? nullptr : slot->attrs.data();
  slot->entry.attr_count = static_cast<uint8_t>(slot->attrs.size());
  slot->entry.output_refs = slot->output_refs.empty() ? nullptr : slot->output_refs.data();
  slot->entry.output_ref_count = static_cast<uint8_t>(slot->output_refs.size());
  slot->entry.behavior = has_behavior ? &slot->behavior : nullptr;
}

}  // namespace

CatalogRegistry &CatalogRegistry::Instance() {
  static CatalogRegistry registry;
  return registry;
}

CatalogRegistry::CatalogRegistry() = default;

CatalogRegistry::~CatalogRegistry() = default;

CatalogRegistry::OwnedCatalogOp *CatalogRegistry::FindMutable(const std::string &logical_name) {
  const auto found = std::find_if(rows_.begin(), rows_.end(), [&logical_name](const OwnedCatalogOp &row) {
    return row.logical_name == logical_name;
  });
  return found == rows_.end() ? nullptr : &*found;
}

bool CatalogRegistry::Register(const CatalogRowBuilder &row, const BehaviorBuilder &behavior) {
  return RegisterBuilt(row.Build(), behavior.Build(), row.Overflowed() || behavior.Overflowed(),
                       RegisterSource::kBuiltin, nullptr);
}

bool CatalogRegistry::Register(const CatalogOpSpec &spec, std::string *err) {
  if (spec.logical_name.empty() || spec.opp_type.empty()) {
    if (err != nullptr) {
      *err = "catalog op requires logical_name and opp_type";
    }
    return false;
  }
  InputRecipe recipe = InputRecipe::kTensorIo;
  LoweringKind lowering = LoweringKind::kAiCoreBinary;
  CniIoStyle io = CniIoStyle::kIndexedFrom0;
  std::vector<GeAttrType> types;
  std::vector<GeAttrDerived> deriveds;
  if (!ParseRecipe(spec.recipe, &recipe, err) || !ParseLowering(spec.lowering, &lowering, err) ||
      !ParseIo(spec.cni_io, &io, err) || !ParseAttrTypes(spec, &types, &deriveds, err) ||
      !ValidateAxisAttr(spec, err) || !ValidateOutputDtypes(spec, err)) {
    return false;
  }
  const BehaviorBuilder behavior = BuildSpecBehavior(spec, recipe);
  const CatalogRowBuilder row = BuildSpecRow(spec, lowering, io, types, deriveds);
  return RegisterBuilt(row.Build(), behavior.Build(), row.Overflowed() || behavior.Overflowed(),
                       RegisterSource::kPythonOverlay, err);
}

bool CatalogRegistry::RegisterBuilt(const OpCatalogEntry &built, const OpBehavior &behavior, bool overflowed,
                                    RegisterSource source, std::string *err) {
  if (built.logical_name == nullptr || built.logical_name[0] == '\0' || built.opp_type == nullptr ||
      built.opp_type[0] == '\0') {
    if (err != nullptr) {
      *err = "catalog op requires logical_name and opp_type";
    }
    return false;
  }
  if (overflowed) {
    if (err != nullptr) {
      *err = "catalog row overflowed (too many attrs, input roles, or output refs)";
    }
    return false;
  }
  const bool has_behavior = !IsPlainBehavior(behavior);
  const bool overlay = source == RegisterSource::kPythonOverlay;
  OwnedCatalogOp *slot = FindMutable(built.logical_name);
  const bool replaced = slot != nullptr;
  if (replaced && !overlay) {
    // Duplicate logical_name in the builtin table: the later row wins, but flag it.
    RT_GLOG(WARNING) << "Builtin catalog op '" << built.logical_name << "' re-registered; the later row wins";
  }
  if (!replaced) {
    rows_.emplace_back();
    slot = &rows_.back();
  }
  StoreStrings(slot, built, behavior, has_behavior);
  StoreAttrs(slot, built);
  StoreOutputRefs(slot, built);
  StoreInputRoles(slot, has_behavior ? behavior : OpBehavior{});
  if (has_behavior) {
    slot->behavior = behavior;
    RebindSlotBehavior(slot);
  } else {
    slot->behavior = OpBehavior{};
  }
  PointEntryAtSlot(slot, built, has_behavior);
  if (overlay) {
    RT_GLOG(INFO) << "Catalog op '" << slot->logical_name << "' registered from python ("
                  << (replaced ? "replaced existing row" : "new row") << ")";
  }
  return true;
}

void CatalogRegistry::AppendLogicalNames(std::vector<std::string> *names) const {
  if (names == nullptr) {
    return;
  }
  for (const auto &row : rows_) {
    if (std::find(names->begin(), names->end(), row.logical_name) == names->end()) {
      names->push_back(row.logical_name);
    }
  }
}

const OpCatalogEntry *CatalogRegistry::Find(const std::string &logical_name) const {
  const auto found = std::find_if(rows_.begin(), rows_.end(), [&logical_name](const OwnedCatalogOp &row) {
    return row.logical_name == logical_name;
  });
  return found == rows_.end() ? nullptr : &found->entry;
}

}  // namespace fx_converter
