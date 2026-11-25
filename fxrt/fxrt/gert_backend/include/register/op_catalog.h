#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ir/graph.h"

namespace fx_converter {

// How temp-OpDesc IR inputs are named when opp_input_names is empty.
enum class CniIoStyle : uint8_t {
  kIndexedFrom0 = 0, // x0, x1, ...
  kDynamicPackedX = 1, // one dynamic IR input "x" (Concat* DYNAMIC_INPUT)
};

// Generic tensor and integer-tuple IO, or the as_strided packing.
enum class InputRecipe : uint8_t {
  kTensorIo = 0,
  kAsStrided,
};

// IR value kind a non-tensor input role rule matches.
enum class IrInputKind : uint8_t {
  kInt = 0,
  kBool,
  kDouble,
  kIntTuple,
  kCount, // not a kind; ordinal arrays are sized from this
};
// kind_ordinals is indexed by IrInputKind. Append a new kind before kCount.
constexpr size_t kIrInputKindCount = static_cast<size_t>(IrInputKind::kCount);
static_assert(static_cast<uint8_t>(IrInputKind::kInt) == 0, "IrInputKind must stay dense from 0");

// What one non-tensor IR input becomes for this op. An op declares ordered
// (kind -> role) rules; skip/axis use the first matching kind. kNamedAttr rules
// of the same kind are consumed in order (first bool -> first named bool attr).
// Undeclared kinds keep the default: tuple -> shape list, int/double -> scalar
// tensor. Undeclared bool is ignored on the aicore path but rejected on the
// dual-stage path; declare kSkip for bool to drop it explicitly. kHostPack /
// kScalarTensorInt32 stay reserved.
enum class InputRole : uint8_t {
  kAxisAttr = 0, // Int -> GE axis / axes attribute
  kNamedAttr, // IR value -> extras[attr_name]; name must match GeAttr::name
  kSkip, // Drop the input (Cast ScalarType Int, ReduceSum keepdim Bool)
  kHostPack, // Append to the trailing host INT64 vector (reserved)
  kScalarTensorInt32, // Materialize as an INT32 tensor input (reserved)
};

struct InputRoleRule {
  IrInputKind match = IrInputKind::kInt;
  InputRole role = InputRole::kAxisAttr;
  const char* attr_name = nullptr; // kNamedAttr: GE attr name
  // -1 matches every occurrence of the kind; n>=0 matches only the n-th (0-based)
  // non-tensor input of that kind (div_mod_scalar skips the trailing mode Int while the
  // earlier scalar Int keeps the default materialization).
  int8_t occurrence = -1;
};

// GE IR type of one attribute. Same idea as REG_OP ATTR(name, Bool|Int|Float|String|ListInt, default).
enum class GeAttrType : uint8_t {
  kBool = 0,
  kInt,
  kFloat,
  kStr,
  kListInt,
};

// Value source for an attr the lowering derives at convert time instead of the
// IR exporter. kInputCount: collected tensor input count (ConcatD "N").
// kOutputDtype: GE dtype of the IR output (Cast "dst_type").
enum class GeAttrDerived : uint8_t {
  kNone = 0,
  kInputCount,
  kOutputDtype,
};

// One GE attr: name + type + default. Apply writes extras[name] if present, else default.
// required matches REG_OP REQUIRED_ATTR: extras must supply the value (no silent default).
// derived marks an attr whose value the lowering publishes into extras (see GeAttrDerived).
struct GeAttr {
  const char* name = nullptr;
  GeAttrType type = GeAttrType::kBool;
  int64_t literal = 0;
  const char* text = nullptr;
  bool required = false;
  GeAttrDerived derived = GeAttrDerived::kNone;
};

// Per-operator collect / select contract. GE attrs live on OpCatalogEntry.
// nullptr behavior on an OpCatalogEntry means plain tensor IO (no collect / select knobs).
struct OpBehavior {
  InputRecipe recipe = InputRecipe::kTensorIo;
  // Ordered non-tensor input classification rules (see InputRole).
  const InputRoleRule* input_roles = nullptr;
  uint8_t input_role_count = 0;
  // DualStage: infer transpose_x1/x2 (or adj_*) from last-two-dim strides, not IR bools.
  bool infer_transpose_from_layout = false;
  bool prefer_high_performance = true;
  bool bool_as_int8 = false;
  // Some CANN keys encode a single representative output slot for a multi-output op
  // (LayerNormV4 keys are 4-in+1-out while the launch keeps y/mean/rstd). Select with
  // one output spec only; launch/tiling op description keep every output.
  bool select_single_output = false;
  // CANN simplifiedKey suffix. Literal ("constant,1") or "{bool_attr}" template ("{sorted},,{largest}").
  const char* select_suffix = nullptr;
  // Normalize a negative captured axis to axis + input_rank + axis_rank_delta before kernel
  // selection and attr wiring. kNoAxisNormalize keeps the raw value (ops with native
  // negative-axis support, e.g. ArgMaxV2). Pack stacks along rank+1 slots (delta 1);
  // ConcatD counts existing dims only (delta 0). Their DSL wrappers mis-tile a raw
  // negative axis, so the converter always hands them a non-negative one.
  static constexpr int8_t kNoAxisNormalize = 127;
  int8_t axis_rank_delta = kNoAxisNormalize;
};

// Look up the declared role for one IR input kind. nullptr means the kind is
// undeclared and keeps its default handling.
inline const InputRoleRule* FindInputRole(const OpBehavior* behavior, IrInputKind kind) {
  if (behavior == nullptr || behavior->input_roles == nullptr) {
    return nullptr;
  }
  const InputRoleRule* roles = behavior->input_roles;
  const InputRoleRule* end = roles + behavior->input_role_count;
  const auto found = std::find_if(roles, end, [kind](const InputRoleRule& rule) { return rule.match == kind; });
  return found == end ? nullptr : found;
}

// Occurrence-aware lookup: an exact ordinal rule wins over the catch-all (-1) one.
inline const InputRoleRule* FindInputRoleAt(
    const OpBehavior* behavior,
    IrInputKind kind,
    InputRole role,
    int64_t ordinal) {
  if (behavior == nullptr || behavior->input_roles == nullptr) {
    return nullptr;
  }
  const InputRoleRule* roles = behavior->input_roles;
  const InputRoleRule* end = roles + behavior->input_role_count;
  const InputRoleRule* fallback = nullptr;
  for (const InputRoleRule* rule = roles; rule != end; ++rule) {
    if (rule->match != kind || rule->role != role) {
      continue;
    }
    if (rule->occurrence == ordinal) {
      return rule;
    }
    if (rule->occurrence < 0 && fallback == nullptr) {
      fallback = rule;
    }
  }
  return fallback;
}

// Any-rule semantics: a kind probes true when ANY rule of that kind carries the role.
// Probing only the first rule of the kind let a leading kSkip mask a later capture
// (AddRmsNormQuant declared its kDouble skips before CaptureAttr(kDouble, "epsilon"),
// which silently killed the capture and let the attr default ride).
inline bool HasInputRole(const OpBehavior* behavior, IrInputKind kind, InputRole role) {
  if (behavior == nullptr || behavior->input_roles == nullptr) {
    return false;
  }
  const InputRoleRule* roles = behavior->input_roles;
  const InputRoleRule* end = roles + behavior->input_role_count;
  return std::any_of(
      roles, end, [kind, role](const InputRoleRule& rule) { return rule.match == kind && rule.role == role; });
}

// Which OpLowering path owns this catalog row.
enum class LoweringKind : uint8_t {
  // OPP aclnn dual-stage: InferShape -> ExecuteOpPrepare -> AllocWs -> ExecuteOpLaunch
  kDualStageAclnn = 0,
  // CANN prebuilt AICore .o: InferShape -> TilingParse/Tiling -> SinkBin -> LaunchKernel*
  kAiCoreBinary = 1,
};

// One output writes one collected input's storage.
struct OutputRef {
  int32_t output = 0;
  int32_t input = 0;
};

// MRT ops.list name -> OPP type + CNI contract + lowering kind.
// Builtin FX_CATALOG_OP rows and Python register_op share one runtime table.
// Registering the same logical name replaces the existing row.
struct OpCatalogEntry {
  const char* logical_name = nullptr; // Must match fxrt::ops::ToStr (ops.list)
  const char* opp_type = nullptr; // Key for OpImplSpaceRegistryV2 / binary_info
  // Tensor IO count after collection; 0 means variadic (use every collected
  // tensor). A scalar the collect path materializes as a tensor
  // (eq_scalar / sub_scalar trailing scalar) counts toward arity.
  size_t arity = 2;
  CniIoStyle cni_io = CniIoStyle::kIndexedFrom0;
  // nullptr: plain tensor IO (no collect / select knobs). GE attrs stay on this entry.
  const OpBehavior* behavior = nullptr;
  LoweringKind lowering = LoweringKind::kDualStageAclnn;
  // OPP input index of an INT64 shape-list (e.g. Expand "shape", Tile "multiples"). -1 = none.
  // Name is opp_input_names[index] (CatalogShapeInputName). IR exports Tuple<int>.
  int32_t shape_input_index = -1;
  // Shape-list tiling StorageShape semantics (ignored when shape_input_index < 0):
  // true  = Broadcast target (Expand): StorageShape dims = list values.
  // false = Int-list vector (Tile multiples / TopKV2 k): StorageShape = [rank], payload = values.
  bool shape_list_broadcast_target = true;
  // Optional comma-separated OPP IR input names (e.g. "x,index" for GatherElements).
  // nullptr → CniIoStyle / AiCoreInputName defaults (x0/x1 or x + shape name).
  const char* opp_input_names = nullptr;
  // Optional comma-separated OPP IR output names (e.g. "values,indices" for TopKV2).
  // nullptr → "y" / "y0,y1".
  const char* opp_output_names = nullptr;
  // Optional CSV forcing a GE dtype per output; empty cells keep the inferred dtype
  // (",int32" forces output 1 to DT_INT32, as TopKV2 indices bins require).
  const char* opp_output_dtypes = nullptr;
  // When true, materialize shape-list / k as DT_INT32 (TopKV2 k bins); else DT_INT64.
  bool shape_list_as_int32 = false;
  // When true, the shape list carries reduce-style axes (ReduceMin/ReduceMax/ReduceMean):
  // negative literal values wrap by the leading tensor rank at Convert. Only for rows whose
  // list is axes -- never sizes (Expand) or multiples (Tile), where negatives are semantics.
  bool shape_list_axes_normalize = false;
  // ge::DataType value forcing the materialized scalar tensor dtype (e.g. ge::DT_INT32 == 3
  // for the Cumsum axis input, whose bins reject a FLOAT axis). -1 keeps the legacy rule
  // of following the leading tensor input dtype.
  int32_t scalar_dtype = -1;
  // Collect appends a constant INT32 vector [0] after the IR tensors. embedding,
  // gather_v2 and index_tensor are GatherV2(x, indices, axis=0): A5 has no Gather
  // binary, and a length-1 INT32 axis must stay a vector rather than a rank-0 scalar.
  bool append_dim0_axis = false;
  // GE IR schema (REG_OP ATTR / OPTIONAL_INPUT / infer depends). Not lowering collect.
  const GeAttr* attrs = nullptr;
  uint8_t attr_count = 0;
  const char* optional_inputs = nullptr;
  // Launch-only optional tail: padded as placeholder io slots (GE gen_placeholder) but
  // NOT registered via AppendIrInput, so tiling sees the optionals as absent and picks
  // the no-optional kernel function variant.
  const char* launch_pad_inputs = nullptr;
  const char* infer_depends = nullptr;
  // Consume the ordered-arguments channel at IR import: the normalized bag drops
  // None/bool/string operands and compacts Int/Double slots, so positional behavior
  // rules (ClassifyAt ordinals, string captures) are only meaningful against the full
  // schema layout. Rows with such rules must opt in; the import then sees every
  // operand in proto order.
  bool ordered_arguments = false;
  // Each pair is one output writing one collected input. Unlisted outputs allocate a
  // fresh buffer. The kernel stores into that address, so a later reader of the same
  // storage sees the write. IR order supplies the sequencing: a reader lowered before
  // this op still sees the previous bytes.
  // DualStage: input == arity is an optional output-only tail, removed from the
  // compute inputs when present. A missing tail keeps normal output allocation.
  const OutputRef* output_refs = nullptr;
  uint8_t output_ref_count = 0;
  // The selected ACLNN wrapper consumes logical TensorV2 layouts.
  bool strided_inputs = false;
};

// Fills an OpBehavior. Unset fields keep the struct defaults. Each call returns *this.
class BehaviorBuilder {
 public:
  constexpr BehaviorBuilder& Recipe(InputRecipe recipe) {
    behavior_.recipe = recipe;
    return *this;
  }
  constexpr BehaviorBuilder& AsStrided() {
    return Recipe(InputRecipe::kAsStrided);
  }
  // Declares how one non-tensor input kind is classified. Skip/axis: first
  // matching kind. kNamedAttr: same-kind rules are consumed in declaration order.
  BehaviorBuilder& Classify(IrInputKind match, InputRole role, const char* attr_name = nullptr) {
    return ClassifyAt(match, role, -1, attr_name);
  }
  // Classify with an occurrence qualifier (see InputRoleRule::occurrence).
  BehaviorBuilder& ClassifyAt(IrInputKind match, InputRole role, int8_t occurrence, const char* attr_name = nullptr) {
    if (owned_role_count_ >= kOwnedRoleCap) {
      overflowed_ = true;
      return *this;
    }
    owned_roles_[owned_role_count_] = InputRoleRule{match, role, attr_name, occurrence};
    ++owned_role_count_;
    behavior_.input_roles = owned_roles_;
    behavior_.input_role_count = owned_role_count_;
    return *this;
  }
  // name is the GE attr this Int becomes (dim / concat_dim / axes). nullptr: axis only.
  BehaviorBuilder& CaptureAxis(const char* name = nullptr) {
    return Classify(IrInputKind::kInt, InputRole::kAxisAttr, name);
  }
  BehaviorBuilder& CaptureAttr(IrInputKind match, const char* name) {
    return Classify(match, InputRole::kNamedAttr, name);
  }
  // Names must match the GeAttr list (TopK: sorted then largest).
  BehaviorBuilder& CaptureTopkBools() {
    return CaptureAttr(IrInputKind::kBool, "sorted").CaptureAttr(IrInputKind::kBool, "largest");
  }
  BehaviorBuilder& CaptureFill(const char* name = "value") {
    return CaptureAttr(IrInputKind::kInt, name)
        .CaptureAttr(IrInputKind::kDouble, name)
        .CaptureAttr(IrInputKind::kBool, name);
  }
  // Cast ScalarType Int is not a tensor IO; dest type is extras["dst_type"].
  BehaviorBuilder& SkipInt() {
    return Classify(IrInputKind::kInt, InputRole::kSkip);
  }
  // Drop only the n-th (0-based) Int among the non-tensor inputs; earlier Ints keep
  // their default handling (div_mod_scalar: scalar materializes, trailing mode drops).
  BehaviorBuilder& SkipIntAt(int8_t occurrence) {
    return ClassifyAt(IrInputKind::kInt, InputRole::kSkip, occurrence);
  }
  // Drop an IR bool input. ReduceSum: the rewrite lowers keepdim as a following view,
  // so the keepdim Bool is not an op input. Undeclared bools fail the dual-stage collect.
  BehaviorBuilder& SkipBool() {
    return Classify(IrInputKind::kBool, InputRole::kSkip);
  }
  constexpr BehaviorBuilder& InferTransposeFromLayout() {
    behavior_.infer_transpose_from_layout = true;
    return *this;
  }
  constexpr BehaviorBuilder& PreferHighPerformance(bool enabled) {
    behavior_.prefer_high_performance = enabled;
    return *this;
  }
  constexpr BehaviorBuilder& BoolAsInt8() {
    behavior_.bool_as_int8 = true;
    return *this;
  }
  // See OpBehavior::select_single_output. LayerNormV4 keys one representative output slot.
  constexpr BehaviorBuilder& SelectSingleOutput() {
    behavior_.select_single_output = true;
    return *this;
  }
  constexpr BehaviorBuilder& SelectSuffix(const char* suffix) {
    behavior_.select_suffix = suffix;
    return *this;
  }
  // See OpBehavior::axis_rank_delta. Pack: 1 (stack appends a rank+1-th slot);
  // ConcatD: 0 (concat counts existing dims).
  constexpr BehaviorBuilder& NormalizeAxis(int8_t rank_delta) {
    behavior_.axis_rank_delta = rank_delta;
    return *this;
  }
  constexpr OpBehavior Build() const {
    return behavior_;
  }
  // True when a Classify call exceeded the owned storage.
  bool Overflowed() const {
    return overflowed_;
  }

  BehaviorBuilder() = default;
  BehaviorBuilder(const BehaviorBuilder& other) {
    Adopt(other);
  }
  BehaviorBuilder& operator=(const BehaviorBuilder& other) {
    if (this != &other) {
      Adopt(other);
    }
    return *this;
  }

 private:
  static constexpr uint8_t kOwnedRoleCap = 8;

  void Adopt(const BehaviorBuilder& other) {
    owned_role_count_ = other.owned_role_count_;
    std::copy_n(other.owned_roles_, static_cast<size_t>(owned_role_count_), owned_roles_);
    overflowed_ = other.overflowed_;
    behavior_ = other.behavior_;
    if (other.behavior_.input_roles == other.owned_roles_) {
      behavior_.input_roles = owned_roles_;
    }
  }

  OpBehavior behavior_{};
  InputRoleRule owned_roles_[kOwnedRoleCap]{};
  uint8_t owned_role_count_ = 0;
  bool overflowed_ = false;
};

// Fills an OpCatalogEntry. Unset fields keep the struct defaults. Each call returns *this.
class CatalogRowBuilder {
 public:
  constexpr explicit CatalogRowBuilder(const char* logical_name) {
    entry_.logical_name = logical_name;
  }

  constexpr CatalogRowBuilder& Opp(const char* opp_type) {
    entry_.opp_type = opp_type;
    return *this;
  }
  constexpr CatalogRowBuilder& Arity(size_t arity) {
    entry_.arity = arity;
    return *this;
  }
  constexpr CatalogRowBuilder& Io(CniIoStyle io) {
    entry_.cni_io = io;
    return *this;
  }
  constexpr CatalogRowBuilder& PackedX() {
    return Io(CniIoStyle::kDynamicPackedX);
  }
  constexpr CatalogRowBuilder& Behavior(const OpBehavior* behavior) {
    entry_.behavior = behavior;
    return *this;
  }
  constexpr CatalogRowBuilder& Lowering(LoweringKind kind) {
    entry_.lowering = kind;
    return *this;
  }
  constexpr CatalogRowBuilder& AiCore() {
    return Lowering(LoweringKind::kAiCoreBinary);
  }
  constexpr CatalogRowBuilder& ShapeInput(int32_t index) {
    entry_.shape_input_index = index;
    return *this;
  }
  // Int-list vector (Tile multiples / TopKV2 k): StorageShape is [rank], not the list values.
  constexpr CatalogRowBuilder& ShapeListVector() {
    entry_.shape_list_broadcast_target = false;
    return *this;
  }
  constexpr CatalogRowBuilder& Inputs(const char* names) {
    entry_.opp_input_names = names;
    return *this;
  }
  constexpr CatalogRowBuilder& Outputs(const char* names) {
    entry_.opp_output_names = names;
    return *this;
  }
  // Output `output` writes collected input `input`. Repeat for every ref output.
  CatalogRowBuilder& Ref(int32_t output, int32_t input) {
    if (owned_ref_count_ >= kOwnedRefCap) {
      overflowed_ = true;
      return *this;
    }
    owned_refs_[owned_ref_count_] = OutputRef{output, input};
    ++owned_ref_count_;
    entry_.output_refs = owned_refs_;
    entry_.output_ref_count = owned_ref_count_;
    return *this;
  }
  // Output 0 writes collected input `input`.
  CatalogRowBuilder& RefInput(int32_t input) {
    return Ref(0, input);
  }
  constexpr CatalogRowBuilder& StridedInputs() {
    entry_.strided_inputs = true;
    return *this;
  }
  constexpr CatalogRowBuilder& ShapeListInt32() {
    entry_.shape_list_as_int32 = true;
    return *this;
  }
  // Mark the shape list as reduce-style axes: negative literals wrap by x rank at Convert.
  constexpr CatalogRowBuilder& NormalizeShapeListAxes() {
    entry_.shape_list_axes_normalize = true;
    return *this;
  }
  // Force the materialized scalar tensor dtype (ge::DataType value, e.g. 3 for DT_INT32).
  constexpr CatalogRowBuilder& ScalarDtype(int32_t ge_dtype) {
    entry_.scalar_dtype = ge_dtype;
    return *this;
  }
  // See OpCatalogEntry::append_dim0_axis.
  constexpr CatalogRowBuilder& AppendDim0Axis() {
    entry_.append_dim0_axis = true;
    return *this;
  }
  // CSV of forced GE dtypes per output; empty cells keep the inferred dtype.
  constexpr CatalogRowBuilder& OutputDtypes(const char* csv) {
    entry_.opp_output_dtypes = csv;
    return *this;
  }
  constexpr CatalogRowBuilder& OptionalInputs(const char* names) {
    entry_.optional_inputs = names;
    return *this;
  }
  constexpr CatalogRowBuilder& LaunchPadInputs(const char* names) {
    entry_.launch_pad_inputs = names;
    return *this;
  }
  // See OpCatalogEntry::ordered_arguments.
  constexpr CatalogRowBuilder& OrderedArguments() {
    entry_.ordered_arguments = true;
    return *this;
  }
  constexpr CatalogRowBuilder& InferDepends(const char* name) {
    entry_.infer_depends = name;
    return *this;
  }
  // GE-style ATTR(name, Type, default). text is the default for kStr.
  CatalogRowBuilder& Attr(const char* name, GeAttrType type, int64_t literal = 0, const char* text = nullptr) {
    return AddAttr(name, type, literal, text, false, GeAttrDerived::kNone);
  }
  // GE-style REQUIRED_ATTR(name, Type). extras must supply the value.
  CatalogRowBuilder& RequiredAttr(const char* name, GeAttrType type, int64_t literal = 0, const char* text = nullptr) {
    return AddAttr(name, type, literal, text, true, GeAttrDerived::kNone);
  }
  // Attr whose value the lowering derives at convert time (see GeAttrDerived).
  CatalogRowBuilder& DerivedAttr(
      const char* name,
      GeAttrType type,
      GeAttrDerived derived,
      bool required = false,
      int64_t literal = 0,
      const char* text = nullptr) {
    return AddAttr(name, type, literal, text, required, derived);
  }
  constexpr OpCatalogEntry Build() const {
    return entry_;
  }
  bool Overflowed() const {
    return overflowed_;
  }

  CatalogRowBuilder(const CatalogRowBuilder& other) {
    Adopt(other);
  }
  CatalogRowBuilder& operator=(const CatalogRowBuilder& other) {
    if (this != &other) {
      Adopt(other);
    }
    return *this;
  }

 private:
  // FlashAttentionScore declares 13 proto attrs (through softmax_out_layout).
  static constexpr uint8_t kOwnedAttrCap = 16;

  CatalogRowBuilder& AddAttr(
      const char* name,
      GeAttrType type,
      int64_t literal,
      const char* text,
      bool required,
      GeAttrDerived derived) {
    if (owned_count_ >= kOwnedAttrCap) {
      overflowed_ = true;
      return *this;
    }
    owned_attrs_[owned_count_] = GeAttr{name, type, literal, text, required, derived};
    ++owned_count_;
    entry_.attrs = owned_attrs_;
    entry_.attr_count = owned_count_;
    return *this;
  }

  void Adopt(const CatalogRowBuilder& other) {
    owned_count_ = other.owned_count_;
    std::copy_n(other.owned_attrs_, static_cast<size_t>(owned_count_), owned_attrs_);
    owned_ref_count_ = other.owned_ref_count_;
    std::copy_n(other.owned_refs_, static_cast<size_t>(owned_ref_count_), owned_refs_);
    overflowed_ = other.overflowed_;
    entry_ = other.entry_;
    if (other.entry_.attrs == other.owned_attrs_) {
      entry_.attrs = owned_attrs_;
    }
    if (other.entry_.output_refs == other.owned_refs_) {
      entry_.output_refs = owned_refs_;
    }
  }

  OpCatalogEntry entry_{};
  GeAttr owned_attrs_[kOwnedAttrCap]{};
  uint8_t owned_count_ = 0;
  // One slot per output that writes a collected input.
  static constexpr uint8_t kOwnedRefCap = 8;
  OutputRef owned_refs_[kOwnedRefCap]{};
  uint8_t owned_ref_count_ = 0;
  bool overflowed_ = false;
};

// Resolve a compute node to its catalog row (nullptr if unknown / custom_call / arity mismatch).
const OpCatalogEntry* LookupOpCatalog(const fxrt::ir::NodePtr& node);

// Logical names the converter lowers: catalog rows (builtin and Python overlay) plus the
// non-catalog names dedicated lowerings declare. The op-support hook keeps this set and
// demotes every other call to custom_call.
std::vector<std::string> ListConverterSupportedLogicalOps();

size_t CatalogCsvCount(const char* csv);
std::string CatalogCsvField(const char* csv, size_t index);

// Optional tail, described once and then bound to the collected input count.
struct AiCoreOptionalLayout {
  const char* pad_names = nullptr;
  size_t required = 0;
  size_t optional_total = 0;
  size_t pad = 0;
  bool has_optional = false;
};

AiCoreOptionalLayout DescribeAiCoreOptionalInputs(const OpCatalogEntry& entry);
bool BindAiCoreOptionalConnected(AiCoreOptionalLayout* layout, size_t connected, const char* op_name, std::string* err);
// OUTPUT count: 1 when opp_output_names is empty, else the CSV length (TopKV2=2).
size_t CatalogOutputArity(const OpCatalogEntry& entry);
// Collected input written by this output, or -1 when the output allocates a fresh buffer.
int32_t CatalogRefInput(const OpCatalogEntry& entry, size_t output_index);
// Shape-list IR name: names[shape_input_index], or "shape".
std::string CatalogShapeInputName(const char* opp_input_names, int32_t shape_input_index);
std::string CatalogShapeInputName(const OpCatalogEntry& entry);

// Logical names Convert can lower: the runtime catalog plus dedicated OpLowering
// targets. FX->IR uses this so unsupported ops become custom_call / python_call.
std::vector<std::string> ListConverterSupportedLogicalOps();

} // namespace fx_converter
