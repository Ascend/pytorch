/**
 * Contiguous empty/empty_strided → AllocMemory + BuildTensor (uninitialized device buffer).
 * Symbolic size + concrete contiguous strides are accepted after const-product fold
 * in IsSymbolicallyContiguous (e.g. (1*128)*50 ≡ 6400). Non-contiguous strides are
 * rejected clearly (no Path B / InstallFx*). Inputs keep the ATen order [size, stride, ...].
 */

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "lowering/op_lowering.h"
#include "shape/shape_list_utils.h"

#include "exe_graph/execute_graph_builder.h"
#include "exe_graph/graph_builder_utils.h"
#include "exe_graph/graph_skeleton_builder.h"
#include "exe_graph/runtime/storage_shape.h"
#include "ir/symbolic/symbolic.h"
#include "ir/value/value.h"
#include "lowering/common/view_utils.h"
#include "lowering/common/wiring.h"
#include "register/op_catalog.h"
#include "lowering/op_lowering_registry.h"
#include "mrt_ge/mrt_node_utils.h"
#include "ops/op_def/ops_name.h"
#include "shape/fx_expr_infer_shape.h"
#include "shape/fx_sym_expr_parser.h"
#include "shape/mrt_to_ge_expr.h"

namespace fx_converter {

// After ViewAlias (150), before IdentityAlias (200).
constexpr int kEmptyStridedPriority = 160;

namespace {

gert::OpImplRegisterV2::InferShapeKernelFunc g_empty_infer_func = nullptr;

bool ContiguousRowMajorStrides(const std::vector<int64_t> &size, const std::vector<int64_t> &stride) {
  if (size.size() != stride.size()) {
    return false;
  }
  int64_t expected = 1;
  for (int64_t i = static_cast<int64_t>(size.size()) - 1; i >= 0; --i) {
    const size_t ui = static_cast<size_t>(i);
    if (size[ui] < 0 || stride[ui] < 0) {
      return false;
    }
    if (size[ui] == 0) {
      return true;
    }
    if (size[ui] != 1 && stride[ui] != expected) {
      return false;
    }
    expected *= size[ui];
  }
  return true;
}

bool ParseEmptyStridedInputs(const fxrt::ir::NodePtr &node, std::vector<int64_t> *size, std::vector<int64_t> *stride,
                             std::string *err) {
  if (node == nullptr || size == nullptr || stride == nullptr) {
    return false;
  }
  if (node->op == fxrt::ops::Op_empty) {
    if (node->inputs.empty() || node->inputs[0] == nullptr || node->inputs[0]->output == nullptr ||
        !node->inputs[0]->output->IsTuple()) {
      if (err != nullptr) *err = "empty lowering: expected ordered size tuple";
      return false;
    }
    *size = ReadIntListSymbolSafe(node->inputs[0]->output->ToTuple());
    return true;
  }
  if (node->inputs.size() < 2U || node->inputs[0] == nullptr || node->inputs[1] == nullptr ||
      node->inputs[0]->output == nullptr || node->inputs[1]->output == nullptr ||
      !node->inputs[0]->output->IsTuple() || !node->inputs[1]->output->IsTuple()) {
    if (err != nullptr) {
      *err = "empty-strided lowering: expected ordered size and stride tuple inputs";
    }
    return false;
  }
  *size = ReadIntListSymbolSafe(node->inputs[0]->output->ToTuple());
  *stride = ReadIntListSymbolSafe(node->inputs[1]->output->ToTuple());
  if (size->size() != stride->size()) {
    if (err != nullptr) {
      *err = "empty-strided lowering: size and stride ranks differ";
    }
    return false;
  }
  return true;
}

fxrt::ir::SymbolicExprPtr AsSymbolicInt(const fxrt::ir::Value *value, FxSymbolEnv &env) {
  if (value == nullptr) {
    return nullptr;
  }
  if (value->IsInt()) {
    return fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicConst>(value->ToInt());
  }
  // The ABI bridge serializes derived SymInts, including constants, as a
  // SymbolicVar name. Reuse the shape parser before comparing expression trees.
  return value->IsSymbol() ? ParseFxSymExpr(value->ToSymbol(), env) : nullptr;
}

bool IsConst(const fxrt::ir::SymbolicExprPtr &expr, int64_t value) {
  const auto *constant = dynamic_cast<const fxrt::ir::SymbolicConst *>(expr.get());
  return constant != nullptr && constant->GetValue() == value;
}

bool EquivalentExpr(const fxrt::ir::SymbolicExprPtr &lhs, const fxrt::ir::SymbolicExprPtr &rhs);

bool EquivalentUnitMul(const fxrt::ir::SymbolicExprPtr &product, const fxrt::ir::SymbolicExprPtr &other) {
  if (product == nullptr || product->GetKind() != fxrt::ir::SymbolicExpr::Kind::Mul) {
    return false;
  }
  const auto *binary = dynamic_cast<const fxrt::ir::SymbolicBinaryOp *>(product.get());
  return binary != nullptr &&
         ((IsConst(binary->getLHS(), 1) && EquivalentExpr(binary->getRHS(), other)) ||
          (IsConst(binary->getRHS(), 1) && EquivalentExpr(binary->getLHS(), other)));
}

bool EquivalentLeaf(const fxrt::ir::SymbolicExprPtr &lhs, const fxrt::ir::SymbolicExprPtr &rhs) {
  if (const auto *lhs_const = dynamic_cast<const fxrt::ir::SymbolicConst *>(lhs.get())) {
    const auto *rhs_const = dynamic_cast<const fxrt::ir::SymbolicConst *>(rhs.get());
    return rhs_const != nullptr && lhs_const->GetValue() == rhs_const->GetValue();
  }
  if (const auto *lhs_var = dynamic_cast<const fxrt::ir::SymbolicVar *>(lhs.get())) {
    const auto *rhs_var = dynamic_cast<const fxrt::ir::SymbolicVar *>(rhs.get());
    return rhs_var != nullptr && lhs_var->GetName() == rhs_var->GetName();
  }
  return lhs->ToString() == rhs->ToString();
}

bool EquivalentBinary(const fxrt::ir::SymbolicExprPtr &lhs, const fxrt::ir::SymbolicExprPtr &rhs) {
  const auto *lhs_binary = dynamic_cast<const fxrt::ir::SymbolicBinaryOp *>(lhs.get());
  const auto *rhs_binary = dynamic_cast<const fxrt::ir::SymbolicBinaryOp *>(rhs.get());
  if (lhs_binary == nullptr || rhs_binary == nullptr) {
    return EquivalentLeaf(lhs, rhs);
  }
  if (EquivalentExpr(lhs_binary->getLHS(), rhs_binary->getLHS()) &&
      EquivalentExpr(lhs_binary->getRHS(), rhs_binary->getRHS())) {
    return true;
  }
  const auto kind = lhs->GetKind();
  const bool commutative = kind == fxrt::ir::SymbolicExpr::Kind::Mul || kind == fxrt::ir::SymbolicExpr::Kind::Add;
  return commutative && EquivalentExpr(lhs_binary->getLHS(), rhs_binary->getRHS()) &&
         EquivalentExpr(lhs_binary->getRHS(), rhs_binary->getLHS());
}

bool TryEvalConst(const fxrt::ir::SymbolicExprPtr &expr, int64_t *out) {
  if (expr == nullptr || out == nullptr) {
    return false;
  }
  if (const auto *constant = dynamic_cast<const fxrt::ir::SymbolicConst *>(expr.get())) {
    *out = constant->GetValue();
    return true;
  }
  if (expr->GetKind() != fxrt::ir::SymbolicExpr::Kind::Mul &&
      expr->GetKind() != fxrt::ir::SymbolicExpr::Kind::Add) {
    return false;
  }
  const auto *binary = dynamic_cast<const fxrt::ir::SymbolicBinaryOp *>(expr.get());
  if (binary == nullptr) {
    return false;
  }
  int64_t lhs_val = 0;
  int64_t rhs_val = 0;
  if (!TryEvalConst(binary->getLHS(), &lhs_val) || !TryEvalConst(binary->getRHS(), &rhs_val)) {
    return false;
  }
  *out = (expr->GetKind() == fxrt::ir::SymbolicExpr::Kind::Mul) ? (lhs_val * rhs_val) : (lhs_val + rhs_val);
  return true;
}

fxrt::ir::SymbolicExprPtr FoldMul(const fxrt::ir::SymbolicExprPtr &lhs, const fxrt::ir::SymbolicExprPtr &rhs) {
  int64_t lhs_val = 0;
  int64_t rhs_val = 0;
  if (TryEvalConst(lhs, &lhs_val) && TryEvalConst(rhs, &rhs_val)) {
    return fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicConst>(lhs_val * rhs_val);
  }
  return lhs * rhs;
}

bool EquivalentExpr(const fxrt::ir::SymbolicExprPtr &lhs, const fxrt::ir::SymbolicExprPtr &rhs) {
  if (lhs == nullptr || rhs == nullptr) {
    return lhs == rhs;
  }
  int64_t lhs_val = 0;
  int64_t rhs_val = 0;
  if (TryEvalConst(lhs, &lhs_val) && TryEvalConst(rhs, &rhs_val)) {
    return lhs_val == rhs_val;
  }
  if (lhs->GetKind() != rhs->GetKind()) {
    return EquivalentUnitMul(lhs, rhs) || EquivalentUnitMul(rhs, lhs);
  }
  return EquivalentBinary(lhs, rhs);
}

bool TupleContainsSymbol(const fxrt::ir::NodePtr &node) {
  if (node == nullptr || node->output == nullptr || !node->output->IsTuple()) {
    return false;
  }
  const auto tuple = node->output->ToTuple();
  return tuple != nullptr && std::any_of(tuple->begin(), tuple->end(), [](const auto &value) {
           return value != nullptr && value->IsSymbol();
         });
}

bool IsSymbolicallyContiguous(const fxrt::ir::NodePtr &size_node, const fxrt::ir::NodePtr &stride_node,
                              FxSymbolEnv &env, std::string *err) {
  if (size_node == nullptr || stride_node == nullptr || size_node->output == nullptr ||
      stride_node->output == nullptr || !size_node->output->IsTuple() || !stride_node->output->IsTuple()) {
    return false;
  }
  const auto sizes = size_node->output->ToTuple();
  const auto strides = stride_node->output->ToTuple();
  if (sizes == nullptr || strides == nullptr || sizes->Size() != strides->Size()) {
    return false;
  }
  fxrt::ir::SymbolicExprPtr expected = fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicConst>(1);
  for (size_t i = sizes->Size(); i > 0; --i) {
    const auto size = AsSymbolicInt((*sizes)[i - 1U].get(), env);
    const auto stride = AsSymbolicInt((*strides)[i - 1U].get(), env);
    if (size == nullptr || stride == nullptr) {
      return false;
    }
    const auto *size_const = dynamic_cast<const fxrt::ir::SymbolicConst *>(size.get());
    if (size_const != nullptr) {
      if (size_const->GetValue() < 0) {
        return false;
      }
      if (size_const->GetValue() == 0) {
        return true;
      }
      if (size_const->GetValue() == 1) {
        continue;
      }
    }
    if (!EquivalentExpr(stride, expected)) {
      if (err != nullptr) {
        *err = "empty-strided lowering: symbolic stride is not provably contiguous at axis " +
               std::to_string(i - 1U) + ": actual=" + stride->ToString() + ", expected=" + expected->ToString();
      }
      return false;
    }
    // Fold const products so expected (1*128)*50 matches concrete stride 6400.
    expected = FoldMul(expected, size);
  }
  return true;
}

bool LowerSymbolicEmptyStrided(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &ctx,
                               LoweringResult *out, std::string *err) {
  auto rule = BuildOutputSymbolicShapeRule(node, ctx.symbol_env, "empty_strided");
  if (rule == nullptr) {
    if (err != nullptr) {
      *err = "empty-strided lowering: no symbolic output shape rule";
    }
    return false;
  }

  const auto out_shape = ShapeOfNode(node);
  const auto out_dtype = DtypeOfNode(node);
  const ge::TensorDesc empty_td;
  const auto td_out = MakeNdTensorDesc(out_shape, out_dtype);
  const FxShapeRule *rule_ptr = rule.get();
  ctx.built->shape_rules.push_back(rule);
  auto *c_rule = AddConstNode(ctx.main, "const_empty_shape_rule_" + tag, &rule_ptr, sizeof(rule_ptr), false,
                              *ctx.sym_gen, FxrtConstType::kVoidPtr);
  auto infer_func = GetFxExprInferShapeFunc();
  auto *c_infer = AddConstNode(ctx.main, "const_empty_infer_func_" + tag, &infer_func, sizeof(infer_func), false,
                               *ctx.sym_gen, FxrtConstType::kKernelFuncPtr);
  uint64_t shape_symbol = 0;
  auto *infer = AddInferShapeNode(ctx.main, tag, {}, td_out, empty_td, c_rule, c_infer, *ctx.sym_gen, &shape_symbol);
  if (ctx.bind_symbols != nullptr) {
    fx_converter::AddDataEdge(ctx.main, ctx.bind_symbols, -1, infer, -1);
  }
  ctx.kernel_nodes->insert(ctx.kernel_nodes->end(), {c_rule, c_infer, infer});

  fx_converter::Node *output_tensor = nullptr;
  uint64_t output_tensor_symbol = 0;
  out->outputs.resize(1);
  TensorRef &out0 = out->outputs[0];
  if (IsFinalOutput(ctx)) {
    output_tensor = AddOutputDataNode(ctx.main, tag, td_out, *ctx.sym_gen, &output_tensor_symbol);
    ctx.kernel_nodes->push_back(output_tensor);
  } else {
    auto dtype_blob = std::make_shared<std::vector<uint8_t>>(sizeof(ge::DataType));
    std::memcpy(dtype_blob->data(), &out_dtype, sizeof(out_dtype));
    ctx.built->binary_bindata_blobs.push_back(dtype_blob);
    auto *c_dtype = AddConstNode(ctx.main, "const_empty_dtype_" + tag, dtype_blob->data(), dtype_blob->size(),
                                 false, *ctx.sym_gen, FxrtConstType::kGeDataType);
    uint64_t size_symbol = 0;
    const CalcTensorSizeInputs size_inputs{
        {c_dtype, 0, c_dtype->GetExtendInfo()->GetOutputSymbol(0)},
        {infer, 0, shape_symbol},
    };
    auto *calc = AddCalcTensorSizeNode(ctx.main, tag, size_inputs, *ctx.sym_gen, &size_symbol);
    uint64_t addr_symbol = 0;
    auto *alloc = AddAllocMemoryNode(ctx.main, tag, empty_td, ctx.rt->s_allocator, ctx.rt->select_l2, size_symbol,
                                     calc, *ctx.sym_gen, &addr_symbol);
    const BuildTensorInputs build_inputs{{infer, 0, shape_symbol}, {alloc, 0, addr_symbol}};
    output_tensor = AddBuildTensorNode(ctx.main, tag, td_out, build_inputs, out_dtype, ge::FORMAT_ND, *ctx.sym_gen,
                                       &output_tensor_symbol, ctx.kernel_nodes);
    ctx.kernel_nodes->insert(ctx.kernel_nodes->end(), {c_dtype, calc, alloc, output_tensor});
    out->mem_to_frees.push_back(alloc);
    out0.unpacked.shape = TensorEndpoint{infer, 0, shape_symbol};
    out0.unpacked.addr = TensorEndpoint{alloc, 0, addr_symbol};
    out0.allocation = out0.unpacked.addr;
  }

  out0.producer = output_tensor;
  out0.symbol = output_tensor_symbol;
  out0.shape = out_shape;
  out0.dtype = out_dtype;
  out->launch = nullptr;
  out->prepare = nullptr;
  out->skip_cni = false;
  out->cni.op_type = "empty_strided";
  out->cni.cni_io = CniIoStyle::kIndexedFrom0;
  out->cni.behavior = nullptr;
  out->cni.output_shape = out_shape;
  out->cni.output_dtype = static_cast<int32_t>(out_dtype);
  out->cni.compute_nodes = {infer};
  return true;
}

}  // namespace

class EmptyStridedOpLowering : public OpLowering {
 public:
  const char *Name() const override { return "EmptyStridedAlloc"; }

  bool Match(const fxrt::ir::NodePtr &node, const LoweringContext *) const override {
    return node != nullptr && (node->op == fxrt::ops::Op_empty_strided || node->op == fxrt::ops::Op_empty);
  }

  std::vector<std::string> SupportedLogicalOps() const override { return {"empty_strided", "empty"}; }

  static bool LowerConcreteEmptyStrided(const fxrt::ir::NodePtr &node, const std::string &tag,
                                        LoweringContext &ctx, LoweringResult *out, std::string *err) {
    std::vector<int64_t> size;
    std::vector<int64_t> stride;
    if (!ParseEmptyStridedInputs(node, &size, &stride, err)) {
      return false;
    }
    if (node->op != fxrt::ops::Op_empty && !ContiguousRowMajorStrides(size, stride)) {
      if (err != nullptr) {
        *err = "empty-strided lowering: non-contiguous stride unsupported (refuse silent alloc)";
      }
      return false;
    }

    ge::Graph *main = ctx.main;
    SharedRuntime &rt = *ctx.rt;
    uint64_t &sym_gen = *ctx.sym_gen;
    const ge::TensorDesc empty_td;
    // Prefer packed size (concrete) over ShapeOfNode (may be dynamic from mul consumers).
    const std::vector<int64_t> out_shape = size;
    const ge::DataType out_dtype = DtypeOfNode(node);
    const ge::TensorDesc td_o = MakeNdTensorDesc(out_shape, out_dtype);

    // empty_strided has no tensor inputs: always use Const StorageShape (FxExprInfer
    // requires >=1 tensor + rule + func).
    fx_converter::Node *shape_producer = nullptr;
    uint64_t s_shape = 0;
    if (!AddConcreteStorageShapeConst(main, ctx.built, tag, out_shape, sym_gen, ctx.kernel_nodes, &shape_producer,
                                      &s_shape, err)) {
      if (err != nullptr && err->empty()) {
        *err = "empty-strided lowering: concrete StorageShape failed";
      }
      return false;
    }
    fx_converter::Node *infer = shape_producer;
    (void)g_empty_infer_func;

    // Dtype Const for CalcTensorSize (SharedRuntime dtype may be float-only).
    auto dtype_blob = std::make_shared<std::vector<uint8_t>>(sizeof(ge::DataType));
    *reinterpret_cast<ge::DataType *>(dtype_blob->data()) = out_dtype;
    ctx.built->binary_bindata_blobs.push_back(dtype_blob);
    fx_converter::Node *c_dtype =
        AddConstNode(main, "const_empty_dtype_" + tag, dtype_blob->data(), dtype_blob->size(), false, sym_gen,
                     FxrtConstType::kGeDataType);
    ctx.kernel_nodes->push_back(c_dtype);
    const uint64_t s_dtype = c_dtype->GetExtendInfo()->GetOutputSymbol(0);

    uint64_t s_size = 0;
    const CalcTensorSizeInputs size_inputs{{c_dtype, 0, s_dtype}, {infer, 0, s_shape}};
    fx_converter::Node *calc = AddCalcTensorSizeNode(main, tag, size_inputs, sym_gen, &s_size);
    ctx.kernel_nodes->push_back(calc);

    uint64_t s_addr = 0;
    fx_converter::Node *alloc =
        AddAllocMemoryNode(main, tag, empty_td, rt.s_allocator, rt.select_l2, s_size, calc, sym_gen, &s_addr);
    ctx.kernel_nodes->push_back(alloc);

    uint64_t s_out_tensor = 0;
    const BuildTensorInputs build_inputs{{infer, 0, s_shape}, {alloc, 0, s_addr}};
    fx_converter::Node *build = AddBuildTensorNode(main, tag, td_o, build_inputs, out_dtype, ge::FORMAT_ND, sym_gen,
                                                   &s_out_tensor, ctx.kernel_nodes);
    ctx.kernel_nodes->push_back(build);

    out->outputs.resize(1);
    TensorRef &out0 = out->outputs[0];
    out0.producer = build;
    out0.symbol = s_out_tensor;
    out0.shape = out_shape;
    out0.dtype = out_dtype;
    out0.unpacked.shape = TensorEndpoint{infer, 0, s_shape};
    out0.unpacked.addr = TensorEndpoint{alloc, 0, s_addr};
    out0.allocation = out0.unpacked.addr;
    out->launch = nullptr;
    out->prepare = nullptr;
    // Keep buffer until a later FreeMemory after consumer Launch / NetOutput.
    out->mem_to_frees.push_back(alloc);
    out->skip_cni = false;
    out->cni.op_type = "Reshape";
    out->cni.cni_io = CniIoStyle::kIndexedFrom0;
    out->cni.behavior = nullptr;
    out->cni.input_shapes = {};
    out->cni.input_dtypes = {};
    out->cni.output_shape = out_shape;
    out->cni.output_dtype = static_cast<int32_t>(out_dtype);
    out->cni.compute_nodes = {infer, calc, alloc, build};
    return true;
  }

  bool Lower(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &ctx, LoweringResult *out,
             std::string *err) override {
    if (out == nullptr || !ValidateLoweringContext(ctx, false, "empty-strided lowering", err)) {
      return false;
    }
    const auto &inputs = node->inputs;
    // empty has an implicit contiguous layout; its dimensions can still change
    // on every execution. Reuse the allocation path without inventing strides.
    if (node->op == fxrt::ops::Op_empty && !inputs.empty() && TupleContainsSymbol(inputs[0])) {
      return LowerSymbolicEmptyStrided(node, tag, ctx, out, err);
    }
    if (inputs.size() >= 2U &&
        (TupleContainsSymbol(inputs[0]) || TupleContainsSymbol(inputs[1]))) {
      if (ctx.symbol_env == nullptr || !IsSymbolicallyContiguous(inputs[0], inputs[1], *ctx.symbol_env, err)) {
        if (err != nullptr && err->empty()) {
          *err = "empty-strided lowering: symbolic stride is not provably contiguous";
        }
        return false;
      }
      return LowerSymbolicEmptyStrided(node, tag, ctx, out, err);
    }
    return LowerConcreteEmptyStrided(node, tag, ctx, out, err);
  }
};

}  // namespace fx_converter

FX_REGISTER_OP_LOWERING(::fx_converter::kEmptyStridedPriority,
                        std::make_unique<::fx_converter::EmptyStridedOpLowering>());
