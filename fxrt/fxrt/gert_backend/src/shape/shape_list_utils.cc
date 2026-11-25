#include "shape/shape_list_utils.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "shape/fx_sym_expr_parser.h"

namespace fx_converter {

namespace {

// One element: concrete Int (kept verbatim, literal -1 included) or Symbol (-> kSymbolicDim).
int64_t ReadElement(const fxrt::ir::ValuePtr &element) {
  if (element == nullptr) {
    return kSymbolicDim;
  }
  if (element->IsInt()) {
    return element->ToInt();
  }
  return kSymbolicDim;
}

}  // namespace

std::vector<int64_t> ReadIntListSymbolSafe(const fxrt::ir::TuplePtr &tuple) {
  std::vector<int64_t> dims;
  if (tuple == nullptr) {
    return dims;
  }
  dims.resize(tuple->Size());
  std::transform(tuple->begin(), tuple->end(), dims.begin(), ReadElement);
  return dims;
}

bool ReadIntListWithSymbols(const fxrt::ir::TuplePtr &tuple, FxSymbolEnv &env, std::vector<int64_t> *dims,
                            std::vector<fxrt::ir::SymbolicExprPtr> *exprs) {
  if (tuple == nullptr || dims == nullptr || exprs == nullptr) {
    return false;
  }
  dims->clear();
  exprs->clear();
  dims->reserve(tuple->Size());
  exprs->reserve(tuple->Size());
  for (const auto &element : *tuple) {
    dims->push_back(ReadElement(element));
    if (element == nullptr || !element->IsSymbol()) {
      exprs->push_back(nullptr);
      continue;
    }
    // The bridge flattens an expression into a SymbolicVar named after its ToString(); this
    // rebuilds the tree and interns its free vars into env.
    auto expr = ParseFxSymExpr(element->ToSymbol(), env);
    if (expr == nullptr) {
      return false;
    }
    // Fold a constant (e.g. a Dynamo-specialized SymInt scalar like Sym(0) / Sym(-1)) back to
    // a plain number: it is a value the graph wrote, not a dim to evaluate per Execute. The
    // runner's positive-dim check only makes sense for real symbols -- a reduce axes payload
    // legitimately holds 0 / -1, and rejecting it there killed every dim=0 / dim=-1 compile.
    if (expr->GetKind() == fxrt::ir::SymbolicExpr::Kind::Constant) {
      dims->back() = expr->Evaluate();
      exprs->push_back(nullptr);
      continue;
    }
    exprs->push_back(std::move(expr));
  }
  return true;
}

bool SymbolicShapeExprsOfNode(const fxrt::ir::NodePtr &node, FxSymbolEnv &env,
                              std::vector<fxrt::ir::SymbolicExprPtr> *exprs) {
  if (exprs == nullptr) {
    return false;
  }
  exprs->clear();
  if (node == nullptr || node->output == nullptr || !node->output->IsTensor()) {
    return false;
  }
  const auto &tensor = node->output->ToTensor();
  if (tensor == nullptr || !tensor->HasSymbolicShape()) {
    return false;
  }
  const auto &syms = tensor->GetSymbolicShape();
  exprs->reserve(syms.size());
  for (const auto &dim : syms) {
    if (dim == nullptr || dim->GetKind() == fxrt::ir::SymbolicExpr::Kind::Constant) {
      exprs->push_back(nullptr);
      continue;
    }
    exprs->push_back(ParseFxSymExpr(dim, env));
  }
  return true;
}

bool IntListHasSymbol(const fxrt::ir::TuplePtr &tuple) {
  if (tuple == nullptr) {
    return false;
  }
  return std::any_of(tuple->begin(), tuple->end(),
                     [](const fxrt::ir::ValuePtr &element) { return element != nullptr && element->IsSymbol(); });
}

bool ResolveKeepDimShapeList(std::vector<int64_t> *size, const std::vector<int64_t> &in_shape) {
  if (size == nullptr || size->empty() || in_shape.size() > size->size()) {
    return false;
  }
  std::vector<int64_t> aligned(size->size(), 1);
  if (!in_shape.empty()) {
    std::copy(in_shape.begin(), in_shape.end(), aligned.begin() + (size->size() - in_shape.size()));
  }
  bool all_resolved = true;
  for (size_t i = 0; i < size->size(); ++i) {
    if ((*size)[i] == -1 || (*size)[i] == std::numeric_limits<int64_t>::min()) {
      if (aligned[i] > 0) {
        (*size)[i] = aligned[i];
      } else {
        all_resolved = false;
      }
    }
    if ((*size)[i] != -1 && (*size)[i] != std::numeric_limits<int64_t>::min() && (*size)[i] <= 0) {
      return false;
    }
  }
  return all_resolved;
}

bool ResolveKeepDimShapeListStrict(std::vector<int64_t> *size, const std::vector<int64_t> &in_shape) {
  if (size == nullptr || size->empty() || in_shape.size() > size->size()) {
    return false;
  }
  std::vector<int64_t> aligned(size->size(), 1);
  if (!in_shape.empty()) {
    std::copy(in_shape.begin(), in_shape.end(), aligned.begin() + (size->size() - in_shape.size()));
  }
  for (size_t i = 0; i < size->size(); ++i) {
    if ((*size)[i] == -1 || (*size)[i] == std::numeric_limits<int64_t>::min()) {
      if (aligned[i] <= 0) {
        return false;
      }
      (*size)[i] = aligned[i];
    }
    if ((*size)[i] <= 0) {
      return false;
    }
  }
  return true;
}

uint16_t Float32ToFp16Bits(float value) {
  uint32_t bits;
  static_assert(sizeof(bits) == sizeof(value), "float must be 32-bit");
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 31) & 0x1U;
  const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127;
  const uint32_t mant = bits & 0x7FFFFFU;
  uint16_t out;
  if (exp >= 16) {  // overflow or inf/nan
    out = (sign << 15) | 0x7C00U;
  } else if (exp > -15) {  // normalized
    const uint32_t biased_exp = static_cast<uint32_t>(exp + 15);
    out = static_cast<uint16_t>((sign << 15) | (biased_exp << 10) | (mant >> 13));
  } else if (exp > -25) {  // denormal
    const uint32_t shifted = (mant | 0x800000U) >> (static_cast<size_t>(-1 - exp));
    out = static_cast<uint16_t>((sign << 15) | shifted);
  } else {  // underflow to zero
    out = static_cast<uint16_t>(sign << 15);
  }
  return out;
}

uint16_t Float32ToBf16Bits(float value) {
  uint32_t bits;
  static_assert(sizeof(bits) == sizeof(value), "float must be 32-bit");
  std::memcpy(&bits, &value, sizeof(bits));
  // BF16 is the upper 16 bits of a float32 (truncating the lower mantissa).
  return static_cast<uint16_t>((bits >> 16) & 0xFFFFU);
}

}  // namespace fx_converter
