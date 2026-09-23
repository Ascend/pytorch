#include <cmath>
#include <algorithm>

#include "common/common.h"
#include "ir/symbolic/symbolic.h"

namespace fxrt {
namespace ir {

int64_t SymbolicVar::Evaluate() const {
  if (!hasValue_) {
    RT_GLOG(EXCEPTION) << "Symbolic variable " << name_ << " has no value.";
  }
  return value_;
}

int64_t SymbolicFloorDiv::Evaluate() const {
  auto lhsVal = lhs_->Evaluate();
  auto rhsVal = rhs_->Evaluate();
  if (rhsVal == 0) {
    RT_GLOG(EXCEPTION) << "Division by zero in symbolic expression.";
  }
  return static_cast<int64_t>(std::floor(static_cast<double>(lhsVal) / static_cast<double>(rhsVal)));
}

int64_t SymbolicCeilDiv::Evaluate() const {
  auto lhsVal = lhs_->Evaluate();
  auto rhsVal = rhs_->Evaluate();
  if (rhsVal == 0) {
    RT_GLOG(EXCEPTION) << "Division by zero in symbolic expression.";
  }
  return static_cast<int64_t>(std::ceil(static_cast<double>(lhsVal) / static_cast<double>(rhsVal)));
}

int64_t SymbolicTrueDiv::Evaluate() const {
  auto lhsVal = lhs_->Evaluate();
  auto rhsVal = rhs_->Evaluate();
  if (rhsVal == 0) {
    RT_GLOG(EXCEPTION) << "Division by zero in symbolic expression.";
  }
  return static_cast<int64_t>(static_cast<double>(lhsVal) / static_cast<double>(rhsVal));
}

int64_t SymbolicMod::Evaluate() const {
  auto lhsVal = lhs_->Evaluate();
  auto rhsVal = rhs_->Evaluate();
  if (rhsVal == 0) {
    RT_GLOG(EXCEPTION) << "Modulo by zero in symbolic expression.";
  }
  return lhsVal % rhsVal;
}

int64_t SymbolicMin::Evaluate() const {
  auto lhsVal = lhs_->Evaluate();
  auto rhsVal = rhs_->Evaluate();
  return std::min(lhsVal, rhsVal);
}

SymbolicExprPtr operator+(SymbolicExprPtr lhs, SymbolicExprPtr rhs) {
  return MakeIntrusive<SymbolicAdd>(lhs, rhs);
}

SymbolicExprPtr operator*(SymbolicExprPtr lhs, SymbolicExprPtr rhs) {
  return MakeIntrusive<SymbolicMul>(lhs, rhs);
}

SymbolicExprPtr operator/(SymbolicExprPtr lhs, SymbolicExprPtr rhs) {
  return MakeIntrusive<SymbolicTrueDiv>(lhs, rhs);
}

SymbolicExprPtr operator%(SymbolicExprPtr lhs, SymbolicExprPtr rhs) {
  return MakeIntrusive<SymbolicMod>(lhs, rhs);
}

SymbolicExprPtr FloorDiv(SymbolicExprPtr lhs, SymbolicExprPtr rhs) {
  return MakeIntrusive<SymbolicFloorDiv>(lhs, rhs);
}

SymbolicExprPtr CeilDiv(SymbolicExprPtr lhs, SymbolicExprPtr rhs) {
  return MakeIntrusive<SymbolicCeilDiv>(lhs, rhs);
}

SymbolicExprPtr Min(SymbolicExprPtr lhs, SymbolicExprPtr rhs) {
  return MakeIntrusive<SymbolicMin>(lhs, rhs);
}

SymbolicExprPtr SymbolicVar::DeepCopy() const {
  auto copy = MakeIntrusive<SymbolicVar>(name_);
  if (hasValue_) {
    copy->SetValue(value_);
  }
  return copy;
}

} // namespace ir
} // namespace fxrt
