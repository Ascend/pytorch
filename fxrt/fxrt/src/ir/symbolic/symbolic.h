#ifndef __IR_SYMBOLIC_SYMBOLIC_H__
#define __IR_SYMBOLIC_SYMBOLIC_H__

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include "ir/common/intrusive_ptr.h"

namespace fxrt {
namespace ir {

class SymbolicExpr;
using SymbolicExprPtr = IntrusivePtr<SymbolicExpr>;

class SymbolicExpr : public RefCounted {
 public:
  enum class Kind {
    Constant,
    Variable,
    Add,
    Mul,
    TrueDiv,
    FloorDiv,
    CeilDiv,
    Mod,
    Min,
  };

  explicit SymbolicExpr(Kind kind) : kind_(kind) {}
  virtual ~SymbolicExpr() = default;
  Kind GetKind() const {
    return kind_;
  }
  virtual int64_t Evaluate() const = 0;
  virtual std::string ToString() const = 0;

  /**
   * @brief Creates a deep copy of this SymbolicExpr object.
   * @return A new SymbolicExpr object with copied data.
   */
  virtual SymbolicExprPtr DeepCopy() const = 0;

 private:
  Kind kind_;
};

class SymbolicConst : public SymbolicExpr {
 public:
  explicit SymbolicConst(int64_t value) : SymbolicExpr(Kind::Constant), value_(value) {}
  int64_t Evaluate() const override {
    return value_;
  }
  std::string ToString() const override {
    return std::to_string(value_);
  }
  int64_t GetValue() const {
    return value_;
  }
  SymbolicExprPtr DeepCopy() const override {
    return MakeIntrusive<SymbolicConst>(value_);
  }

  static bool classof(const SymbolicExpr* e) {
    return e->GetKind() == Kind::Constant;
  }

 private:
  int64_t value_;
};

class SymbolicVar : public SymbolicExpr {
 public:
  explicit SymbolicVar(const std::string& name) : SymbolicExpr(Kind::Variable), name_(name) {}
  int64_t Evaluate() const override;
  std::string ToString() const override {
    return name_;
  }
  void SetValue(int64_t value) {
    value_ = value;
    hasValue_ = true;
  }
  const std::string& GetName() const {
    return name_;
  }
  SymbolicExprPtr DeepCopy() const override;

  static bool classof(const SymbolicExpr* e) {
    return e->GetKind() == Kind::Variable;
  }

 private:
  std::string name_;
  bool hasValue_{false};
  int64_t value_{-1}; // for evaluation
};

class SymbolicBinaryOp : public SymbolicExpr {
 public:
  SymbolicBinaryOp(Kind kind, SymbolicExprPtr lhs, SymbolicExprPtr rhs) : SymbolicExpr(kind), lhs_(lhs), rhs_(rhs) {}

  SymbolicExprPtr getLHS() const {
    return lhs_;
  }
  SymbolicExprPtr getRHS() const {
    return rhs_;
  }

 protected:
  SymbolicExprPtr lhs_;
  SymbolicExprPtr rhs_;
};

class SymbolicAdd : public SymbolicBinaryOp {
 public:
  SymbolicAdd(SymbolicExprPtr lhs, SymbolicExprPtr rhs) : SymbolicBinaryOp(Kind::Add, lhs, rhs) {}
  int64_t Evaluate() const override {
    return lhs_->Evaluate() + rhs_->Evaluate();
  }
  std::string ToString() const override {
    return "(" + lhs_->ToString() + " + " + rhs_->ToString() + ")";
  }
  SymbolicExprPtr DeepCopy() const override {
    return MakeIntrusive<SymbolicAdd>(lhs_->DeepCopy(), rhs_->DeepCopy());
  }

  static bool classof(const SymbolicExpr* e) {
    return e->GetKind() == Kind::Add;
  }
};

class SymbolicMul : public SymbolicBinaryOp {
 public:
  SymbolicMul(SymbolicExprPtr lhs, SymbolicExprPtr rhs) : SymbolicBinaryOp(Kind::Mul, lhs, rhs) {}
  int64_t Evaluate() const override {
    return lhs_->Evaluate() * rhs_->Evaluate();
  }
  std::string ToString() const override {
    return "(" + lhs_->ToString() + " * " + rhs_->ToString() + ")";
  }
  SymbolicExprPtr DeepCopy() const override {
    return MakeIntrusive<SymbolicMul>(lhs_->DeepCopy(), rhs_->DeepCopy());
  }

  static bool classof(const SymbolicExpr* e) {
    return e->GetKind() == Kind::Mul;
  }
};

class SymbolicTrueDiv : public SymbolicBinaryOp {
 public:
  SymbolicTrueDiv(SymbolicExprPtr lhs, SymbolicExprPtr rhs) : SymbolicBinaryOp(Kind::TrueDiv, lhs, rhs) {}
  int64_t Evaluate() const override;
  std::string ToString() const override {
    return "(" + lhs_->ToString() + " / " + rhs_->ToString() + ")";
  }
  SymbolicExprPtr DeepCopy() const override {
    return MakeIntrusive<SymbolicTrueDiv>(lhs_->DeepCopy(), rhs_->DeepCopy());
  }

  static bool classof(const SymbolicExpr* e) {
    return e->GetKind() == Kind::TrueDiv;
  }
};

class SymbolicFloorDiv : public SymbolicBinaryOp {
 public:
  SymbolicFloorDiv(SymbolicExprPtr lhs, SymbolicExprPtr rhs) : SymbolicBinaryOp(Kind::FloorDiv, lhs, rhs) {}
  int64_t Evaluate() const override;
  std::string ToString() const override {
    return "floor_div(" + lhs_->ToString() + ", " + rhs_->ToString() + ")";
  }
  SymbolicExprPtr DeepCopy() const override {
    return MakeIntrusive<SymbolicFloorDiv>(lhs_->DeepCopy(), rhs_->DeepCopy());
  }

  static bool classof(const SymbolicExpr* e) {
    return e->GetKind() == Kind::FloorDiv;
  }
};

class SymbolicCeilDiv : public SymbolicBinaryOp {
 public:
  SymbolicCeilDiv(SymbolicExprPtr lhs, SymbolicExprPtr rhs) : SymbolicBinaryOp(Kind::CeilDiv, lhs, rhs) {}
  int64_t Evaluate() const override;
  std::string ToString() const override {
    return "ceil_div(" + lhs_->ToString() + ", " + rhs_->ToString() + ")";
  }
  SymbolicExprPtr DeepCopy() const override {
    return MakeIntrusive<SymbolicCeilDiv>(lhs_->DeepCopy(), rhs_->DeepCopy());
  }

  static bool classof(const SymbolicExpr* e) {
    return e->GetKind() == Kind::CeilDiv;
  }
};

class SymbolicMod : public SymbolicBinaryOp {
 public:
  SymbolicMod(SymbolicExprPtr lhs, SymbolicExprPtr rhs) : SymbolicBinaryOp(Kind::Mod, lhs, rhs) {}
  int64_t Evaluate() const override;
  std::string ToString() const override {
    return "(" + lhs_->ToString() + " % " + rhs_->ToString() + ")";
  }
  SymbolicExprPtr DeepCopy() const override {
    return MakeIntrusive<SymbolicMod>(lhs_->DeepCopy(), rhs_->DeepCopy());
  }

  static bool classof(const SymbolicExpr* e) {
    return e->GetKind() == Kind::Mod;
  }
};

class SymbolicMin : public SymbolicBinaryOp {
 public:
  SymbolicMin(SymbolicExprPtr lhs, SymbolicExprPtr rhs) : SymbolicBinaryOp(Kind::Min, lhs, rhs) {}
  int64_t Evaluate() const override;
  std::string ToString() const override {
    return "min(" + lhs_->ToString() + ", " + rhs_->ToString() + ")";
  }
  SymbolicExprPtr DeepCopy() const override {
    return MakeIntrusive<SymbolicMin>(lhs_->DeepCopy(), rhs_->DeepCopy());
  }

  static bool classof(const SymbolicExpr* e) {
    return e->GetKind() == Kind::Min;
  }
};

// A helper to create symbolic expressions
SymbolicExprPtr operator+(SymbolicExprPtr lhs, SymbolicExprPtr rhs);
SymbolicExprPtr operator*(SymbolicExprPtr lhs, SymbolicExprPtr rhs);
SymbolicExprPtr operator/(SymbolicExprPtr lhs, SymbolicExprPtr rhs);
SymbolicExprPtr operator%(SymbolicExprPtr lhs, SymbolicExprPtr rhs);
SymbolicExprPtr FloorDiv(SymbolicExprPtr lhs, SymbolicExprPtr rhs);
SymbolicExprPtr CeilDiv(SymbolicExprPtr lhs, SymbolicExprPtr rhs);
SymbolicExprPtr Min(SymbolicExprPtr lhs, SymbolicExprPtr rhs);

} // namespace ir
} // namespace fxrt

#endif // __IR_SYMBOLIC_SYMBOLIC_H__
