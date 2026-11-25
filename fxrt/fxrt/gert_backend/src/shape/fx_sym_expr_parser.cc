#include "shape/fx_sym_expr_parser.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace fx_converter {
namespace {

constexpr int kMaxDepth = 64;

// Recursive-descent parser over SymbolicExpr::ToString() output. Every binary node is printed
// fully parenthesized, so a paren always introduces exactly one infix operator.
class SymExprParser {
 public:
  SymExprParser(const std::string &text, FxSymbolEnv &env) : text_(text), env_(env) {}

  fxrt::ir::SymbolicExprPtr Run() {
    auto expr = ParseExpr();
    SkipSpace();
    if (expr == nullptr || pos_ != text_.size()) {
      return nullptr;  // unparsed tail: reject rather than silently truncate
    }
    return expr;
  }

 private:
  void SkipSpace() {
    while (pos_ < text_.size() && text_[pos_] == ' ') {
      ++pos_;
    }
  }

  char Peek() {
    SkipSpace();
    return (pos_ < text_.size()) ? text_[pos_] : '\0';
  }

  bool Eat(char c) {
    if (Peek() != c) {
      return false;
    }
    ++pos_;
    return true;
  }

  fxrt::ir::SymbolicExprPtr ParseExpr() {
    if (depth_ >= kMaxDepth) {
      return nullptr;
    }
    ++depth_;
    auto expr = ParseAtom();
    --depth_;
    return expr;
  }

  fxrt::ir::SymbolicExprPtr ParseAtom() {
    const char c = Peek();
    if (c == '(') {
      return ParseParen();
    }
    if ((std::isalpha(static_cast<unsigned char>(c)) != 0) || c == '_') {
      return ParseIdentOrCall();
    }
    if ((std::isdigit(static_cast<unsigned char>(c)) != 0) || c == '-') {
      return ParseInt();
    }
    return nullptr;
  }

  fxrt::ir::SymbolicExprPtr ParseParen() {
    if (!Eat('(')) {
      return nullptr;
    }
    auto lhs = ParseExpr();
    if (lhs == nullptr) {
      return nullptr;
    }
    const char op = Peek();
    if (op != '+' && op != '*' && op != '/' && op != '%') {
      return nullptr;
    }
    ++pos_;
    auto rhs = ParseExpr();
    if (rhs == nullptr || !Eat(')')) {
      return nullptr;
    }
    switch (op) {
      case '+':
        return lhs + rhs;
      case '*':
        return lhs * rhs;
      case '/':
        return lhs / rhs;
      default:
        return lhs % rhs;
    }
  }

  fxrt::ir::SymbolicExprPtr ParseIdentOrCall() {
    SkipSpace();
    const size_t start = pos_;
    while (pos_ < text_.size() &&
           ((std::isalnum(static_cast<unsigned char>(text_[pos_])) != 0) || text_[pos_] == '_')) {
      ++pos_;
    }
    const std::string ident = text_.substr(start, pos_ - start);
    if (ident.empty()) {
      return nullptr;
    }
    if (Peek() != '(') {
      return env_.EnsureIrVar(ident);
    }
    ++pos_;  // '('
    auto lhs = ParseExpr();
    if (lhs == nullptr || !Eat(',')) {
      return nullptr;
    }
    auto rhs = ParseExpr();
    if (rhs == nullptr || !Eat(')')) {
      return nullptr;
    }
    if (ident == "floor_div") {
      return fxrt::ir::FloorDiv(lhs, rhs);
    }
    if (ident == "ceil_div") {
      return fxrt::ir::CeilDiv(lhs, rhs);
    }
    if (ident == "min") {
      return fxrt::ir::Min(lhs, rhs);
    }
    return nullptr;
  }

  fxrt::ir::SymbolicExprPtr ParseInt() {
    SkipSpace();
    const size_t start = pos_;
    if (pos_ < text_.size() && text_[pos_] == '-') {
      ++pos_;
    }
    const size_t digits_start = pos_;
    while (pos_ < text_.size() && (std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0)) {
      ++pos_;
    }
    if (pos_ == digits_start) {
      return nullptr;
    }
    const std::string digits = text_.substr(start, pos_ - start);
    errno = 0;
    char *end = nullptr;
    const int64_t value = static_cast<int64_t>(std::strtoll(digits.c_str(), &end, 10));
    if (errno != 0 || end == nullptr || *end != '\0') {
      return nullptr;
    }
    return fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicConst>(value);
  }

  const std::string text_;  // by value: ParseFxSymExpr passes a ToString() temporary
  FxSymbolEnv &env_;
  size_t pos_ = 0;
  int depth_ = 0;
};

}  // namespace

fxrt::ir::SymbolicExprPtr ParseFxSymExpr(const fxrt::ir::SymbolicExprPtr &bridged, FxSymbolEnv &env) {
  if (bridged == nullptr) {
    return nullptr;
  }
  // A constant survives the bridge intact ("c:" prefix) and holds no mutable state, so it can be
  // shared as-is. Everything else goes through ToString(): for a bridged variable that string is
  // the original printed expression, and for a real tree (should the bridge ever stop flattening)
  // it round-trips through the same grammar while re-interning vars against env.
  if (bridged->GetKind() == fxrt::ir::SymbolicExpr::Kind::Constant) {
    return bridged;
  }
  SymExprParser parser(bridged->ToString(), env);
  return parser.Run();
}

}  // namespace fx_converter
