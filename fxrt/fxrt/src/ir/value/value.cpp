#include <iostream>
#include <utility>
#include <vector>
#include "ir/value/value.h"

namespace fxrt {
namespace ir {
constexpr const char* kTagStrings[] = {
    "None",
    "Tensor",
    "Double",
    "Int",
    "Bool",
    "String",
    "Tuple",
    "Symbol",
};

std::vector<ir::TensorPtr> Tuple::ToTensorList() {
  std::vector<ir::TensorPtr> tensorList;
  tensorList.reserve(elements_.size());
  for (size_t i = 0; i < elements_.size(); ++i) {
    (void)tensorList.emplace_back(elements_[i]->ToTensor());
  }
  return tensorList;
}

std::vector<int64_t> Tuple::ToIntList() {
  std::vector<int64_t> intList;
  intList.reserve(elements_.size());
  for (size_t i = 0; i < elements_.size(); ++i) {
    (void)intList.emplace_back(elements_[i]->ToInt());
  }
  return intList;
}

std::vector<uint8_t> Tuple::ToBoolList() {
  std::vector<uint8_t> boolList;
  boolList.reserve(elements_.size());
  for (size_t i = 0; i < elements_.size(); ++i) {
    (void)boolList.emplace_back(static_cast<uint8_t>(elements_[i]->ToBool()));
  }
  return boolList;
}

std::vector<double> Tuple::ToDoubleList() {
  std::vector<double> doubleList;
  doubleList.reserve(elements_.size());
  for (size_t i = 0; i < elements_.size(); ++i) {
    (void)doubleList.emplace_back(elements_[i]->ToDouble());
  }
  return doubleList;
}

const char* TagToString(Value::Tag tag) {
  return kTagStrings[static_cast<size_t>(tag)];
}

Value::Value(const TensorPtr& v) : tag_(Tag::Tensor), tensor_(v) {}
Value::Value(double v) : tag_(Tag::Double), double_(v) {}
Value::Value(int64_t v) : tag_(Tag::Int), int_(v) {}
Value::Value(bool v) : tag_(Tag::Bool), bool_(v) {}
Value::Value(std::string&& v) : tag_(Tag::String) {
  new (&string_) std::string(std::move(v));
}
Value::Value(const TuplePtr& v) : tag_(Tag::Tuple), tuple_(v) {}
Value::Value(const SymbolicExprPtr& v) : tag_(Tag::Symbol), symbol_(v) {}

Value::~Value() {
  switch (tag_) {
    case Tag::Tensor:
      tensor_.~IntrusivePtr();
      break;
    case Tag::String:
      string_.~basic_string();
      break;
    case Tag::Tuple:
      tuple_.~IntrusivePtr();
      break;
    case Tag::Symbol:
      symbol_.~IntrusivePtr();
      break;
    default:
      break;
  }
}

Value::Value(Value&& other) noexcept : tag_(other.tag_) {
  switch (tag_) {
    case Tag::Tensor:
      new (&tensor_) TensorPtr(std::move(other.tensor_));
      break;
    case Tag::Double:
      double_ = other.double_;
      break;
    case Tag::Int:
      int_ = other.int_;
      break;
    case Tag::Bool:
      bool_ = other.bool_;
      break;
    case Tag::String:
      new (&string_) std::string(std::move(other.string_));
      break;
    case Tag::Tuple:
      new (&tuple_) TuplePtr(std::move(other.tuple_));
      break;
    case Tag::Symbol:
      new (&symbol_) SymbolicExprPtr(std::move(other.symbol_));
      break;
    case Tag::None:
      break;
  }
}

Value& Value::operator=(Value&& other) noexcept {
  if (this != &other) {
    // NOTE:
    // `Value` is managed by intrusive refcounting (RefCounted base). We must NOT
    // destroy and placement-new the whole object here, otherwise RefCounted will
    // be re-constructed and its refCount_ will be reset to 0, corrupting the
    // reference count of existing ValuePtr holders.
    //
    // Also, `tag_` is declared `const` (see value.h), so move-assignment cannot
    // change the tag. This matches the semantics of copy-assignment: the tag
    // must be the same.
    if (tag_ != other.tag_) {
      RT_GLOG(EXCEPTION) << "Cannot assign Value with different tag. Current tag: " << TagToString(tag_)
                         << ", other tag: " << TagToString(other.tag_);
    }

    switch (tag_) {
      case Tag::Tensor:
        tensor_ = std::move(other.tensor_);
        break;
      case Tag::Double:
        double_ = other.double_;
        break;
      case Tag::Int:
        int_ = other.int_;
        break;
      case Tag::Bool:
        bool_ = other.bool_;
        break;
      case Tag::String:
        string_ = std::move(other.string_);
        break;
      case Tag::Tuple:
        tuple_ = std::move(other.tuple_);
        break;
      case Tag::Symbol:
        symbol_ = std::move(other.symbol_);
        break;
      case Tag::None:
        break;
    }
  }
  return *this;
}

Value& Value::operator=(const Value& other) {
  if (this != &other) {
    if (tag_ != other.tag_) {
      RT_GLOG(EXCEPTION) << "Cannot assign Value with different tag. Current tag: " << TagToString(tag_)
                         << ", other tag: " << TagToString(other.tag_);
    }
    switch (tag_) {
      case Tag::Tensor:
        tensor_ = other.tensor_;
        break;
      case Tag::Double:
        double_ = other.double_;
        break;
      case Tag::Int:
        int_ = other.int_;
        break;
      case Tag::Bool:
        bool_ = other.bool_;
        break;
      case Tag::String:
        string_ = other.string_;
        break;
      case Tag::Tuple:
        tuple_ = other.tuple_;
        break;
      case Tag::Symbol:
        symbol_ = other.symbol_;
        break;
      case Tag::None:
        break;
    }
  }
  return *this;
}

#define CHECK_TAG(expected)                                                                       \
  if (tag_ != expected) {                                                                         \
    RT_GLOG(EXCEPTION) << "Bad Value access, value: " << *this << ", type: " << TagToString(tag_) \
                       << ", expected type: " << TagToString(expected);                           \
  }

const TensorPtr& Value::ToTensor() const {
  CHECK_TAG(Tag::Tensor);
  return tensor_;
}
double Value::ToDouble() const {
  CHECK_TAG(Tag::Double);
  return double_;
}
int64_t Value::ToInt() const {
  if (tag_ == Tag::Symbol) {
    return symbol_->Evaluate();
  }
  CHECK_TAG(Tag::Int);
  return int_;
}
bool Value::ToBool() const {
  CHECK_TAG(Tag::Bool);
  return bool_;
}
const std::string& Value::ToString() const {
  CHECK_TAG(Tag::String);
  return string_;
}
const TuplePtr& Value::ToTuple() const {
  CHECK_TAG(Tag::Tuple);
  return tuple_;
}
const SymbolicExprPtr& Value::ToSymbol() const {
  CHECK_TAG(Tag::Symbol);
  return symbol_;
}

std::ostream& operator<<(std::ostream& os, const TuplePtr& tuple) {
  if (tuple == nullptr) {
    os << "Tuple(Null)";
  } else {
    os << "Tuple(";
    bool first = true;
    for (const auto& item : *tuple) {
      if (!first) {
        os << ", ";
      }
      os << item;
      first = false;
    }
    os << ")";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const ValuePtr& value) {
  if (value == nullptr) {
    os << "Null";
  } else {
    os << *value;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<const Value*>& values) {
  os << "{";
  for (size_t i = 0; i < values.size(); ++i) {
    os << *(values[i]);
    if (i < values.size() - 1) {
      os << ", ";
    }
  }
  os << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Value& value) {
  switch (value.tag_) {
    case Value::Tag::None:
      os << "None";
      break;
    case Value::Tag::Tensor:
      os << value.ToTensor();
      break;
    case Value::Tag::Double:
      os << value.ToDouble();
      break;
    case Value::Tag::Int:
      os << value.ToInt();
      break;
    case Value::Tag::Bool:
      os << (value.ToBool() ? "true" : "false");
      break;
    case Value::Tag::String:
      os << "\"" << value.ToString() << "\"";
      break;
    case Value::Tag::Tuple:
      os << value.ToTuple();
      break;
    case Value::Tag::Symbol:
      os << value.ToSymbol()->ToString();
      break;
  }
  return os;
}

void VisitAllTensors(const ir::ValuePtr& value, const std::function<void(const ir::TensorPtr&)>& func) {
  if (value->IsTensor()) {
    func(value->ToTensor());
  } else if (value->IsTuple()) {
    for (auto& item : *value->ToTuple()) {
      VisitAllTensors(item, func);
    }
  }
}

TuplePtr Tuple::DeepCopy() const {
  std::vector<ValuePtr> new_elements;
  new_elements.reserve(elements_.size());
  for (const auto& element : elements_) {
    if (element) {
      new_elements.push_back(element->DeepCopy());
    } else {
      new_elements.push_back(nullptr);
    }
  }
  return MakeIntrusive<Tuple>(std::move(new_elements));
}

ValuePtr Value::DeepCopy() const {
  switch (tag_) {
    case Tag::Tensor:
      if (tensor_) {
        return MakeIntrusive<Value>(tensor_->DeepCopy());
      } else {
        return MakeIntrusive<Value>(TensorPtr{nullptr});
      }
    case Tag::Double:
      return MakeIntrusive<Value>(double_);
    case Tag::Int:
      return MakeIntrusive<Value>(int_);
    case Tag::Bool:
      return MakeIntrusive<Value>(bool_);
    case Tag::String:
      return MakeIntrusive<Value>(std::string(string_));
    case Tag::Tuple:
      if (tuple_) {
        return MakeIntrusive<Value>(tuple_->DeepCopy());
      } else {
        return MakeIntrusive<Value>(TuplePtr{nullptr});
      }
    case Tag::Symbol:
      if (symbol_) {
        return MakeIntrusive<Value>(symbol_->DeepCopy());
      } else {
        return MakeIntrusive<Value>(SymbolicExprPtr{nullptr});
      }
    case Tag::None:
    default:
      return MakeIntrusive<Value>();
  }
}

} // namespace ir
} // namespace fxrt
