#ifndef __IR_VALUE_VALUE_H__
#define __IR_VALUE_VALUE_H__

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <stdexcept>
#include <functional>
#include <ostream>

#include "common/common.h"
#include "common/visible.h"
#include "ir/common/intrusive_ptr.h"
#include "ir/tensor/tensor.h"
#include "ir/symbolic/symbolic.h"

namespace fxrt {
namespace ir {

class Value;
using ValuePtr = IntrusivePtr<Value>;
class Tuple;
using TuplePtr = IntrusivePtr<Tuple>;

/**
 * @brief A tuple of values.
 *
 * This class provides holds a vector of Value objects.
 */
class Tuple : public RefCounted {
 public:
  /**
   * @brief Default constructor. Creates an empty Tuple.
   */
  Tuple() = default;

  /**
   * @brief Constructs a Tuple from a vector of Value objects.
   * @param elements The vector of elements to include in the tuple.
   */
  explicit Tuple(const std::vector<ValuePtr>& elements) : elements_(elements) {}
  explicit Tuple(std::vector<ValuePtr>&& elements) : elements_(std::move(elements)) {}

  Tuple(const Tuple&) = delete;
  Tuple& operator=(const Tuple&) = delete;
  Tuple(Tuple&&) = delete;
  Tuple& operator=(Tuple&&) = delete;

  /**
   * @brief Get the size of the tuple.
   * @return The number of elements in the tuple.
   */
  size_t Size() const {
    return elements_.size();
  }

  /**
   * @brief Retrieves an element by index.
   * @param index The index of the element to retrieve.
   * @return The element as ValuePtr.
   */
  const ValuePtr& operator[](size_t index) const {
    CHECK_IF_FAIL(index < elements_.size());
    return elements_[index];
  }

  auto begin() const {
    return elements_.cbegin();
  }
  auto end() const {
    return elements_.cend();
  }
  auto begin() {
    return elements_.begin();
  }
  auto end() {
    return elements_.end();
  }

  std::vector<ir::TensorPtr> ToTensorList();
  std::vector<int64_t> ToIntList();
  std::vector<uint8_t> ToBoolList();
  std::vector<double> ToDoubleList();

  /**
   * @brief Creates a deep copy of this Tuple object.
   * @return A new Tuple object with copied elements.
   */
  TuplePtr DeepCopy() const;

 private:
  std::vector<ValuePtr> elements_;
};

/**
 * @brief A generic container for different types of values.
 *
 * This class can hold a variety of types, including tensors, scalars, strings,
 * and tuples. It uses a tagged union to store the data efficiently.
 */
class DA_API Value : public RefCounted {
 public:
  /**
   * @brief Enumeration of the possible types a Value can hold.
   */
  enum class Tag { None, Tensor, Double, Int, Bool, String, Tuple, Symbol };

  /**
   * @brief Default constructor. Creates a None value.
   */
  Value() : tag_(Tag::None) {}
  /**
   * @brief Constructs a Value from a TensorPtr.
   * @param v The TensorPtr value.
   */
  explicit Value(const TensorPtr& v);
  /**
   * @brief Constructs a Value from a double.
   * @param v The double value.
   */
  explicit Value(double v);
  /**
   * @brief Constructs a Value from an int64_t.
   * @param v The int64_t value.
   */
  explicit Value(int64_t v);
  /**
   * @brief Constructs a Value from a bool.
   * @param v The bool value.
   */
  explicit Value(bool v);
  /**
   * @brief Constructs a Value from a std::string by moving.
   * @param v The string value.
   */
  explicit Value(std::string&& v);
  /**
   * @brief Constructs a Value from a TuplePtr.
   * @param v The TuplePtr value.
   */
  explicit Value(const TuplePtr& v);
  /**
   * @brief Constructs a Value from a SymbolicExprPtr.
   * @param v The SymbolicExprPtr value.
   */
  explicit Value(const SymbolicExprPtr& v);

  /**
   * @brief Destructor.
   */
  ~Value();

  /**
   * @brief Move constructor.
   * @param other The Value to move from.
   */
  Value(Value&& other) noexcept;
  /**
   * @brief Move assignment operator.
   * @param other The Value to move from.
   */
  Value& operator=(Value&& other) noexcept;
  /**
   * @brief Assignment operator.
   * @param other The Value to assign from.
   * @throw runtime exception if the tag is different.
   */
  Value& operator=(const Value& other);

  /**
   * @brief Creates a deep copy of this Value object.
   * @return A new Value object with copied data.
   */
  ValuePtr DeepCopy() const;

  /** @name Type checkers */
  ///@{
  bool IsTensor() const {
    return tag_ == Tag::Tensor;
  }
  bool IsDouble() const {
    return tag_ == Tag::Double;
  }
  bool IsInt() const {
    return tag_ == Tag::Int;
  }
  bool IsBool() const {
    return tag_ == Tag::Bool;
  }
  bool IsString() const {
    return tag_ == Tag::String;
  }
  bool IsTuple() const {
    return tag_ == Tag::Tuple;
  }
  bool IsSymbol() const {
    return tag_ == Tag::Symbol;
  }
  bool IsNone() const {
    return tag_ == Tag::None;
  }
  ///@}

  /** @name Value extractors
   *  These methods extract the underlying value. They will throw a
   *  std::runtime_error if the type does not match.
   */
  ///@{
  const TensorPtr& ToTensor() const;
  double ToDouble() const;
  int64_t ToInt() const;
  bool ToBool() const;
  const std::string& ToString() const;
  const TuplePtr& ToTuple() const;
  const SymbolicExprPtr& ToSymbol() const;
  ///@}
  Tag GetTag() const {
    return tag_;
  }

  /**
   * @brief Overloads the output stream operator for ValuePtr.
   * @param os The output stream.
   * @param value The ValuePtr to output.
   * @return The output stream.
   */
  friend std::ostream& operator<<(std::ostream& os, const Value& value);

 private:
  /**
   * @brief Tag string representation.
   * @param tag Input tag enumeration.
   * @return String representation of the tag enumeration.
   */
  friend const char* TagToString(Tag tag);

  const Tag tag_; ///< The tag indicating the type of the value.
  union {
    TensorPtr tensor_;
    double double_;
    int64_t int_;
    bool bool_;
    std::string string_;
    TuplePtr tuple_;
    SymbolicExprPtr symbol_;
  };
};

std::ostream& operator<<(std::ostream& os, const Value& value);
std::ostream& operator<<(std::ostream& os, const ValuePtr& value);
std::ostream& operator<<(std::ostream& os, const std::vector<const Value*>& values);
std::ostream& operator<<(std::ostream& os, const TuplePtr& tuple);

// Recursively visits all tensors contained within the given Value.
void VisitAllTensors(const ir::ValuePtr& value, const std::function<void(const ir::TensorPtr&)>& func);

} // namespace ir
} // namespace fxrt

#endif // __IR_VALUE_VALUE_H__
