/**
 * Port of boost::any for C++11 compilers.
 * See http://www.boost.org/libs/any for Documentation.
 *
 * See also:
 *   + http://en.cppreference.com/w/cpp/any
 *   + http://en.cppreference.com/w/cpp/experimental/any
 *   + http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4562.html#any
 *   + https://cplusplus.github.io/LWG/lwg-active.html#2509
 *
 * Copyright Kevlin Henney, 2000, 2001, 2002. All rights reserved.
 * Copyright Claudio Fantacci, 2018. All rights reserved.
 * Copyright (c) 2021 Huawei Technologies Co., Ltd
 *
 * what:  variant type boost::any
 * who:   contributed by Kevlin Henney,
 *        with features contributed and bugs found by Antony Polukhin, Ed Brey, Mark Rodgers, Peter Dimov and James Curran,
 *        with C++11 compiler port by Claudio Fantacci
 * when:  July 2001, April 2013 - May 2013, September 2018
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE.md or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef ANY_H
#define ANY_H

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <typeinfo>
#include <type_traits>


namespace c10
{

/**
 * The class any describes a type-safe container for single values of any type.
 * An object of class any stores an instance of any type that satisfies the
 * constructor requirements or is empty, and this is referred to as the state
 * of the class any object. The stored instance is called the contained object.
 * Two states are equivalent if they are either both empty or if both are not
 * empty and if the contained objects are equivalent.
 * The non-member any_cast functions provide type-safe access to the contained object.
 */
class any final
{
public:
  /**
   * Constructs an empty object.
   */
  any() noexcept :
      content(nullptr)
  { }


  /**
   * Copies content of other into a new instance, so that any content is equivalent
   * in both type and value to those of other prior to the constructor call,
   * or empty if other is empty.
   */
  any(const any& other) :
      content(other.content ? other.content->clone() : nullptr)
  { }


  /**
   * Moves content of other into a new instance, so that any content is equivalent
   * in both type and value to those of other prior to the constructor call,
   * or empty if other is empty.
   */
  any(any&& other) noexcept :
      content(other.content)
  {
    other.content = nullptr;
  }


  /**
   * Constructs an object with initial content an object of type std::decay_t<ValueType>,
   * direct-initialized from std::forward<ValueType>(value). If
   * std::is_copy_constructible<std::decay_t<ValueType>>::value is false, the program is ill-formed.
   */
  template<typename ValueType>
  any(const ValueType& value) :
      content(new holder<typename std::remove_cv<typename std::decay<const ValueType>::type>::type>(value))
  { }


  /**
   * Constructs an object with initial content an object of type std::decay_t<ValueType>,
   * direct-initialized from std::forward<ValueType>(value). If
   * std::is_copy_constructible<std::decay_t<ValueType>>::value is false, the program is ill-formed.
   */
  template<typename ValueType>
  any(ValueType&& value, typename std::enable_if<!std::is_same<any&, ValueType>::value>::type* = nullptr, typename std::enable_if<!std::is_const<ValueType>::value>::type* = nullptr) :
      content(new holder<typename std::decay<ValueType>::type>(static_cast<ValueType&&>(value)))
  { }


  /**
   * Destruct the object.
   */
  ~any() noexcept
  {
    delete content;
  }


  /**
   * Assigns contents to the contained value.
   * Assigns by copying the state of rhs, as if by any(rhs).swap(*this).
   *
   * @param rhs object whose contained value to assign
   */
  any& operator=(const any& rhs)
  {
    any(rhs).swap(*this);
    return *this;
  }


  /**
   * Assigns contents to the contained value.
   * Assigns by moving the state of rhs, as if by any(std::move(rhs)).swap(*this).
   * rhs is left in a valid but unspecified state after the assignment.
   *
   * @param rhs object whose contained value to assign
   */
  any& operator=(any&& rhs) noexcept
  {
    rhs.swap(*this);
    any().swap(rhs);
    return *this;
  }


  /**
   * Assigns contents to the contained value.
   * Assigns the type and value of rhs, as if by any(std::forward<ValueType>(rhs)).swap(*this).
   * This overload only participates in overload resolution if std::decay_t<ValueType> is not
   * the same type as any and std::is_copy_constructible_v<std::decay_t<ValueType>> is true.
   *
   * @param rhs object whose contained value to assign
   */
  template <class ValueType>
  any& operator=(ValueType&& rhs)
  {
    any(static_cast<ValueType&&>(rhs)).swap(*this);
    return *this;
  }


  /**
   * If not empty, destroys the contained object.
   */
  void reset() noexcept
  {
    any().swap(*this);
  }


  /**
   * Swaps the content of two any objects.
   *
   * @param other object to swap with
   */
  any& swap(any& rhs) noexcept
  {
    std::swap(content, rhs.content);
    return *this;
  }

  /**
   * Checks whether the object contains a value.
   *
   * @return true if instance contains a value, otherwise false.
   */
  bool has_value() const noexcept
  {
    return content;
  }


  /**
   * Queries the contained type.
   *
   * The typeid of the contained value if instance is non-empty, otherwise typeid(void).
   */
  const std::type_info& type() const noexcept
  {
    return content ? content->type() : typeid(void);
  }


private:
  class placeholder
  {
  public:
    virtual ~placeholder()
    { }

  public:
    virtual const std::type_info& type() const noexcept = 0;

    virtual placeholder* clone() const = 0;

  };


  template<typename ValueType>
  class holder : public placeholder
  {
  public:
    holder(const ValueType& value) :
        held(value)
    { }


    holder(ValueType&& value) :
        held(static_cast<ValueType&&>(value))
    { }


    virtual const std::type_info& type() const noexcept
    {
      return typeid(ValueType);
    }


    virtual placeholder* clone() const
    {
      return new holder(held);
    }


    ValueType held;

  private:
    holder& operator=(const holder &);
  };


private:
  template<typename ValueType>
  friend ValueType* any_cast(any*) noexcept;

  placeholder* content;
};


/**
 * Overloads the std::swap algorithm for std::any. Swaps the content of two any objects by calling lhs.swap(rhs).
 *
 * @param lhs objects to swap
 * @param rhs objects to swap
 */
inline void swap(any& lhs, any& rhs) noexcept
{
  lhs.swap(rhs);
}


/**
 * Defines a type of object to be thrown by the value-returning forms of libanyboost::any_cast on failure.
 */
class bad_any_cast : public std::bad_cast
{
public:
  /**
   * Returns the explanatory string.
   *
   * Pointer to a null-terminated string with explanatory information. The pointer is guaranteed to be
   * valid at least until the exception object from which it is obtained is destroyed, or until a
   * non-const member function on the exception object is called.
   */
  virtual const char* what() const noexcept override
  {
    return "bad any_cast";
  }
};


/**
 * Performs type-safe access to the contained object.
 *
 * Throws libanyboost::bad_any_cast if the typeid of the requested
 * ValueType does not match that of the contents of operand.
 *
 * @param operand target any object
 */
template<typename ValueType>
ValueType* any_cast(any* operand) noexcept
{
  return operand && operand->type() == typeid(ValueType) ? std::addressof(static_cast<any::holder<typename std::remove_cv<ValueType>::type>*>(operand->content)->held) : nullptr;
}


/**
 * Performs type-safe access to the contained object.
 *
 * Throws libanyboost::bad_any_cast if the typeid of the requested
 * ValueType does not match that of the contents of operand.
 *
 * @param operand target any object
 */
template<typename ValueType>
inline const ValueType* any_cast(const any* operand) noexcept
{
  return any_cast<ValueType>(const_cast<any*>(operand));
}


/**
 * Performs type-safe access to the contained object.
 *
 * Throws libanyboost::bad_any_cast if the typeid of the requested
 * ValueType does not match that of the contents of operand.
 *
 * @param operand target any object
 */
template<typename ValueType>
ValueType any_cast(any& operand)
{
  typedef typename std::remove_reference<ValueType>::type nonref;

  nonref* result = any_cast<nonref>(std::addressof(operand));
  if(!result)
    throw bad_any_cast();

  typedef typename std::conditional<std::is_reference<ValueType>::value, ValueType, typename std::add_lvalue_reference<ValueType>::type>::type ref_type;

  return static_cast<ref_type>(*result);
}


/**
 * Performs type-safe access to the contained object.
 *
 * Throws libanyboost::bad_any_cast if the typeid of the requested
 * ValueType does not match that of the contents of operand.
 *
 * @param operand target any object
 */
template<typename ValueType>
inline ValueType any_cast(const any& operand)
{
  typedef typename std::remove_reference<ValueType>::type nonref;
  return any_cast<const nonref&>(const_cast<any&>(operand));
}


/**
 * Performs type-safe access to the contained object.
 *
 * Throws libanyboost::bad_any_cast if the typeid of the requested
 * ValueType does not match that of the contents of operand.
 *
 * @param operand target any object
 */
template<typename ValueType>
inline ValueType any_cast(any&& operand)
{
  static_assert(std::is_rvalue_reference<ValueType&&>::value || std::is_const< typename std::remove_reference<ValueType>::type >::value,
                "any_cast shall not be used for getting nonconst references to temporary objects");

  return any_cast<ValueType>(operand);
}

}


#endif /* ANY_H */