// Copyright (c) 2020 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <exception>
#include <memory>
#include <type_traits>
#include <typeinfo>

namespace c10 {

class AnyCastException : public std::exception {
 public:
  const char* what() const throw() {
    return "c10:Any Type Cast ERROR";
  }
};

class Any final {
  struct HolderBase {
    HolderBase() = default;
    virtual ~HolderBase() = default;
    virtual std::unique_ptr<HolderBase> Clone() const = 0;
    virtual const std::type_info& TypeInfo() const = 0;
  };

  template <typename T>
  struct Holder : public HolderBase {
    explicit Holder(const T& val) : value(val) {}
    explicit Holder(T&& val) : value(std::move(val)) {}

    Holder(const Holder& other) = delete;
    Holder& operator=(const Holder& other) = delete;

    std::unique_ptr<HolderBase> Clone() const override {
      return std::make_unique<Holder>(value);
    }

    const std::type_info& TypeInfo() const override {
      return typeid(T);
    }

    T value;
  };

 public:
  Any() = default;
  ~Any() = default;

  Any(const Any& other) {
    if (other.any_type_value_ != nullptr) {
      any_type_value_ = other.any_type_value_->Clone();
    }
  }

  // make a tmp Any object to call operator = of unique ptr
  Any& operator=(Any other) {
    any_type_value_ = std::move(other.any_type_value_);
    return *this;
  }

  Any(Any&& other) : any_type_value_(std::move(other.any_type_value_)) {}
  Any& operator=(Any&& other) = delete;

  template<typename T>
  Any(const T& value)
  : any_type_value_(std::make_unique<Holder<std::decay_t<T>>>(value)) {}

  template <typename T>
  Any(T&& value,
      typename std::enable_if<!std::is_same<Any&, T>::value>::type* = nullptr,
      typename std::enable_if<!std::is_const<T>::value>::type* = nullptr)
      : any_type_value_(std::make_unique<Holder<std::decay_t<T>>>(value)) {}

 private:
  // should be used in try catch otherwise no error will be reported
  template <typename T>
  friend T CastAs(Any& val);

  template <typename T>
  friend T CastAs(const Any& val);

  template <typename T>
  friend T CastAs(Any&& val);

  template <typename T>
  friend T* CastAs(Any* val_address);

  template <typename T>
  friend const T* CastAs(const Any* val_address);

  std::unique_ptr<HolderBase> any_type_value_ = nullptr;
};

template <typename T>
T CastAs(Any& val) {
  // for Pytorch C++ standard is 14
  // so remove_cv_t and remove_reference_t is available
  // if C++ standard is 11, should changed as below:
  // using remover_cvref_t = typename
  // std::remove_cv<std::remove_reference<T>::type>::type;

  using remover_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;
  return static_cast<T>(*CastAs<remover_cvref_t>(&val));
}

template <typename T>
T CastAs(const Any& val) {
  using remover_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;
  return static_cast<T>(*CastAs<remover_cvref_t>(&val));
}

template <typename T>
T CastAs(Any&& val) {
  using remover_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;
  return static_cast<T>(std::move(*CastAs<remover_cvref_t>(&val)));
}

template <typename T>
T* CastAs(Any* val_address) {
  if (val_address == nullptr ||
      val_address->any_type_value_->TypeInfo() != typeid(T)) {
    throw AnyCastException();
  }
  return &static_cast<Any::Holder<std::decay_t<T>>&>(
      *(val_address->any_type_value_.get()))
      .value;
}

template <typename T>
const T* CastAs(const Any* val_address) {
  using remover_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;
  if (val_address == nullptr ||
      val_address->any_type_value_->TypeInfo() != typeid(T)) {
    throw AnyCastException();
  }
  return &static_cast<Any::Holder<remover_cvref_t>&>(
      *(val_address->any_type_value_.get()))
      .value;
}

} // namespace c10