// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#include <c10/util/ArrayRef.h>
#include <c10/util/C++17.h>
#include <c10/util/SmallVector.h>

namespace c10 {
namespace npu {
namespace hash_utils {
using hash_t = size_t;
constexpr hash_t hash_seed = 0x7863a7de;

template <typename T>
inline hash_t hash_combine(hash_t seed, const T& value) {
  std::hash<T> hasher;
  seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}

template <typename T>
inline hash_t hash_combine(hash_t seed, const ArrayRef<T>& values) {
  for (auto& v : values) {
    seed = hash_combine(seed, v);
  }
  return seed;
}

template <typename T>
inline hash_t hash_combine(hash_t seed, const std::vector<T>& values) {
  for (auto& v : values) {
    seed = hash_combine(seed, v);
  }
  return seed;
}

template <typename T, unsigned N>
inline hash_t hash_combine(hash_t seed, const SmallVector<T, N>& values) {
  for (auto& v : values) {
    seed = hash_combine(seed, v);
  }
  return seed;
}

template <typename T = void>
hash_t multi_hash() {
  return hash_seed;
}

template <typename T, typename... Args>
hash_t multi_hash(const T& value, Args... args) {
  return hash_combine(multi_hash(args...), value);
}
} // namespace hash_utils
} // namespace npu
} // namespace c10
