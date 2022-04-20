#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/C++17.h>
#include <c10/util/SmallVector.h>

namespace at_npu {
namespace native {
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
inline hash_t hash_combine(hash_t seed, const c10::ArrayRef<T>& values) {
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
inline hash_t hash_combine(hash_t seed, const c10::SmallVector<T, N>& values) {
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
} // namespace native
} // namespace at_npu
