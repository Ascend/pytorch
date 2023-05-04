// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
#ifndef TORCH_NPU_TOOLKIT_PROFILER_SINGLETON_INC
#define TORCH_NPU_TOOLKIT_PROFILER_SINGLETON_INC

#include <memory>

namespace torch_npu {
namespace toolkit {
namespace profiler {
template<typename T>
class Singleton {
public:
  static T *GetInstance() noexcept(std::is_nothrow_constructible<T>::value) {
    static T instance;
    return &instance;
  }

  virtual ~Singleton() = default;

protected:
  explicit Singleton() = default;

private:
  explicit Singleton(const Singleton &obj) = delete;
  Singleton& operator=(const Singleton &obj) = delete;
  explicit Singleton(Singleton &&obj) = delete;
  Singleton& operator=(Singleton &&obj) = delete;
};
} // profiler
} // toolkit
} // torch_npu
#endif
