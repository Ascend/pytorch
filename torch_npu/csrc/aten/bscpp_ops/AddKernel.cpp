// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
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

#include <sycl/sycl.hpp>
#include <acl/acl.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

using namespace at;
using namespace sycl;
using namespace sycl::access;
template <typename T> using LocalAccessor = accessor<T, 1, mode::read_write, target::local>;

template <typename T, int group_count, typename kernel_name>
void bscpp_add_kernel(queue &sycl_queue, const T *in1, const T *in2, T *out, size_t length) {
  sycl_queue.submit([&](handler &cgh) {
      cgh.parallel_for_work_group<kernel_name>(sycl::range<1> { group_count }, [=](group<1> grp) {
          const auto gid = grp.get_id(0);
          const auto idx_base = gid * length;
          [[loop::parallel]] for (size_t i = 0; i < length; ++i) {
              out[i + idx_base] = in1[i + idx_base] + in2[i + idx_base];
          }
      });
  });
  sycl_queue.wait();
}

template<class T, class kernel_name>
Tensor bscpp_add_launch(const Tensor &self, const Tensor &other) {
  Tensor result = at::empty(self.sizes(), self.options());
  const T *self_ptr = static_cast<T*>(self.storage().data_ptr().get());
  const T *other_ptr = static_cast<T*>(other.storage().data_ptr().get());
  T *result_ptr = static_cast<T*>(result.storage().data_ptr().get());

  aclrtContext acl_context;
  aclrtGetCurrentContext(&acl_context);
  sycl::context sycl_context = sycl::make_context<sycl::backend::cce>(acl_context);
  int device_id = 0;
  aclrtGetDevice(&device_id);
  auto npu_stream = c10_npu::getCurrentNPUStream(device_id);
  auto acl_stream = npu_stream.stream();
  queue sycl_queue = sycl::make_queue<sycl::backend::cce>(acl_stream, sycl_context);

  const T *input1 = self_ptr;
  const T *input2 = other_ptr;
  constexpr size_t group_count = 1;
  size_t len = result.numel() / group_count;
  bscpp_add_kernel<T, group_count, kernel_name>(sycl_queue, input1, input2, result_ptr, len);

  return result;
}

at::Tensor NPUNativeFunctions::bscpp_add(const at::Tensor &self, const at::Tensor &other) {
  at::Tensor fortmat_self = at_npu::native::NPUNativeFunctions::npu_format_cast(self, ACL_FORMAT_ND);
  at::Tensor fortmat_other = at_npu::native::NPUNativeFunctions::npu_format_cast(other, ACL_FORMAT_ND);

  if (fortmat_self.scalar_type() == at::kHalf) {
    return bscpp_add_launch<half, class bscpp_add_launch_half>(fortmat_self, fortmat_other);
  } else if (fortmat_self.scalar_type() == at::kFloat) {
    return bscpp_add_launch<float, class bscpp_add_launch_float>(fortmat_self, fortmat_other);
  } else {
    return bscpp_add_launch<int64_t, class bscpp_add_launch_int>(fortmat_self, fortmat_other);
  }
}

}
} // namespace at_npu
