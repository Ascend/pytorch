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
#include <ATen/record_function.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/framework/OpRunner.h"

namespace at_npu {
namespace native {

using namespace at;
using namespace sycl;
using namespace sycl::access;
template <typename T> using LocalAccessor = accessor<T, 1, mode::read_write, target::local>;


queue get_sycl_queue() {
  aclrtContext acl_context;
  aclrtGetCurrentContext(&acl_context);
  sycl::context sycl_context = sycl::make_context<sycl::backend::cce>(acl_context);
  int device_id;
  aclrtGetDevice(&device_id);

  aclrtStream acl_stream = c10_npu::option::OptionsManager::CheckQueueEnable() ?
      c10_npu::getCurrentNPUStreamNoWait(device_id) : c10_npu::getCurrentNPUStream(device_id).stream();
  queue sycl_queue = sycl::make_queue<sycl::backend::cce>(acl_stream, sycl_context);
  return sycl_queue;
}

template <typename T, int group_count, typename kernel_name>
void bscpp_add_kernel(queue sycl_queue, const T *in1, const T *in2, T *out, size_t length) {
  sycl_queue.submit([&](handler& cgh) {
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
void bscpp_add_launch(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  const T *self_ptr = static_cast<T*>(self.storage().data_ptr().get());
  const T *other_ptr = static_cast<T*>(other.storage().data_ptr().get());
  T *result_ptr = static_cast<T*>(result.storage().data_ptr().get());

  queue sycl_queue = get_sycl_queue();

  constexpr size_t group_count = 1;
  size_t len = result.numel();
  auto launch_bisheng_kernel = [sycl_queue, self_ptr, other_ptr, result_ptr, len] () {
        bscpp_add_kernel<T, group_count, kernel_name>(sycl_queue, self_ptr, other_ptr, result_ptr, len);
    };
  OpRunner opRun;
  opRun.Name("bscpp_add")
       .Func(launch_bisheng_kernel)
       .Run();
}

at::Tensor NPUNativeFunctions::bscpp_add(const at::Tensor& self, const at::Tensor& other) {
  static auto warn_once = [](){ 
      std::cout << "Warning: kernel [bscpp_add] is written by BiShengCPP." << std::endl;
      return true;
  }();
  auto self_format = NPUNativeFunctions::get_npu_format(self);
  at::Tensor fortmat_self = NPUNativeFunctions::npu_format_cast(self, ACL_FORMAT_ND);
  at::Tensor fortmat_other = NPUNativeFunctions::npu_format_cast(other, ACL_FORMAT_ND);
  at::Tensor result = at::empty(fortmat_self.sizes(), fortmat_self.options());
  if (fortmat_self.scalar_type() == at::kHalf) {
    bscpp_add_launch<half, class bscpp_add_launch_half>(fortmat_self, fortmat_other, result);
  } else if (fortmat_self.scalar_type() == at::kFloat) {
    bscpp_add_launch<float, class bscpp_add_launch_float>(fortmat_self, fortmat_other, result);
  } else if (fortmat_self.scalar_type() == at::kInt) {
    bscpp_add_launch<int32_t, class bscpp_add_launch_int>(fortmat_self, fortmat_other, result);
  } else {
    AT_ERROR("bscpp_add not implemented for '", toString(fortmat_self.scalar_type()), "'");
  }
  at::Tensor output = (self_format == ACL_FORMAT_ND) ?
      result : NPUNativeFunctions::npu_format_cast(result, ACL_FORMAT_ND);

  return output;
}
}
} // namespace at_npu
