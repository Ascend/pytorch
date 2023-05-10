// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include <acl/acl.h>
#include <sycl/sycl.hpp>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

using namespace at;
using namespace sycl;
using namespace sycl::access;
template <typename T>
using LocalAccessor = accessor<T, 1, mode::read_write, target::local>;

const size_t UB_BUF_SIZE = 256 * 1024;

template <typename T, int group_count, typename kernel_name>
void relu_forward_kernel(queue &sycl_queue, const T *in, T *out, size_t length) {
  const size_t num_accessor = 2;
  const size_t ub_load = UB_BUF_SIZE / (sizeof(T) * num_accessor);
  const size_t times = length / ub_load;
  size_t inner_length;
  for (size_t n = 0; n <= times; ++n) {
    if (length >= ub_load) {
      inner_length = ub_load;
      length -= ub_load;
    } else {
      inner_length = length;
    }
    sycl_queue.submit([&](handler &cgh) {
      LocalAccessor<T> local_in(inner_length, cgh);
      LocalAccessor<T> local_out(inner_length, cgh);
      cgh.parallel_for_work_group<kernel_name>(
          sycl::range<1>{group_count}, [=](group<1> grp) {
            const auto gid = grp.get_id(0);
            const auto idx_base = gid * inner_length;
            [[loop::parallel]] for (size_t i = 0; i < inner_length; ++i) {
              local_in[i] = in[i + idx_base + n * ub_load];
              local_out[i] = local_in[i] < 0 ? (T)0 : local_in[i];
              out[i + idx_base + n * ub_load] = local_out[i];
            }
          });
    });
    sycl_queue.wait();
  }
}

template <class T, class kernel_name>
torch::Tensor relu_forward_launch(torch::Tensor self) {
  Tensor result = at::empty(self.sizes(), self.options());
  const T *self_ptr = static_cast<T *>(self.storage().data_ptr().get());
  T *result_ptr = static_cast<T *>(result.storage().data_ptr().get());

  aclrtContext acl_context;
  aclrtGetCurrentContext(&acl_context);
  sycl::context sycl_context = sycl::make_context<sycl::backend::cce>(acl_context);
  int device_id = 0;
  aclrtGetDevice(&device_id);
  auto npu_stream = c10_npu::getCurrentNPUStream(device_id);
  auto acl_stream = npu_stream.stream();
  queue sycl_queue = sycl::make_queue<sycl::backend::cce>(acl_stream, sycl_context);

  const T *input1 = self_ptr;
  constexpr size_t group_count = 1;
  size_t len = result.numel() / group_count;
  relu_forward_kernel<T, group_count, kernel_name>(sycl_queue, input1, result_ptr, len);
  return result;
}

torch::Tensor relu_forward(torch::Tensor self) {
  at::Tensor fortmat_self = at_npu::native::NPUNativeFunctions::npu_format_cast(self, ACL_FORMAT_ND);

  const at::Tensor self_dtype = fortmat_self.scalar_type() == at::kDouble
    ? at_npu::native::NPUNativeFunctions::npu_dtype_cast(fortmat_self, at::kFloat) : fortmat_self;

  if (!at_npu::native::NPUNativeFunctions::check_match(self_dtype)) {
    at::Tensor contiguous_self =
        at_npu::native::NPUNativeFunctions::format_contiguous(self_dtype);
    self_dtype.copy_(contiguous_self);
  }

  torch::Tensor ret_tensor;
  if (self_dtype.scalar_type() == at::kHalf) {
    ret_tensor = relu_forward_launch<half, class relu_forward_launch_half>(self_dtype);
  } else if (self_dtype.scalar_type() == at::kFloat) {
    ret_tensor = relu_forward_launch<float, class relu_forward_launch_float>(self_dtype);
  } else {
    ret_tensor = relu_forward_launch<int, class relu_forward_launch_int>(self_dtype);
  }

  if (self.scalar_type() == at::kDouble) {
    ret_tensor = at_npu::native::NPUNativeFunctions::npu_dtype_cast(ret_tensor, at::kDouble);
  }
  return ret_tensor;
}

template <typename T, int group_count, typename kernel_name>
void relu_backward_kernel(queue &sycl_queue, const T *in, const T *grad_output, T *out, size_t length) {
  const size_t num_accessor = 4;
  const size_t ub_load = UB_BUF_SIZE / (sizeof(T) * num_accessor); //ub最大256K
  const size_t times = length / ub_load;
  size_t inner_length;
  for (size_t n = 0; n <= times; ++n) {
    if (length >= ub_load) {
      inner_length = ub_load;
      length -= ub_load;
    } else {
      inner_length = length;
    }
    sycl_queue.submit([&](handler &cgh) {
      LocalAccessor<T> local_in1(inner_length, cgh);
      LocalAccessor<T> local_in2(inner_length, cgh);
      LocalAccessor<T> local_out(inner_length, cgh);
      cgh.parallel_for_work_group<kernel_name>(
          sycl::range<1>{group_count}, [=](group<1> grp) {
            const auto gid = grp.get_id(0);
            const auto idx_base = gid * inner_length;
            [[loop::parallel]] for (size_t i = 0; i < inner_length; ++i) {
              local_in1[i] = grad_output[i + idx_base + n * ub_load];
              local_in2[i] = in[i + idx_base + n * ub_load] <= 0 ? 0 : 1;
              local_out[i] = local_in1[i] * local_in2[i];
              out[i + idx_base + n * ub_load] = local_out[i];
            }
          });
    });
    sycl_queue.wait();
  }
}

template <class T, class kernel_name>
torch::Tensor relu_backward_launch(torch::Tensor grad, torch::Tensor self) {
  Tensor result = at::empty(self.sizes(), self.options());
  const T *self_ptr = static_cast<T *>(self.storage().data_ptr().get());
  const T *grad_ptr = static_cast<T *>(grad.storage().data_ptr().get());
  T *result_ptr = static_cast<T *>(result.storage().data_ptr().get());

  aclrtContext acl_context;
  aclrtGetCurrentContext(&acl_context);
  sycl::context sycl_context = sycl::make_context<sycl::backend::cce>(acl_context);
  int device_id = 0;
  aclrtGetDevice(&device_id);
  auto npu_stream = c10_npu::getCurrentNPUStream(device_id);
  auto acl_stream = npu_stream.stream();
  queue sycl_queue = sycl::make_queue<sycl::backend::cce>(acl_stream, sycl_context);

  const T *input1 = self_ptr;
  const T *grad_output = grad_ptr;
  constexpr size_t group_count = 1;
  size_t len = result.numel() / group_count;
  relu_backward_kernel<T, group_count, kernel_name>(sycl_queue, input1, grad_output, result_ptr, len);
  return result;
}

torch::Tensor relu_backward(torch::Tensor grad, torch::Tensor self) {
  at::TensorList inputs = {grad, self};
  at::TensorList outputs;
  at_npu::native::NPUNativeFunctions::check_memory_overlaps(inputs, outputs);

  if (!at_npu::native::NPUNativeFunctions::check_match(self)) {
    at::Tensor contiguous_self = at_npu::native::NPUNativeFunctions::format_contiguous(self);
    self.copy_(contiguous_self);
  }

  if (self.scalar_type() == at::kHalf) {
    return relu_backward_launch<half, class relu_backward_launch_half>(grad, self);
  } else if (self.scalar_type() == at::kFloat) {
    return relu_backward_launch<float, class relu_backward_launch_float>(grad, self);
  } else {
    return relu_backward_launch<int, class relu_backward_launch_int>(grad, self);
  }
}

PYBIND11_MODULE(relu_bisheng, m) {
  m.def("forward", &relu_forward, "relu forward");
  m.def("backward", &relu_backward, "relu backward");
}
