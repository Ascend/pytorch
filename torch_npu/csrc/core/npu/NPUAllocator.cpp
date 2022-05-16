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

#include <c10/core/DeviceType.h>

#include "torch_npu/csrc/core/npu/NPUAllocator.h"

C10_DEFINE_bool(
    caffe2_report_npu_memory_usage,
    false,
    "If set, print out detailed memory usage on NPU");

C10_DEFINE_bool(
    caffe2_npu_allocator_do_zero_fill,
    false,
    "If set, do memory zerofilling when allocating on NPU");

C10_DEFINE_bool(
    caffe2_npu_allocator_do_junk_fill,
    false,
    "If set, fill memory with deterministic junk when allocating on NPU");

namespace c10_npu {
void memset_junk_npu(void* data, size_t num) {
  // This garbage pattern is NaN when interpreted as floating point values,
  // or as very large integer values.
  static constexpr int32_t kJunkPattern = 0x7fedbeef;
  static constexpr int64_t kJunkPattern64 =
      (static_cast<int64_t>(kJunkPattern) << 32) | kJunkPattern;
  int32_t int64_count = num / sizeof(kJunkPattern64);
  int32_t remaining_bytes = num % sizeof(kJunkPattern64);
  int64_t* data_i64 = reinterpret_cast<int64_t*>(data);
  for (int i = 0; i < int64_count; i++) {
    data_i64[i] = kJunkPattern64;
  }
  if (remaining_bytes > 0) {
    memcpy(data_i64 + int64_count, &kJunkPattern64, remaining_bytes);
  }
}

void* alloc_npu(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  // We might have clowny upstream code that tries to alloc a negative number
  // of bytes. Let's catch it early.
  CAFFE_ENFORCE(
      ((ptrdiff_t)nbytes) >= 0,
      "alloc_cpu() seems to have been called with negative number: ",
      nbytes);

  void* data = nullptr;
#ifdef __ANDROID__
  data = memalign(gAlignment, nbytes);
#elif defined(_MSC_VER)
  data = _aligned_malloc(nbytes, gAlignment);
#else
  int err = posix_memalign(&data, gAlignment, nbytes);
  if (err != 0) {
    CAFFE_THROW(
        "DefaultCPUAllocator: can't allocate memory: you tried to allocate ",
        nbytes,
        " bytes. Error code ",
        err,
        " (",
        strerror(err),
        ")");
  }
#endif

  CAFFE_ENFORCE(
      data,
      "DefaultCPUAllocator: not enough memory: you tried to allocate ",
      nbytes,
      " bytes. Buy new RAM!");

  // move data to a thread's NUMA node
  c10::NUMAMove(data, nbytes, c10::GetCurrentNUMANode());
  CHECK(
      !FLAGS_caffe2_npu_allocator_do_zero_fill ||
      !FLAGS_caffe2_npu_allocator_do_junk_fill)
      << "Cannot request both zero-fill and junk-fill at the same time";
  if (FLAGS_caffe2_npu_allocator_do_zero_fill) {
    memset(data, 0, nbytes);
  } else if (FLAGS_caffe2_npu_allocator_do_junk_fill) {
    memset_junk_npu(data, nbytes);
  }

  return data;
}

void free_npu(void* data) {
#ifdef _MSC_VER
  _aligned_free(data);
#else
  free(data);
#endif
}
struct DefaultNPUAllocator final : at::Allocator {
  DefaultNPUAllocator() {}
  ~DefaultNPUAllocator() override {}
  at::DataPtr allocate(size_t nbytes) const override {
    void* data = alloc_npu(nbytes);
    return {data, data, &free_npu, at::Device(at_npu::key::NativeDeviceType)};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &free_npu;
  }
};

at::Allocator* GetNPUAllocator() {
  return GetAllocator(at_npu::key::NativeDeviceType);
}

void SetNPUAllocator(at::Allocator* alloc) {
  SetAllocator(at_npu::key::NativeDeviceType, alloc);
}

// Global default NPU Allocator
static DefaultNPUAllocator g_npu_alloc;

at::Allocator* GetDefaultNPUAllocator() {
  return &g_npu_alloc;
}

// Need to register NPU Allocator for NPU Device
} // namespace torch_npu