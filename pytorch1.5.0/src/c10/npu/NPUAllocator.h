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

#include <cstring>
#include <unordered_map>

#include <c10/core/Allocator.h>
#include <c10/util/Logging.h>
#include <c10/util/numa.h>

// TODO: rename to c10
C10_DECLARE_bool(caffe2_report_npu_memory_usage);
C10_DECLARE_bool(caffe2_npu_allocator_do_zero_fill);
C10_DECLARE_bool(caffe2_npu_allocator_do_junk_fill);

namespace c10 {

// Use 64-byte alignment should be enough for computation up to AVX512.
constexpr size_t gAlignment = 64;

using MemoryDeleter = void (*)(void*);

// A helper function that is basically doing nothing.
// C10_API void NoDelete(void*);

// Fill the data memory region of num bytes with a particular garbage pattern.
// The garbage value is chosen to be NaN if interpreted as floating point value,
// or a very large integer.
C10_API void memset_junk_npu(void* data, size_t num);

C10_API void* alloc_npu(size_t nbytes);
C10_API void free_npu(void* data);

// Get the CPU Allocator.
C10_API at::Allocator* GetNPUAllocator();
// Sets the CPU allocator to the given allocator: the caller gives away the
// ownership of the pointer.
C10_API void SetNPUAllocator(at::Allocator* alloc);

// Get the Default CPU Allocator
C10_API at::Allocator* GetDefaultNPUAllocator();

} // namespace c10
