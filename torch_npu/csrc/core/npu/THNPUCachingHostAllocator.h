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

#include <c10/core/Allocator.h>
#include <c10/npu/NPUStream.h>
#include <c10/util/Exception.h>

C10_NPU_API c10::Allocator* getTHNPUCachingHostAllocator(void);

C10_NPU_API aclError
THNPUCachingHostAllocator_recordEvent(void* ptr, at::npu::NPUStream stream);

// Releases cached pinned memory allocations via npuHostFree
C10_NPU_API void THNPUCachingHostAllocator_emptyCache(void);

c10::Allocator* getPinnedMemoryAllocator(void);
