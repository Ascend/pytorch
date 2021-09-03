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

#ifndef __NATIVE_NPU_UTILS_NUP_UTILS__
#define __NATIVE_NPU_UTILS_NUP_UTILS__

#include <stdint.h>
#include <third_party/acl/inc/acl/acl.h>
#include <third_party/acl/inc/acl/acl_base.h>
#include <third_party/acl/inc/ge/ge_error_codes.h>
#include <string>
#include <vector>
#include "ATen/ATen.h"
#include "c10/npu/npu_log.h"

using std::string;
using std::vector;

namespace at {
namespace native {
namespace npu {

// smallvector max size
const int N = 32;
// npu tensor max size
const int SHAPE_SIZE = 8;
// HALF_MAX and HALF_MIN of NPU support
const int NPU_HALF_MAX = 65504;
const int NPU_HALF_MIN = -65504;
const int NPU_MAX_OP_EXEC_TRY_NUM = 2;

typedef enum MemoryType{
  MEMORY_DEVICE,
  MEMORY_HOST,
} MemoryType;

class NpuUtils {
 public:
  static bool check_match(const Tensor* tensor);
  static Tensor format_contiguous(const Tensor& src);
  static Tensor format_contiguous_add_copy_optimize(const Tensor& src);
  static void RefreshFormat(const Tensor& tensor);
  static void format_fresh_view(
      Tensor& x,
      const Tensor& y);

  static bool check_5d_5d_match(const Tensor& tensor);
  static bool IsOomError(aclError ret, int index);
};
} // namespace npu
} // namespace native
} // namespace at

#endif
