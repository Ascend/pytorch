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

#ifndef __PULGIN_NATIVE_NPU_UTILS_NUP_UTILS__
#define __PULGIN_NATIVE_NPU_UTILS_NUP_UTILS__

#include <stdint.h>
#include <string>
#include <vector>
#include <ATen/ATen.h>
#include "torch_npu/csrc/core/npu/npu_log.h"

#include "third_party/acl/inc/ge/ge_error_codes.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_op.h"

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"

using std::string;
using std::vector;

namespace at_npu
{
  namespace native
  {

    // smallvector max size
    const int N = 32;
    // npu tensor max size
    const int SHAPE_SIZE = 8;
    // HALF_MAX and HALF_MIN of NPU support
    const int NPU_HALF_MAX = 65504;
    const int NPU_HALF_MIN = -65504;
    const int NPU_MAX_OP_EXEC_TRY_NUM = 2;

    typedef enum CompileType
    {
      MEMORY_HOST_COMPILE_DEPENDENT = 1,
      MEMORY_HOST_COMPILE_INDEPENDENT = 2,
    } CompileType;

    class NpuUtils
    {
    public:
      static bool check_match(const at::Tensor *tensor);
      static at::Tensor format_contiguous(const at::Tensor &src);
      static at::Tensor format_contiguous_add_copy_optimize(const at::Tensor &src);
      static void RefreshFormat(const at::Tensor &tensor);
      static void format_fresh_view(
          at::Tensor &x,
          const at::Tensor &y);

      static bool check_5d_5d_match(const at::Tensor &tensor);
      static bool IsOomError(aclError ret, int index);
      static void check_1d(const at::Tensor &t, const char *arg, const char *fn);
    };
    const std::string AclDateTypeToString(aclDataType descDType);
    const std::string AclFormatToString(aclFormat descFormat);
  } // namespace native
} // namespace at_npu

#endif // __NATIVE_NPU_UTILS_NUP_UTILS__
