// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at_npu
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <c10/npu/NPUStream.h>
#include <c10/npu/NPUException.h>

#include "torch_npu/csrc/framework/utils/NpuProfilingDispatch.h"

namespace at_npu
{
  namespace native
  {

    NpuProfilingDispatch &NpuProfilingDispatch::Instance()
    {
      static NpuProfilingDispatch npuProfilingDispatch;
      return npuProfilingDispatch;
    }

    void NpuProfilingDispatch::init()
    {
      profStepInfo = c10::npu::acl::init_stepinfo();
    }

    void NpuProfilingDispatch::start()
    {
      this->init();
      auto stream = c10::npu::getCurrentNPUStream();
      auto ret = c10::npu::acl::start_deliver_op(
          profStepInfo,
          aclprofStepTag::ACL_STEP_START,
          stream);
      if (ret != ACL_ERROR_NONE)
      {
        NPU_LOGE("npu profiling start fail, error code: %d", ret);
        C10_NPU_SHOW_ERR_MSG();
      }
    }

    void NpuProfilingDispatch::stop()
    {
      auto stream = c10::npu::getCurrentNPUStream();
      auto ret = c10::npu::acl::stop_deliver_op(
          profStepInfo,
          aclprofStepTag::ACL_STEP_END,
          stream);
      if (ret != ACL_ERROR_NONE)
      {
        NPU_LOGE("npu profiling stop fail, error code: %d", ret);
        C10_NPU_SHOW_ERR_MSG();
      }
      this->destroy();
    }

    void NpuProfilingDispatch::destroy()
    {
      if (profStepInfo != nullptr)
      {
        c10::npu::acl::destroy_stepinfo(profStepInfo);
      }
    }

  }
}
