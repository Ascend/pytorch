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

#include "GlobalStep.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include <third_party/acl/inc/acl/acl_op_compiler.h>

namespace at {
namespace native {

GlobalStep& GlobalStep::Instance() {
    static GlobalStep globalStep(0, 1);
    return globalStep;
}

void GlobalStep::GlobalStepInc() {
     GLOBAL_STEP++;
}

int64_t GlobalStep::GetGlobalStep() const {
    return GLOBAL_STEP;
}

void GlobalStep::SetStartFuzzCompileStep(const int64_t step) {
    START_FUZZ_COMPILE_STEP = step;
}

int64_t GlobalStep::GetStartFuzzCompileStep() const {
    return START_FUZZ_COMPILE_STEP;
}

TORCH_NPU_API bool check_fuzz_enable(){
    int64_t globalstep = GlobalStep::Instance().GetGlobalStep();
    int64_t globalstartstep = GlobalStep::Instance().GetStartFuzzCompileStep();

    return (globalstep >= globalstartstep);
}

void global_step_inc() {
    #ifdef USE_NPU
    GlobalStep::Instance().GlobalStepInc();
    // To invoke the interface only once, check whether the GLOBAL_STEP equal to START_FUZZ_COMPILE_STEP is OK.
    if(GlobalStep::Instance().GetGlobalStep() == GlobalStep::Instance().GetStartFuzzCompileStep()) {
        NPU_LOGD("GLOBAL_STEP = %ld, START_FUZZ_COMPILE_STEP = %ld, start fuzz compile!", 
        GlobalStep::Instance().GetGlobalStep(), GlobalStep::Instance().GetStartFuzzCompileStep());
        
        aclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_FUZZ);
    }
    #endif
}

void set_start_fuzz_compile_step(int64_t step) {
    #ifdef USE_NPU
    GlobalStep::Instance().SetStartFuzzCompileStep(step);
    #endif
}

}
}
