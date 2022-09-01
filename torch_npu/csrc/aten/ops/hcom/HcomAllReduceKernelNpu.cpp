// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu{
namespace native{
    
using namespace at_npu::native;

at::Tensor NPUNativeFunctions::npu_hcom_allreduce(const at::Tensor& self,
    std::string reduction,
    std::string group,
    int64_t fusion,
    int64_t fusion_id,
    double alpha,
    double beta,
    c10::optional<int64_t> hccl_comm) {
    AT_ERROR("npu_hcom_allreduce is not implemented for Tensor");
}

at::Tensor& NPUNativeFunctions::npu_hcom_allreduce_out(const at::Tensor& self,
    std::string reduction,
    std::string group,
    int64_t fusion,
    int64_t fusion_id,
    double alpha,
    double beta,
    c10::optional<int64_t> hccl_comm,
    at::Tensor& out) {
    OpCommand cmd;
    cmd.Name("HcomAllReduce")
        .Input(self)
        .Attr("reduction", reduction)
        .Attr("group", group)
        .Attr("fusion", fusion)
        .Attr("fusion_id", fusion_id)
        .Attr("alpha", static_cast<float>(alpha))
        .Attr("beta", static_cast<float>(beta));
    if (hccl_comm.has_value()) {
        cmd.Attr("comm", hccl_comm.value());
    }
    cmd.Output(out)
        .Run();
    return out;
}

} // namespace native
} // namespace at_npu