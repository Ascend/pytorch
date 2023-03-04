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
#include "third_party/acl/inc/op_proto/data_flow_ops.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/graph/util/TdtChannelForPrint.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace {
at_npu::native::DynamicInputRegFunc outfeedenque_func =
    [](at_npu::native::DyNumAndIndex num_and_index,
       std::string op_name) -> ge::OperatorPtr {
  auto ge_op = std::make_shared<ge::op::OutfeedEnqueueOpV2>(op_name.c_str());
  ge_op->create_dynamic_input_byindex_x(num_and_index.front().first,
                                        num_and_index.front().second);
  return ge_op;
};
}
namespace at_npu {
namespace native {
void NPUNativeFunctions::npu_enque_tensor(at::TensorList tensors,
                                          string tensor_name,
                                          int64_t capacity) {
  OpCommand cmd;
  cmd.Name("OutfeedEnqueueOpV2");
  size_t input_num = tensors.size();
  for (size_t i = 0UL; i < input_num; i++) {
    string input_name = "x" + std::to_string(i);
    cmd.InputWithMetaInfo(tensors[i], input_name, tensor_name);
  }

  std::string channel_name =
      at_npu::native::TdtChannelForPrint::GetInstance().GetChannelName(capacity);
  TORCH_CHECK(!channel_name.empty(), "Get channel for npu enque tensor failed");
  cmd.DynamicInputReg(outfeedenque_func, {{input_num, 0}})
      .Input(tensor_name)
      .Attr("channel_name", channel_name)
      .Run();
}
}  // namespace native
}  // namespace at_npu