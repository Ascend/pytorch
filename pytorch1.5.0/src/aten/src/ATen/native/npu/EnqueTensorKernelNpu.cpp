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
#include <c10/npu/OptionsManager.h>
#include <ATen/native/npu/graph/util/TdtChannelForPrint.h>
#include <ATen/native/npu/utils/OpAdapter.h>
#include <ATen/native/npu/utils/CalcuOpUtil.h>
#include <third_party/acl/inc/op_proto/data_flow_ops.h>
namespace {
at::native::npu::DynamicInputRegFunc outfeedenque_func =
  [] (c10::npu::graph::DyNumAndIndex num_and_index,
      std::string op_name) -> ge::OperatorPtr {
    auto ge_op =
      std::make_shared<ge::op::OutfeedEnqueueOpV2>(op_name.c_str());
    ge_op->create_dynamic_input_byindex_x(num_and_index.front().first,
                                          num_and_index.front().second);
    return ge_op;
  };
}
namespace at {
namespace native {
using namespace at::native::npu;
void enque_tensor_npu(TensorList tensors, string tensor_name) {
  OpCommand cmd;
  cmd.Name("OutfeedEnqueueOpV2");
  size_t input_num = tensors.size();
  for (size_t i = 0UL; i < input_num; i++) {
    string input_name = "x" + to_string(i);
    cmd.Input(tensors[i], input_name);
  }
  
  std::string channel_name = at::native::npu::TdtChannelForPrint::GetInstance().GetChannelName();
  TORCH_CHECK(!channel_name.empty(), "Get channel for npu enque tensor failed");
  cmd.DynamicInputReg(outfeedenque_func, {{input_num, 0}})
     .Input(tensor_name)
     .Attr("channel_name", channel_name)
     .Run();
}
}    
}