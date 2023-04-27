// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <torch/script.h>

#include <torch_npu/library_npu.h>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    TORCH_CHECK(false, "Please input the model name!");
  }

  if (!argv[1]) {
    TORCH_CHECK(false, "Got invalid model name!");
  }

  std::cout << "module: " << argv[1] << std::endl;

  // init npu
  auto device = at::Device(at_npu::key::NativeDeviceType);
  torch_npu::init_npu(device);

  // load model
  torch::jit::script::Module module = torch::jit::load(argv[1]);

  module.to(device);

  // run model
  torch::jit::setGraphExecutorOptimize(false);
  std::vector<torch::jit::IValue> input_tensor;
  input_tensor.push_back(torch::randn({1, 3, 244, 244}).to(device));
  at::Tensor output = module.forward(input_tensor).toTensor();

  // finalize npu
  torch_npu::finalize_npu();
  std::cout << "resnet_model run success!" << std::endl;

  return 0;
}
