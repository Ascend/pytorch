// Copyright (c) 2022 Huawei Technologies Co., Ltd
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

#include <torch/library.h>
#include <torch/script.h>

#include "torch_npu/library_npu.h"


int main(int argc, char*argv[]) {
    if (1 == argc || argc > 2) {
        TORCH_CHECK(false, "Please input the model name!")
    }

    if (!argv[1]) {
        TORCH_CHECK(false, "Got invalid model name!")
    }
    std::string pt_model_path = argv[1];

    // init device
    auto device = at::Device(at_npu::key::NativeDeviceType);
    torch_npu::init_npu(device);

    auto input_tensor = torch::rand({2, 3, 4, 4}).to(device);
    auto result = input_tensor + input_tensor;
    std::cout << "add result: " << result << std::endl;

    
    torch::jit::script::Module script_model = torch::jit::load(pt_model_path);
    script_model.to(device); // move pt model  to npu device
    auto inputs = torch::rand({4, 3, 4, 4}).to(device);
    auto model_out = script_model.forward({inputs});
    std::cout << "model output:" << model_out << std::endl;

    // finalize npu device
    torch_npu::finalize_npu();
}
