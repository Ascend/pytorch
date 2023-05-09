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

    // init device, device_id:0
    torch_npu::init_npu(0);
    // now, we must use "xla" not "npu"
    auto device = at::Device("xla:0");

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
