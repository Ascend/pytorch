#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <torch/script.h>

#include <torch_npu/torch_npu.h>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    TORCH_CHECK(false, "Please input the model name!");
  }

  if (!argv[1]) {
    TORCH_CHECK(false, "Got invalid model name!");
  }

  std::cout << "module: " << argv[1] << std::endl;

  // init npu
  torch_npu::init_npu("npu:0");
  auto device = at::Device("npu:0");

  // load model
  torch::jit::script::Module module = torch::jit::load(argv[1]);

  module.to(device);

  // run model
  torch::jit::setGraphExecutorOptimize(false);
  std::vector<torch::jit::IValue> input_tensor;
  input_tensor.push_back(torch::randn({1, 3, 244, 244}).to(device));
  at::Tensor output = module.forward(input_tensor).toTensor();

  std::cout << output.slice(1, 0, 5) << std::endl;
  std::cout << "resnet_model run success!" << std::endl;

  torch_npu::finalize_npu();
  return 0;
}
