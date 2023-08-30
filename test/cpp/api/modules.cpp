#include <gtest/gtest.h>
#include <c10/util/irange.h>

#include <torch/torch.h>
#include "torch_npu/csrc/libs/torch_npu.h"

namespace F = torch::nn::functional;
using namespace torch::nn;

auto device = at::Device("npu:0");

TEST(FunctionalTest, Conv1d) {
  torch_npu::init_npu(device);
  auto x = torch::arange(30, torch::dtype(torch::kFloat16).requires_grad(true))
               .reshape({2, 3, 5}).to(device);
  auto weight =
      torch::arange(18, torch::dtype(torch::kFloat16).requires_grad(true))
          .reshape({2, 3, 3}).to(device);

  auto y = F::conv1d(x, weight, F::Conv1dFuncOptions().stride(1));
  auto expected = torch::tensor(
      {{{312., 348., 384.}, {798., 915., 1032.}},

       {{852., 888., 924.}, {2552., 2670., 2788.}}},
      torch::kFloat16).to(device);

  ASSERT_TRUE(torch::allclose(y, expected));
  torch_npu::finalize_npu();
}
