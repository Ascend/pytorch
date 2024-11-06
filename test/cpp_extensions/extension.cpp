#include <thread>
#include <chrono>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/aten/common/from_blob.h"
#include "torch_npu/csrc/framework/OpCommand.h"
// test   in  .setup with relative path
#include <tmp.h>

using namespace at;

Tensor tanh_add(Tensor x, Tensor y)
{
    return x.tanh() + y.tanh();
}

Tensor npu_add(const Tensor &self_, const Tensor &other_)
{
    TORCH_INTERNAL_ASSERT(self_.device().type() == c10::DeviceType::PrivateUse1);
    TORCH_INTERNAL_ASSERT(other_.device().type() == c10::DeviceType::PrivateUse1);
    return at::add(self_, other_, 1);
}

bool check_storage_sizes(const Tensor &tensor, const c10::IntArrayRef &sizes)
{
    auto tensor_sizes = at_npu::native::get_npu_storage_sizes(tensor);
    if (tensor_sizes.size() == sizes.size()) {
        return std::equal(tensor_sizes.begin(), tensor_sizes.end(), sizes.begin());
    }
    return false;
}

bool check_from_blob()
{
    auto data = torch::tensor({1.0, 2.0, 3.0}, torch::kFloat).to(at::Device("npu:0"));
    auto tensor = at_npu::native::from_blob(data.data_ptr(), data.sizes(), torch::dtype(torch::kFloat));

    bool dtype_same = (tensor.dtype() == torch::kFloat);
    bool num_same = (tensor.numel() == 3);
    bool pos1_same = (tensor[0].item<float>() == 1);
    bool pos2_same = (tensor[1].item<float>() == 2);
    bool pos3_same = (tensor[2].item<float>() == 3);
    tensor = tensor -1;
    bool sub_same = ((tensor[2].item<float>() == 2));
    return dtype_same && num_same && pos1_same && pos2_same && pos3_same && sub_same;
}

bool check_from_blob_strides()
{
    auto data = torch::tensor({1, 2, 3, 4, 5, 6, 7, 8, 9}, torch::kInt32).to(at::Device("npu:0"));
    auto tensor = at_npu::native::from_blob(data.data_ptr(), {3, 3}, {1, 3}, torch::kInt32);  // sizes = {3,3}, strides = {1,3}

    bool dtype_same = (tensor.dtype() == torch::kInt32);
    bool num_same = (tensor.numel() == data.numel());
    const std::vector<int64_t> expected_strides = {1, 3};
    auto result_strides = tensor.strides();
    bool stride_same = std::equal(result_strides.begin(), result_strides.end(), expected_strides.begin());
    bool pos_same = true;
    for (const auto i : c10::irange(tensor.size(0))) {
    for (const auto j : c10::irange(tensor.size(1))) {
        // NOTE: This is column major because the strides are swapped.
        if (tensor[i][j].item<int32_t>() != (1 + (j * tensor.size(1)) + i))
            pos_same = false;
        }
    }
    auto tensor_clone = tensor.clone();
    bool clone_same = at::equal(tensor_clone, tensor);
    auto tensor_add = tensor + 1;
    bool add_same = true;
    for (const auto i : c10::irange(tensor_add.size(0))) {
    for (const auto j : c10::irange(tensor_add.size(1))) {
        // NOTE: This is column major because the strides are swapped.
        if (tensor_add[i][j].item<int32_t>() != (2 + (j * tensor_add.size(1)) + i))
            add_same = false;
        }
    }
    return dtype_same && num_same && pos_same && stride_same && clone_same && add_same;
}

Tensor blocking_ops(Tensor x)
{
    auto blocking_call = []() -> int {
        std::this_thread::sleep_for(std::chrono::seconds(180));
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("blocking_ops", blocking_call);

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tanh_add", &tanh_add, "tanh(x) + tanh(y)");
    m.def("npu_add", &npu_add, "x + y");
    m.def("check_storage_sizes", &check_storage_sizes, "check_storage_sizes");
    m.def("check_from_blob", &check_from_blob, "check_from_blob");
    m.def("check_from_blob_strides", &check_from_blob_strides, "check_from_blob_strides");
    m.def("blocking_ops", &blocking_ops, "blocking_ops");
}
