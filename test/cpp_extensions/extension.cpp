#include <thread>
#include <chrono>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/OpHook.h"
// test   in  .setup with relative path
#include <tmp.h>

using namespace at;

static int g_op_hook_call_count = 0;    // test op_hook

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

Tensor blocking_ops(Tensor x)
{
    auto blocking_call = []() -> int {
        std::this_thread::sleep_for(std::chrono::seconds(180));
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("blocking_ops", blocking_call);

    return x;
}

void register_op_hook()
{
    at_npu::native::RegisterOpHookBeginFn(
        [](const std::string &op_name) -> void {
            g_op_hook_call_count++;
    });
    at_npu::native::RegisterOpHookPreFn([](const at::Tensor &at_tensor) -> void {
        if (!at_tensor.defined()) {
            return;
        }
        g_op_hook_call_count++;
    });
    at_npu::native::RegisterOpHookPostFn([](const at::Tensor &at_tensor) -> void {
        if (!at_tensor.defined()) {
            return;
        }
        g_op_hook_call_count++;
    });
    at_npu::native::RegisterOpHookEndFn([]() -> void {
        g_op_hook_call_count++;
    });
}

int get_op_hook_call_count()
{
    return g_op_hook_call_count;
}

void reset_op_hook_call_count()
{
    g_op_hook_call_count = 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tanh_add", &tanh_add, "tanh(x) + tanh(y)");
    m.def("npu_add", &npu_add, "x + y");
    m.def("check_storage_sizes", &check_storage_sizes, "check_storage_sizes");
    m.def("blocking_ops", &blocking_ops, "blocking_ops");
    m.def("register_op_hook", &register_op_hook, "register_op_hook");
    m.def("get_op_hook_call_count", &get_op_hook_call_count, "get_op_hook_call_count");
    m.def("reset_op_hook_call_count", &reset_op_hook_call_count, "reset_op_hook_call_count");
}
