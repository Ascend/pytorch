#include "triton_runtime.h"
#include <torch/torch.h>
#include <iostream>
#include <torch_npu/csrc/core/npu/NPUStream.h>

using namespace triton_runtime;

void run_add_kernel() {
    auto& rt = TritonRuntime::instance();

    // register kernel
    auto s = rt.register_kernel("examples/my_kernels.py", "add_kernel");
    if (!s.ok()) {
        std::cerr << "Register failed: " << s.error_message() << std::endl;
        return;
    }

    rt.print_kernel_signature("add_kernel");

    // Prepare data on NPU
    auto x = torch::rand({1000}, torch::kFloat32).to("npu");
    auto y = torch::rand({1000}, torch::kFloat32).to("npu");
    auto out = torch::empty_like(x);
    int n_elements = 1000;
    int BLOCK_SIZE = 1024;

    auto grid = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch Triton kernel
    s = rt.run("add_kernel", Grid(grid, 1, 1), x, y, out, n_elements, BLOCK_SIZE);
    if (!s.ok()) {
        std::cerr << "Launch failed: " << s.error_message() << std::endl;
        return;
    }

    // validate result
    auto libtorch_result = torch::add(x, y);
    std::cout << "C++ libtorch add[0:5]: " << libtorch_result.slice(0, 0, 5) << std::endl;
    std::cout << "C++ triton  out[0:5]: " << out.slice(0, 0, 5) << std::endl;
    auto diff = (libtorch_result - out).abs().max().item<float>();
    std::cout << "C++: max diff: " << diff << std::endl;
    std::cout << "C++: match: " << (diff < 1e-5 ? "YES" : "NO") << std::endl;
}

// C++ lambda grid — like Triton's lambda meta: (...) but resolved entirely
// in C++ without Python involvement.
void run_add_kernel_lambda_grid() {
    auto& rt = TritonRuntime::instance();

    auto s = rt.register_kernel("examples/my_kernels.py", "add_kernel");
    if (!s.ok()) {
        std::cerr << "Register failed: " << s.error_message() << std::endl;
        return;
    }

    auto x = torch::rand({1000}, torch::kFloat32).to("npu");
    auto y = torch::rand({1000}, torch::kFloat32).to("npu");
    auto out = torch::empty_like(x);
    int n_elements = 1000;
    int BLOCK_SIZE = 1024;

    // meta["key"] returns variant<int32_t,float,bool>, same as Triton's bound_args.
    GridFn grid_fn = [](BoundArgs& meta) -> Grid {
        int n  = meta["n_elements"];
        int bs = meta["BLOCK_SIZE"];
        return Grid((n + bs - 1) / bs, 1, 1);
    };

    s = rt.run("add_kernel", Grid(grid_fn), x, y, out,
               n_elements, BLOCK_SIZE);
    if (!s.ok()) {
        std::cerr << "Launch failed: " << s.error_message() << std::endl;
        return;
    }

    auto libtorch_result = torch::add(x, y);
    std::cout << "C++ libtorch add[0:5]: " << libtorch_result.slice(0, 0, 5) << std::endl;
    std::cout << "C++ triton  out[0:5]: " << out.slice(0, 0, 5) << std::endl;
    auto diff = (libtorch_result - out).abs().max().item<float>();
    std::cout << "C++: max diff: " << diff << std::endl;
    std::cout << "C++: match: " << (diff < 1e-5 ? "YES" : "NO") << std::endl;
}

void run_layer_norm_kernel() {
    auto& rt = TritonRuntime::instance();

    // register kernel
    auto s = rt.register_kernel("examples/my_kernels.py", "layer_norm_kernel");
    if (!s.ok()) {
        std::cerr << "Register failed: " << s.error_message() << std::endl;
        return;
    }

    rt.print_kernel_signature("layer_norm_kernel");

    // Prepare data on NPU
    int n_rows = 128;
    int n_cols = 512;
    float eps = 1e-5;
    auto x = torch::randn({n_rows, n_cols}, torch::kFloat32).to("npu");
    auto weight = torch::ones({n_cols}, torch::kFloat32).to("npu");
    auto bias = torch::zeros({n_cols}, torch::kFloat32).to("npu");
    auto out = torch::empty_like(x);
    int64_t stride = x.stride(0);

    // BLOCK_SIZE = next_power_of_2(n_cols)
    int BLOCK_SIZE = 1;
    while (BLOCK_SIZE < n_cols) BLOCK_SIZE *= 2;

    std::cout << "C++ BLOCK_SIZE: " << BLOCK_SIZE << std::endl;

    // Launch Triton kernel
    s = rt.run("layer_norm_kernel", Grid(n_rows, 1, 1),
               out, x, weight, bias,
               n_rows, n_cols, stride, eps,
               BLOCK_SIZE);
    if (!s.ok()) {
        std::cerr << "Launch failed: " << s.error_message() << std::endl;
        return;
    }

    // validate result
    auto ref = torch::layer_norm(x, {n_cols}, weight, bias, eps);
    std::cout << "C++ triton  out[0,:5]: " << out.index({0, torch::indexing::Slice(0, 5)}) << std::endl;
    std::cout << "C++ torch   ref[0,:5]: " << ref.index({0, torch::indexing::Slice(0, 5)}) << std::endl;
    auto diff = (ref - out).abs().max().item<float>();
    std::cout << "C++: max diff: " << diff << std::endl;
    std::cout << "C++: match: " << (diff < 1e-3 ? "YES" : "NO") << std::endl;
}

int main() {
    // List registered kernels
    auto& rt = TritonRuntime::instance();

    // simple case — explicit grid
    std::cout << "=== Add Kernel (explicit grid) ===" << std::endl;
    run_add_kernel();


    // lambda grid — C++ callable, resolves without Python
    std::cout << "\n=== Add Kernel (C++ lambda grid) ===" << std::endl;
    run_add_kernel_lambda_grid();

    for (const auto& name : rt.list_kernels()) {
        std::cout << "  registered: " << name << std::endl;
    }


    // complicate case
    std::cout << "\n=== LayerNorm Kernel ===" << std::endl;
    run_layer_norm_kernel();

    for (const auto& name : rt.list_kernels()) {
        std::cout << "  registered: " << name << std::endl;
    }

    return 0;
}
