#include <Python.h>
#include <link.h>

#include <filesystem>
#include <iostream>
#include <vector>
#include <torch/torch.h>

#include "torch_npu/torch_npu.h"

namespace {
std::vector<std::filesystem::path> loaded_cores()
{
    std::vector<std::filesystem::path> cores;
    dl_iterate_phdr(
        [](struct dl_phdr_info* info, size_t, void* data) {
            const std::filesystem::path path(info->dlpi_name);
            if (path.filename() == "libtorch_npu.so") {
                static_cast<std::vector<std::filesystem::path>*>(data)->push_back(path);
            }
            return 0;
        },
        &cores);
    return cores;
}

void check_cpp_add()
{
    const auto input = torch::tensor({1, 2}).to(torch::Device("npu:0"));
    TORCH_CHECK(at::equal(at::add(input, input).cpu(), torch::tensor({2, 4})), "C++ NPU add mismatch");
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: mixed_wheel <standalone-libtorch_npu.so>\n";
        return 2;
    }
    try {
        const auto cores = loaded_cores();
        TORCH_CHECK(cores.size() == 1 && std::filesystem::equivalent(cores.front(), argv[1]),
                    "Expected the standalone NPU core before Python initialization");
        torch_npu::init_npu("npu:0");
        check_cpp_add();

        Py_Initialize();
        const int status = PyRun_SimpleString(
            "import torch_npu\n"
            "import torch\n"
            "x = torch.tensor([1, 2], device='npu:0')\n"
            "if (x + x).cpu().tolist() != [2, 4]:\n"
            "    raise RuntimeError('Python NPU add mismatch')\n");
        TORCH_CHECK(status == 0, "Python wheel import or NPU add failed");
        TORCH_CHECK(loaded_cores() == cores, "Importing the wheel changed the loaded NPU core set");
        std::cout << "PASS: Python wheel and C++ NPU add share the standalone core" << std::endl;
        return Py_FinalizeEx() == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
