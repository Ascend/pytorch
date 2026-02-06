#ifndef BUILD_LIBTORCH
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>


#include <torch/csrc/THP.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/inductor/mlir/hacl_rt.h"

namespace {
static uint64_t kBiShengStartAddr = 0xbadbeef;

void ReadBinFile(const char* file_name, uint32_t& fileSize, char*& buffer)
{
    std::filebuf* pbuf;
    std::ifstream filestr;
    size_t size;
    filestr.open(file_name, std::ios::binary);
    TORCH_CHECK(filestr, "open file failed!");
    pbuf = filestr.rdbuf();
    const std::streamoff end_pos = pbuf->pubseekoff(0, std::ios::end, std::ios::in);
    pbuf->pubseekpos(0, std::ios::in);

    TORCH_CHECK(end_pos > 0 && end_pos <= static_cast<std::streamoff>(std::numeric_limits<uint32_t>::max()),
                "invalid file size");

    size = static_cast<size_t>(end_pos);
    buffer = new char[size];
    TORCH_CHECK(buffer != nullptr, "cannot malloc buffer size");
    pbuf->sgetn(buffer, static_cast<std::streamsize>(size));
    fileSize = static_cast<uint32_t>(size);

    filestr.close();
}

const uintptr_t RegisterBinaryKernel(const char* func_name, const char* bin_file, char* buffer)
{
    rtDevBinary_t binary;
    void* binHandle = nullptr;
    uint32_t bufferSize = 0;
    ReadBinFile(bin_file, bufferSize, buffer);
    TORCH_CHECK(buffer != nullptr, "ReadBinFile failed");
    binary.data = buffer;
    binary.length = bufferSize;
    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    binary.version = 0;
    rtError_t rtRet = rtDevBinaryRegister(&binary, &binHandle);
    TORCH_CHECK(rtRet == RT_ERROR_NONE, "rtDevBinaryRegister failed!");
    kBiShengStartAddr += 1;
    rtRet = rtFunctionRegister(binHandle, reinterpret_cast<void*>(kBiShengStartAddr), func_name, (void*)func_name, 0);
    TORCH_CHECK(rtRet == RT_ERROR_NONE, "rtFunctionRegister failed!");
    return reinterpret_cast<const uintptr_t>(kBiShengStartAddr);
}
} // namespace

void TORCH_NPU_API THNPMLIR_init(PyObject* module)
{
    auto torch_C_m = py::handle(module).cast<py::module>();
    auto mlir_m = torch_C_m.def_submodule("mlir", "MLIR bindings");
    mlir_m.def("load_kernel_binary", [](const char* func_name, const char* bin_file) {
        char* buffer = nullptr;
        const uintptr_t addr = RegisterBinaryKernel(func_name, bin_file, buffer);
        delete[] buffer;
        return addr;
    });
}
#endif
