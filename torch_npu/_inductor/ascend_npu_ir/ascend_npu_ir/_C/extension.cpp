#include <iostream>
#include <fstream>
#include <pybind11/pybind11.h>

#include "hacl_rt.h"

namespace py = pybind11;
static uint64_t kBiShengStartAddr = 0xbadbeef;

void ReadBinFile(const char* file_name, uint32_t* fileSize, char** buffer)
{
    std::filebuf* pbuf;
    std::ifstream filestr;
    size_t size;
    filestr.open(file_name, std::ios::binary);
    if (!filestr) {
        printf("open file failed!");
        throw std::runtime_error("open file failed!\n");
    }
    pbuf = filestr.rdbuf();
    size = pbuf->pubseekoff(0, std::ios::end, std::ios::in);
    pbuf->pubseekpos(0, std::ios::in);
       
    *buffer = new char[size];
    if (NULL == *buffer) {
        printf("cannot malloc buffer size\n");
        throw std::runtime_error("cannot malloc buffer size");
    }
    pbuf->sgetn(*buffer, size);
    *fileSize = size;

    filestr.close();
}

const uintptr_t RegisterBinaryKernel(const char* func_name, const char* bin_file, char* buffer)
{
    rtDevBinary_t binary;
    void* binHandle = NULL;
    uint32_t bufferSize = 0;
    ReadBinFile(bin_file, &bufferSize, &buffer);
    if (NULL == buffer) {
        printf("ReadBinFile failed\n");
        return reinterpret_cast<uintptr_t>(nullptr);
    }
    binary.data = buffer;
    binary.length = bufferSize;
    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    binary.version = 0;
    rtError_t rtRet = rtDevBinaryRegister(&binary, &binHandle);
    if (rtRet != RT_ERROR_NONE) {
        printf("rtDevBinaryRegister failed!\n");
        return reinterpret_cast<uintptr_t>(nullptr);
    }
    kBiShengStartAddr += 1;
    rtRet = rtFunctionRegister(binHandle, reinterpret_cast<void *>(kBiShengStartAddr), func_name, (void*)func_name, 0);
    if (rtRet != RT_ERROR_NONE) {
        printf("rtFunctionRegister failed!\n");
        return reinterpret_cast<const uintptr_t>(nullptr);
    }
    return reinterpret_cast<const uintptr_t> (kBiShengStartAddr);
}

PYBIND11_MODULE(_C, m) {
    m.def("load_kernel_binary", [](const char* func_name, const char* bin_file){
        char* buffer = nullptr;
        const uintptr_t kBiShengStartAddr = RegisterBinaryKernel(func_name, bin_file, buffer);
        delete buffer;
        return kBiShengStartAddr;
    });
}