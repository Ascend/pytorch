#include <sys/types.h>
#include <atomic>
#include <iostream>
#include <torch/extension.h>

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

extern "C" {
using c10_npu::NPUCachingAllocator::DeviceStats;
static bool useflag = false;
static std::atomic<int> alloc_count{0};
static std::atomic<int> free_count{0};

void* my_malloc(ssize_t size, int device, aclrtStream stream)
{
    void *ptr;
    aclrtMallocAlign32(&ptr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
    std::cout<<"alloc ptr = "<<ptr<<", size = "<<size<<std::endl;
    useflag = true;
    alloc_count++;
    return ptr;
}

void my_free(void* ptr, ssize_t size, int device, aclrtStream stream)
{
    std::cout<<"free ptr = "<<ptr<<std::endl;
    aclrtFree(ptr);
    free_count++;
}

bool check_custom_allocator_used()
{
    return useflag;
}

int get_alloc_count()
{
    return alloc_count.load();
}

int get_free_count()
{
    return free_count.load();
}

void reset_alloc_free_count()
{
    alloc_count.store(0);
    free_count.store(0);
}

DeviceStats my_get_device_stats(c10::DeviceIndex device)
{
    DeviceStats stats;
    return stats;
}

void my_reset_peak_status(c10::DeviceIndex device)
{
    std::cout<<"resetPeakStatus success!"<<std::endl;
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_malloc", &my_malloc, "");
    m.def("my_free", &my_free, "");
    m.def("check_custom_allocator_used", &check_custom_allocator_used, "");
    m.def("get_alloc_count", &get_alloc_count, "");
    m.def("get_free_count", &get_free_count, "");
    m.def("reset_alloc_free_count", &reset_alloc_free_count, "");
}
