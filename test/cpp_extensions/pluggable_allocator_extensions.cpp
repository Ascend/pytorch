#include <sys/types.h>
#include <iostream>
#include <torch/extension.h>

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"

extern "C" {
static bool useflag = false;

void* my_malloc(ssize_t size, int device, aclrtStream stream)
{
    void *ptr;
    aclrtMallocAlign32(&ptr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
    std::cout<<"alloc ptr = "<<ptr<<", size = "<<size<<std::endl;
    useflag = true;
    return ptr;
}

void my_free(void* ptr, ssize_t size, int device, aclrtStream stream)
{
    std::cout<<"free ptr = "<<ptr<<std::endl;
    aclrtFree(ptr);
}

bool check_custom_allocator_used()
{
    return useflag;
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_malloc", &my_malloc, "");
    m.def("my_free", &my_free, "");
    m.def("check_custom_allocator_used", &check_custom_allocator_used, "");
}
